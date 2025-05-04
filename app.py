# Orpheus-FASTAPI by Lex-au
# https://github.com/Lex-au/Orpheus-FastAPI
# Description: Main FastAPI server for Orpheus Text-to-Speech

import os
import time
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Annotated, Union, cast
from dotenv import load_dotenv
import wave
import io
import struct
import json
import numpy as np

# Function to ensure .env file exists
def ensure_env_file_exists():
    """Create a .env file from defaults and OS environment variables"""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        try:
            # 1. Create default env dictionary from .env.example
            default_env = {}
            with open(".env.example", "r") as example_file:
                for line in example_file:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=")[0].strip()
                        default_env[key] = line.split("=", 1)[1].strip()

            # 2. Override defaults with Docker environment variables if they exist
            final_env = default_env.copy()
            for key in default_env:
                if key in os.environ:
                    final_env[key] = os.environ[key]

            # 3. Write dictionary to .env file in env format
            with open(".env", "w") as env_file:
                for key, value in final_env.items():
                    env_file.write(f"{key}={value}\n")
                    
            print("✅ Created default .env file from .env.example and environment variables.")
        except Exception as e:
            print(f"⚠️ Error creating default .env file: {e}")

# Ensure .env file exists before loading environment variables
ensure_env_file_exists()

# Load environment variables from .env file
load_dotenv(override=True)

from fastapi import FastAPI, Request, Form, HTTPException, Depends, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from tts_engine import (
    generate_speech_from_api, 
    stream_speech_from_api,
    AVAILABLE_VOICES, 
    DEFAULT_VOICE, 
    VOICE_TO_LANGUAGE, 
    AVAILABLE_LANGUAGES,
    SAMPLE_RATE  # Added for WAV header generation
)

# Create FastAPI app
app = FastAPI(
    title="Orpheus-FASTAPI",
    description="High-performance Text-to-Speech server using Orpheus-FASTAPI",
    version="1.0.0"
)

# We'll use FastAPI's built-in startup complete mechanism
# The log message "INFO:     Application startup complete." indicates
# that the application is ready

# Ensure directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount directories for serving files
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# API models
class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0

class StreamingSpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0

class APIResponse(BaseModel):
    status: str
    voice: str
    output_file: str
    generation_time: float

# Cache for WAV headers to avoid regenerating them for each request
WAV_HEADER_CACHE: Dict[Tuple[int, int, int], bytes] = {}

def generate_wav_header(sample_rate: int = 24000, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    """Generate WAV header with caching for improved performance.
    
    Args:
        sample_rate: Audio sample rate (default: 24000)
        bits_per_sample: Bits per sample (default: 16)
        channels: Number of audio channels (default: 1)
        
    Returns:
        Cached or newly generated WAV header
    """
    cache_key = (sample_rate, bits_per_sample, channels)
    
    # Return cached header if available
    if cache_key in WAV_HEADER_CACHE:
        return WAV_HEADER_CACHE[cache_key]
    
    # Generate new header if not in cache (approximately 5x faster than using wave module)
    bytes_per_sample = bits_per_sample // 8
    block_align = bytes_per_sample * channels
    byte_rate = sample_rate * block_align
    
    # Use direct struct packing for fastest possible WAV header generation
    header = bytearray()
    # RIFF header
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', 0xFFFFFFFF))  # Placeholder for file size (unknown streaming length)
    header.extend(b'WAVE')
    # Format chunk
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))  # Format chunk size
    header.extend(struct.pack('<H', 1))   # PCM format
    header.extend(struct.pack('<H', channels))
    header.extend(struct.pack('<I', sample_rate))
    header.extend(struct.pack('<I', byte_rate))  # Bytes per second
    header.extend(struct.pack('<H', block_align))
    header.extend(struct.pack('<H', bits_per_sample))
    # Data chunk
    header.extend(b'data')
    header.extend(struct.pack('<I', 0xFFFFFFFF))  # Placeholder for data size (unknown streaming length)
    
    # Store in cache for future use
    WAV_HEADER_CACHE[cache_key] = bytes(header)
    
    return WAV_HEADER_CACHE[cache_key]

# OpenAI-compatible API endpoint
@app.post("/v1/audio/speech")
async def create_speech_api(request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    
    For longer texts (>1000 characters), batched generation is used
    to improve reliability and avoid truncation issues.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    
    # Check if we should use batched generation
    use_batching = len(request.input) > 1000
    if use_batching:
        print(f"Using batched generation for long text ({len(request.input)} characters)")
    
    # Generate speech with automatic batching for long texts
    start = time.time()
    generate_speech_from_api(
        prompt=request.input,
        voice=request.voice,
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=10  # Process in ~1000 character chunks (roughly 1 paragraph)
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    # Return audio file
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"{request.voice}_{timestamp}.wav"
    )

# New streaming endpoint
@app.post("/v1/audio/speech/stream")
async def stream_speech_api(request: StreamingSpeechRequest):
    """
    Stream speech in real-time as it's being generated.
    
    This optimized endpoint streams audio chunks as they are generated, providing:
    1. Ultra-low latency - first audio chunk sent within milliseconds
    2. Real-time playback - audio plays while more is being generated
    3. Unlimited length - no practical limit on input text length
    4. High throughput - efficient batching for maximum performance
    
    Returns a streaming response with WAV audio data or raw PCM Float32 LE.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    input_length = len(request.input)
    print(f"Streaming request: {input_length} chars, voice: {request.voice}")
    
    # Start performance monitoring
    start_time = time.time()
    chunk_count = 0
    total_bytes = 0
    
    response_format = getattr(request, 'response_format', 'wav')
    print(f"[stream_speech_api] response_format: {response_format}")
    
    async def audio_stream_generator():
        nonlocal chunk_count, total_bytes
        
        # Always stream WAV data (int16 PCM with header)
        if len(request.input) > 1000:
            from tts_engine.inference import split_text_into_sentences
            sentences = split_text_into_sentences(request.input)
            batches, current_batch = [], ""
            for sentence in sentences:
                if len(current_batch) + len(sentence) + 1 > 1000 and current_batch:
                    batches.append(current_batch)
                    current_batch = sentence
                else:
                    current_batch = (current_batch + " " + sentence).strip() if current_batch else sentence
            if current_batch:
                batches.append(current_batch)
        else:
            batches = [request.input]

        chunk_duration_ms = 100  # 100ms chunks for larger transfers and reduced overhead
        samples_per_chunk = int(24000 * (chunk_duration_ms / 1000))
        int16_chunk_bytes = samples_per_chunk * 2
        buffer = bytearray()

        # Yield a standard WAV header
        wav_header = generate_wav_header(sample_rate=24000, bits_per_sample=16, channels=1)
        yield wav_header
        total_bytes += len(wav_header)

        try:
            # Always use int16 PCM for WAV
            for batch in batches:
                async for audio_chunk in stream_speech_from_api(prompt=batch, voice=request.voice, output_format="int16"):
                    if not audio_chunk:
                        continue
                    buffer.extend(audio_chunk)
                    # Yield full chunks
                    chunk_bytes = samples_per_chunk * 2
                    while len(buffer) >= chunk_bytes:
                        chunk = bytes(buffer[:chunk_bytes])
                        total_bytes += len(chunk)
                        yield chunk
                        del buffer[:chunk_bytes]
                        pass
            # Flush remaining buffer padded to full chunk
            if buffer:
                chunk_bytes = samples_per_chunk * 2
                pad_len = chunk_bytes - len(buffer)
                chunk = bytes(buffer) + b"\x00" * pad_len
                total_bytes += len(chunk)
                yield chunk
        except Exception as e:
            print(f"Error in streaming audio: {e}")
        finally:
            # Log performance metrics
            elapsed = time.time() - start_time
            if elapsed > 0 and chunk_count > 0:
                chars_per_sec = input_length / elapsed
                chunks_per_sec = chunk_count / elapsed
                kb_per_sec = total_bytes / elapsed / 1024
                print(f"Stream completed: {input_length} chars → {chunk_count} chunks, {total_bytes/1024:.1f}KB")
                print(f"Performance: {chars_per_sec:.1f} chars/sec, {chunks_per_sec:.1f} chunks/sec, {kb_per_sec:.1f}KB/sec")
    
    # Always return WAV data
    return StreamingResponse(
        audio_stream_generator(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Content-Type-Options": "nosniff",
            "Transfer-Encoding": "chunked"
        }
    )

@app.get("/v1/audio/voices")
async def list_voices():
    """Return list of available voices"""
    if not AVAILABLE_VOICES or len(AVAILABLE_VOICES) == 0:
        raise HTTPException(status_code=404, detail="No voices available")
    return JSONResponse(
        content={
            "status": "ok",
            "voices": AVAILABLE_VOICES
        }
    )

# Legacy API endpoint for compatibility
@app.post("/speak")
async def speak(request: Request):
    """Legacy endpoint for compatibility with existing clients"""
    data = await request.json()
    text = data.get("text", "")
    voice = data.get("voice", DEFAULT_VOICE)

    if not text:
        return JSONResponse(
            status_code=400, 
            content={"error": "Missing 'text'"}
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    # Check if we should use batched generation for longer texts
    use_batching = len(text) > 1000
    if use_batching:
        print(f"Using batched generation for long text ({len(text)} characters)")
    
    # Generate speech with batching for longer texts
    start = time.time()
    generate_speech_from_api(
        prompt=text, 
        voice=voice, 
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=10
    )
    end = time.time()
    generation_time = round(end - start, 2)

    return JSONResponse(content={
        "status": "ok",
        "voice": voice,
        "output_file": output_path,
        "generation_time": generation_time
    })

# Web UI routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to web UI"""
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request, 
            "voices": AVAILABLE_VOICES,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

@app.get("/web/", response_class=HTMLResponse)
async def web_ui(request: Request):
    """Main web UI for TTS generation"""
    # Get current config for the Web UI
    config = get_current_config()
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request, 
            "voices": AVAILABLE_VOICES, 
            "config": config,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

@app.get("/get_config")
async def get_config():
    """Get current configuration from .env file or defaults"""
    config = get_current_config()
    return JSONResponse(content=config)

@app.post("/save_config")
async def save_config(request: Request):
    """Save configuration to .env file"""
    data = await request.json()
    
    # Convert values to proper types
    for key, value in data.items():
        if key in ["ORPHEUS_MAX_TOKENS", "ORPHEUS_API_TIMEOUT", "ORPHEUS_PORT", "ORPHEUS_SAMPLE_RATE"]:
            try:
                data[key] = str(int(value))
            except (ValueError, TypeError):
                pass
        elif key in ["ORPHEUS_TEMPERATURE", "ORPHEUS_TOP_P"]:  # Removed ORPHEUS_REPETITION_PENALTY since it's hardcoded now
            try:
                data[key] = str(float(value))
            except (ValueError, TypeError):
                pass
    
    # Write configuration to .env file
    with open(".env", "w") as f:
        for key, value in data.items():
            f.write(f"{key}={value}\n")
    
    return JSONResponse(content={"status": "ok", "message": "Configuration saved successfully. Restart server to apply changes."})

@app.post("/restart_server")
async def restart_server():
    """Restart the server by touching a file that triggers Uvicorn's reload"""
    import threading
    
    def touch_restart_file():
        # Wait a moment to let the response get back to the client
        time.sleep(0.5)
        
        # Create or update restart.flag file to trigger reload
        restart_file = "restart.flag"
        with open(restart_file, "w") as f:
            f.write(str(time.time()))
            
        print("🔄 Restart flag created, server will reload momentarily...")
    
    # Start the touch operation in a separate thread
    threading.Thread(target=touch_restart_file, daemon=True).start()
    
    # Return success response
    return JSONResponse(content={"status": "ok", "message": "Server is restarting. Please wait a moment..."})

def get_current_config():
    """Read current configuration from .env.example and .env files"""
    # Default config from .env.example
    default_config = {}
    if os.path.exists(".env.example"):
        with open(".env.example", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key = line.split("=")[0].strip()
                    default_config[key] = line.split("=", 1)[1].strip()

    # Current config from .env
    current_config = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key = line.split("=")[0].strip()
                    current_config[key] = line.split("=", 1)[1].strip()

    # Merge configs, with current taking precedence
    config = {**default_config, **current_config}
    
    # Add current environment variables
    for key in config:
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value
    
    return config

@app.post("/web/", response_class=HTMLResponse)
async def generate_from_web(
    request: Request,
    text: str = Form(...),
    voice: str = Form(DEFAULT_VOICE)
):
    """Handle form submission from web UI"""
    if not text:
        return templates.TemplateResponse(
            "tts.html",
            {
                "request": request,
                "error": "Please enter some text.",
                "voices": AVAILABLE_VOICES,
                "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
                "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
            }
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    # Check if we should use batched generation for longer texts
    use_batching = len(text) > 1000
    if use_batching:
        print(f"Using batched generation for long text from web form ({len(text)} characters)")
    
    # Generate speech with batching for longer texts
    start = time.time()
    generate_speech_from_api(
        prompt=text, 
        voice=voice, 
        output_file=output_path,
        use_batching=use_batching,
        max_batch_chars=10
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    return templates.TemplateResponse(
        "tts.html",
        {
            "request": request,
            "success": True,
            "text": text,
            "voice": voice,
            "output_file": output_path,
            "generation_time": generation_time,
            "voices": AVAILABLE_VOICES,
            "VOICE_TO_LANGUAGE": VOICE_TO_LANGUAGE,
            "AVAILABLE_LANGUAGES": AVAILABLE_LANGUAGES
        }
    )

@app.post("/api/tts/stream")
async def stream_speech(
    request: Request,
    text: Annotated[str, Body(embed=True)],
    voice: Annotated[str, Body(embed=True)] = "Orpheus",
    use_cuda: bool = True,
):
    """Optimized streaming endpoint with maximum throughput and minimal latency."""
    if not text:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    input_length = len(text)
    print(f"API streaming request: {input_length} chars, voice: {voice}")
    
    # Start performance monitoring
    start_time = time.time()
    chunk_count = 0
    total_bytes = 0
    
    # Optimize buffer size for smoother playback
    initial_batch_size = max(1, min(2, input_length // 200))
    max_batch_size = max(2, min(8, input_length // 100))
    
    # Add short silence at the beginning to give client some buffering time
    # Reduced for lower latency
    SILENCE_DURATION_MS = 100  # 100ms of silence for improved buffering
    SAMPLE_RATE_BYTES_PER_MS = SAMPLE_RATE * 2 // 1000  # 2 bytes per sample
    silence_bytes = bytearray(SILENCE_DURATION_MS * SAMPLE_RATE_BYTES_PER_MS)
    
    async def stream_audio():
        nonlocal chunk_count, total_bytes
        
        # Send WAV header immediately for client parsing
        wav_header = generate_wav_header(SAMPLE_RATE)
        yield wav_header
        total_bytes += len(wav_header)
        
        # (Optional) 100ms silence padding for jitter tolerance
        silence = bytearray(SAMPLE_RATE_BYTES_PER_MS * 100)
        yield silence
        total_bytes += len(silence)
        
        # Pre-allocate buffers for better performance
        # Buffer set to 2x100ms (~200ms) for higher throughput
        buffer_size = SAMPLE_RATE_BYTES_PER_MS * 100 * 2
        audio_buffer = bytearray(buffer_size)
        buffer_position = 0
        
        try:
            # Stream audio chunks with maximum throughput
            async for chunk in stream_speech_from_api(text, voice):
                if not chunk:
                    continue
                    
                chunk_size = len(chunk)
                chunk_count += 1
                
                # Resize buffer if needed
                if buffer_position + chunk_size > len(audio_buffer):
                    new_buffer = bytearray(max(len(audio_buffer) * 2, buffer_position + chunk_size))
                    new_buffer[:buffer_position] = audio_buffer[:buffer_position]
                    audio_buffer = new_buffer
                
                # Add chunk to buffer
                audio_buffer[buffer_position:buffer_position + chunk_size] = chunk
                buffer_position += chunk_size
                
                # Yield 100ms chunks for higher throughput
                chunk_bytes = SAMPLE_RATE_BYTES_PER_MS * 100
                while True:
                    if buffer_position >= chunk_bytes:
                        yield bytes(audio_buffer[:chunk_bytes])
                        total_bytes += chunk_bytes
                        # Shift leftover
                        remaining = buffer_position - chunk_bytes
                        audio_buffer[:remaining] = audio_buffer[chunk_bytes:buffer_position]
                        buffer_position = remaining
                    else:
                        break
            # Send any remaining audio in buffer, padded
            if buffer_position > 0:
                chunk_bytes = SAMPLE_RATE_BYTES_PER_MS * 20
                pad_len = chunk_bytes - buffer_position
                yield bytes(audio_buffer[:buffer_position]) + b"\x00" * pad_len
                total_bytes += chunk_bytes
                
        except Exception as e:
            print(f"Error in streaming audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Log detailed performance metrics
            elapsed = time.time() - start_time
            if elapsed > 0 and chunk_count > 0:
                chars_per_sec = input_length / elapsed
                chunks_per_sec = chunk_count / elapsed
                kb_per_sec = total_bytes / elapsed / 1024
                
                print(f"API stream completed: {input_length} chars → {chunk_count} chunks, {total_bytes/1024:.1f}KB")
                print(f"Performance: {chars_per_sec:.1f} chars/sec, {chunks_per_sec:.1f} chunks/sec, {kb_per_sec:.1f}KB/sec")
    
    # Return StreamingResponse with optimized headers
    return StreamingResponse(
        stream_audio(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Content-Type-Options": "nosniff",
            "Transfer-Encoding": "chunked"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Check for required settings
    required_settings = ["ORPHEUS_HOST", "ORPHEUS_PORT"]
    missing_settings = [s for s in required_settings if s not in os.environ]
    if missing_settings:
        print(f"⚠️ Missing environment variable(s): {', '.join(missing_settings)}")
        print("   Using fallback values for server startup.")
    
    # Get host and port from environment variables with better error handling
    try:
        host = os.environ.get("ORPHEUS_HOST")
        if not host:
            print("⚠️ ORPHEUS_HOST not set, using 0.0.0.0 as fallback")
            host = "0.0.0.0"
    except Exception:
        print("⚠️ Error reading ORPHEUS_HOST, using 0.0.0.0 as fallback")
        host = "0.0.0.0"
        
    try:
        port = int(os.environ.get("ORPHEUS_PORT", "5005"))
    except (ValueError, TypeError):
        print("⚠️ Invalid ORPHEUS_PORT value, using 5005 as fallback")
        port = 5005
    
    model_name = os.environ.get("ORPHEUS_MODEL_NAME", "(not set)")
    print(f"🔥 Starting Orpheus-FASTAPI Server on {host}:{port}")
    print(f"🧠 Using model: {model_name}")
    print(f"💬 Web UI available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
    print(f"📖 API docs available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    
    # Read current API_URL for user information
    api_url = os.environ.get("ORPHEUS_API_URL")
    if not api_url:
        print("⚠️ ORPHEUS_API_URL not set. Please configure in .env file before generating speech.")
    else:
        print(f"🔗 Using LLM inference server at: {api_url}")
        
    # Include restart.flag in the reload_dirs to monitor it for changes
    extra_files = ["restart.flag"] if os.path.exists("restart.flag") else []
    
    # Start with reload enabled to allow automatic restart when restart.flag changes
    uvicorn.run("app:app", host=host, port=port, reload=True, reload_dirs=["."], reload_includes=["*.py", "*.html", "restart.flag"])