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
    # stream_speech_from_api,  # Removed: function no longer exists
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
    header.extend(struct.pack('<I', 0))  # Placeholder for file size (filled at end)
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
    header.extend(struct.pack('<I', 0))  # Placeholder for data size (filled at end)
    
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
        max_batch_chars=1000  # Process in ~1000 character chunks (roughly 1 paragraph)
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
    
    Returns a streaming response with WAV audio data.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    input_length = len(request.input)
    print(f"Streaming request: {input_length} chars, voice: {request.voice}")
    
    # Start performance monitoring
    start_time = time.time()
    chunk_count = 0
    total_bytes = 0
    
    # Optimize buffer size based on input text length
    # Larger buffers for longer texts improve network efficiency
    initial_batch_size = max(2, min(4, input_length // 200))
    max_batch_size = max(8, min(32, input_length // 100))
    
    # Add short silence at the beginning to give client more buffering time
    # This helps prevent buffer underruns during playback startup
    SILENCE_DURATION_MS = 250  # 250ms of silence
    SAMPLE_RATE_BYTES_PER_MS = SAMPLE_RATE * 2 // 1000  # 2 bytes per sample
    silence_bytes = bytearray(SILENCE_DURATION_MS * SAMPLE_RATE_BYTES_PER_MS)
    
    async def audio_stream_generator():
        nonlocal chunk_count, total_bytes
        
        # Get cached WAV header (or generate if not in cache)
        wav_header = generate_wav_header(SAMPLE_RATE)
        yield wav_header
        total_bytes += len(wav_header)
        
        # Add silence padding at the beginning to help client buffering
        yield bytes(silence_bytes)
        total_bytes += len(silence_bytes)
        
        # Pre-allocate audio buffer for better memory efficiency
        buffer_size = 16384  # Increased from 8192 to 16384 (16KB)
        audio_buffer = bytearray(buffer_size)
        buffer_position = 0
        
        # Track timing for consistent delivery
        last_yield_time = time.time()
        target_chunk_duration_ms = 100  # Target ~100ms per chunk for smooth playback
        
        # Dynamic batching parameters with better consistency
        current_batch_size = initial_batch_size
        chunks_since_yield = 0
        
        # Collect initial incoming chunks for more consistent delivery
        initial_chunks = []
        initial_collection_complete = False
        initial_collection_target = 5  # Collect first 5 chunks before starting to yield
        
        try:
            # Stream audio chunks from TTS engine with optimized batching
            async for audio_chunk in stream_speech_from_api(
                prompt=request.input,
                voice=request.voice
            ):
                if not audio_chunk:
                    continue
                    
                chunk_size = len(audio_chunk)
                chunk_count += 1
                
                # Special handling for initial chunks to build buffer
                if not initial_collection_complete and len(initial_chunks) < initial_collection_target:
                    initial_chunks.append(audio_chunk)
                    # If we've collected enough initial chunks, process them all at once
                    if len(initial_chunks) >= initial_collection_target:
                        for chunk in initial_chunks:
                            # Resize buffer if needed
                            if buffer_position + len(chunk) > len(audio_buffer):
                                new_buffer = bytearray(max(len(audio_buffer) * 2, buffer_position + len(chunk)))
                                new_buffer[:buffer_position] = audio_buffer[:buffer_position]
                                audio_buffer = new_buffer
                            
                            # Add chunk to buffer
                            audio_buffer[buffer_position:buffer_position + len(chunk)] = chunk
                            buffer_position += len(chunk)
                        
                        # Yield the combined initial chunks
                        yield bytes(audio_buffer[:buffer_position])
                        total_bytes += buffer_position
                        buffer_position = 0
                        initial_collection_complete = True
                        last_yield_time = time.time()
                    continue
                
                # Resize buffer if needed
                if buffer_position + chunk_size > len(audio_buffer):
                    new_buffer = bytearray(max(len(audio_buffer) * 2, buffer_position + chunk_size))
                    new_buffer[:buffer_position] = audio_buffer[:buffer_position]
                    audio_buffer = new_buffer
                
                # Add chunk to buffer
                audio_buffer[buffer_position:buffer_position + chunk_size] = audio_chunk
                buffer_position += chunk_size
                chunks_since_yield += 1
                
                # Yield buffer based on improved adaptive strategy
                should_yield = False
                current_time = time.time()
                elapsed_since_last_yield = (current_time - last_yield_time) * 1000  # in ms
                
                # Yield based on consistent timing (~100ms chunks) for smooth playback
                if elapsed_since_last_yield >= target_chunk_duration_ms:
                    should_yield = True
                # Also yield if buffer gets very large
                elif buffer_position >= buffer_size:
                    should_yield = True
                # Yield based on number of chunks collected with adaptive batch size
                elif chunks_since_yield >= current_batch_size:
                    should_yield = True
                
                if should_yield and buffer_position > 0:
                    yield bytes(audio_buffer[:buffer_position])
                    total_bytes += buffer_position
                    buffer_position = 0
                    chunks_since_yield = 0
                    last_yield_time = current_time
                    
                    # Adaptively adjust batch size based on timing
                    # For smooth playback, we want consistent chunk sizes
                    if elapsed_since_last_yield < target_chunk_duration_ms * 0.8:
                        # Yielding too quickly, increase batch size
                        current_batch_size = min(current_batch_size + 1, max_batch_size)
                    elif elapsed_since_last_yield > target_chunk_duration_ms * 1.2:
                        # Yielding too slowly, decrease batch size
                        current_batch_size = max(initial_batch_size, current_batch_size - 1)
            
            # Send any remaining audio in buffer
            if buffer_position > 0:
                yield bytes(audio_buffer[:buffer_position])
                total_bytes += buffer_position
                
        except Exception as e:
            print(f"Error in streaming audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Log performance metrics
            elapsed = time.time() - start_time
            if elapsed > 0 and chunk_count > 0:
                chars_per_sec = input_length / elapsed
                chunks_per_sec = chunk_count / elapsed
                kb_per_sec = total_bytes / elapsed / 1024
                
                print(f"Stream completed: {input_length} chars → {chunk_count} chunks, {total_bytes/1024:.1f}KB")
                print(f"Performance: {chars_per_sec:.1f} chars/sec, {chunks_per_sec:.1f} chunks/sec, {kb_per_sec:.1f}KB/sec")
    
    # Use optimized headers for streaming response
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
        max_batch_chars=1000
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
                    key, value = line.split("=", 1)
                    default_config[key] = value
    
    # Current config from .env
    current_config = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    current_config[key] = value
    
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
        max_batch_chars=1000
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
    initial_batch_size = max(2, min(4, input_length // 200))
    max_batch_size = max(8, min(32, input_length // 100))
    
    # Add short silence at the beginning to give client more buffering time
    SILENCE_DURATION_MS = 250  # 250ms of silence
    SAMPLE_RATE_BYTES_PER_MS = SAMPLE_RATE * 2 // 1000  # 2 bytes per sample
    silence_bytes = bytearray(SILENCE_DURATION_MS * SAMPLE_RATE_BYTES_PER_MS)
    
    async def stream_audio():
        nonlocal chunk_count, total_bytes
        
        # Use cached WAV header for maximum performance
        wav_header = generate_wav_header(SAMPLE_RATE)
        yield wav_header
        total_bytes += len(wav_header)
        
        # Add silence padding at the beginning to help client buffering
        yield bytes(silence_bytes)
        total_bytes += len(silence_bytes)
        
        # Pre-allocate buffers for better performance
        buffer_size = 16384  # Increased from 8192 to 16KB
        audio_buffer = bytearray(buffer_size)
        buffer_position = 0
        
        # Track timing for consistent delivery
        last_yield_time = time.time()
        target_chunk_duration_ms = 100  # Target ~100ms per chunk
        
        # Dynamic batching parameters
        current_batch_size = initial_batch_size
        chunks_since_yield = 0
        
        # Collect initial incoming chunks for more consistent delivery
        initial_chunks = []
        initial_collection_complete = False
        initial_collection_target = 5  # Collect first 5 chunks before starting to yield
        
        try:
            # Stream audio chunks with maximum throughput
            async for chunk in stream_speech_from_api(text, voice):
                if not chunk:
                    continue
                    
                chunk_size = len(chunk)
                chunk_count += 1
                
                # Special handling for initial chunks to build buffer
                if not initial_collection_complete and len(initial_chunks) < initial_collection_target:
                    initial_chunks.append(chunk)
                    # If we've collected enough initial chunks, process them all at once
                    if len(initial_chunks) >= initial_collection_target:
                        for c in initial_chunks:
                            # Resize buffer if needed
                            if buffer_position + len(c) > len(audio_buffer):
                                new_buffer = bytearray(max(len(audio_buffer) * 2, buffer_position + len(c)))
                                new_buffer[:buffer_position] = audio_buffer[:buffer_position]
                                audio_buffer = new_buffer
                            
                            # Add chunk to buffer
                            audio_buffer[buffer_position:buffer_position + len(c)] = c
                            buffer_position += len(c)
                        
                        # Yield the combined initial chunks
                        yield bytes(audio_buffer[:buffer_position])
                        total_bytes += buffer_position
                        buffer_position = 0
                        initial_collection_complete = True
                        last_yield_time = time.time()
                    continue
                
                # Resize buffer if needed
                if buffer_position + chunk_size > len(audio_buffer):
                    new_buffer = bytearray(max(len(audio_buffer) * 2, buffer_position + chunk_size))
                    new_buffer[:buffer_position] = audio_buffer[:buffer_position]
                    audio_buffer = new_buffer
                
                # Add chunk to buffer
                audio_buffer[buffer_position:buffer_position + chunk_size] = chunk
                buffer_position += chunk_size
                chunks_since_yield += 1
                
                # Adaptive yielding strategy based on timing and buffering
                should_yield = False
                current_time = time.time()
                elapsed_since_last_yield = (current_time - last_yield_time) * 1000  # in ms
                
                # Yield based on consistent timing (~100ms chunks) for smooth playback
                if elapsed_since_last_yield >= target_chunk_duration_ms:
                    should_yield = True
                # Also yield if buffer gets very large
                elif buffer_position >= buffer_size:
                    should_yield = True
                # Yield based on number of chunks collected with adaptive batch size
                elif chunks_since_yield >= current_batch_size:
                    should_yield = True
                
                if should_yield and buffer_position > 0:
                    yield bytes(audio_buffer[:buffer_position])
                    total_bytes += buffer_position
                    buffer_position = 0
                    chunks_since_yield = 0
                    last_yield_time = current_time
                    
                    # Adaptively adjust batch size based on timing
                    # For smooth playback, we want consistent chunk sizes
                    if elapsed_since_last_yield < target_chunk_duration_ms * 0.8:
                        # Yielding too quickly, increase batch size
                        current_batch_size = min(current_batch_size + 1, max_batch_size)
                    elif elapsed_since_last_yield > target_chunk_duration_ms * 1.2:
                        # Yielding too slowly, decrease batch size
                        current_batch_size = max(initial_batch_size, current_batch_size - 1)
            
            # Send any remaining audio in buffer
            if buffer_position > 0:
                yield bytes(audio_buffer[:buffer_position])
                total_bytes += buffer_position
                
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
        media_type="audio/wav",
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
    
    print(f"🔥 Starting Orpheus-FASTAPI Server on {host}:{port}")
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