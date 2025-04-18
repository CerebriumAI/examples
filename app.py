# Orpheus-FASTAPI by Lex-au
# https://github.com/Lex-au/Orpheus-FastAPI
# Description: Main FastAPI server for Orpheus Text-to-Speech

import os
import time
import asyncio
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv

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

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import io
import struct
import json

from tts_engine import (
    generate_speech_from_api, 
    stream_speech_from_api,  # Added for streaming endpoint
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
    
    This endpoint streams audio chunks as they are generated, allowing for:
    1. Lower latency - first audio starts playing almost immediately
    2. Real-time playback - audio plays while more is being generated
    3. Unlimited length - no practical limit on input text length
    
    Returns a streaming response with WAV audio data.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    # Generate WAV header (44 bytes)
    # We'll use a standard WAV header with PCM format
    async def audio_stream_generator():
        # First yield the WAV header
        yield generate_wav_header(SAMPLE_RATE)
        print("Sent WAV header")
        
        # Then stream the audio chunks as they're generated
        async for audio_chunk in stream_speech_from_api(
            prompt=request.input,
            voice=request.voice
        ):
            yield audio_chunk
    
    return StreamingResponse(
        audio_stream_generator(),
        media_type="audio/wav"
    )

# Helper function to generate a WAV header
def generate_wav_header(sample_rate):
    # WAV header format
    channels = 1  # Mono
    bits_per_sample = 16  # 16-bit PCM
    
    # We don't know the data size in advance for streaming, so use a placeholder
    # Some audio players might have issues with this, but most modern browsers handle it well
    data_size = 0  # Placeholder for unknown stream length
    
    # Calculate header values
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    # Create header - RIFF chunk descriptor
    header = bytearray()
    header.extend(b'RIFF')  # ChunkID
    header.extend(struct.pack('<I', data_size + 36))  # ChunkSize (file size - 8)
    header.extend(b'WAVE')  # Format
    
    # fmt sub-chunk
    header.extend(b'fmt ')  # Subchunk1ID
    header.extend(struct.pack('<I', 16))  # Subchunk1Size (16 for PCM)
    header.extend(struct.pack('<H', 1))  # AudioFormat (1 for PCM)
    header.extend(struct.pack('<H', channels))  # NumChannels
    header.extend(struct.pack('<I', sample_rate))  # SampleRate
    header.extend(struct.pack('<I', byte_rate))  # ByteRate
    header.extend(struct.pack('<H', block_align))  # BlockAlign
    header.extend(struct.pack('<H', bits_per_sample))  # BitsPerSample
    
    # data sub-chunk
    header.extend(b'data')  # Subchunk2ID
    header.extend(struct.pack('<I', data_size))  # Subchunk2Size (data size)
    
    return header

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