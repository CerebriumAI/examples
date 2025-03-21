# Orpheus-FASTAPI by Lex-au
# https://github.com/Lex-au/Orpheus-FastAPI
# Description: Main FastAPI server for Orpheus Text-to-Speech

import os
import time
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from tts_engine import generate_speech_from_api, AVAILABLE_VOICES, DEFAULT_VOICE

# Create FastAPI app
app = FastAPI(
    title="Orpheus-FASTAPI",
    description="High-performance Text-to-Speech server using Orpheus-FASTAPI",
    version="1.0.0"
)

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
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{request.voice}_{timestamp}.wav"
    
    # Generate speech
    start = time.time()
    generate_speech_from_api(
        prompt=request.input,
        voice=request.voice,
        output_file=output_path
    )
    end = time.time()
    generation_time = round(end - start, 2)
    
    # Return audio file
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"{request.voice}_{timestamp}.wav"
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
    
    # Generate speech
    start = time.time()
    generate_speech_from_api(prompt=text, voice=voice, output_file=output_path)
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
        {"request": request, "voices": AVAILABLE_VOICES}
    )

@app.get("/web/", response_class=HTMLResponse)
async def web_ui(request: Request):
    """Main web UI for TTS generation"""
    return templates.TemplateResponse(
        "tts.html",
        {"request": request, "voices": AVAILABLE_VOICES}
    )

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
                "voices": AVAILABLE_VOICES
            }
        )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/{voice}_{timestamp}.wav"
    
    # Generate speech
    start = time.time()
    generate_speech_from_api(prompt=text, voice=voice, output_file=output_path)
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
            "voices": AVAILABLE_VOICES
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("🔥 Starting Orpheus-FASTAPI Server (CUDA)")
    uvicorn.run("app:app", host="0.0.0.0", port=5005, reload=True)
