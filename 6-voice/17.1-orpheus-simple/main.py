import struct
import time
import io
from typing import Literal, Optional
import os
import sys
sys.path.insert(0, "/")  
from orpheus_tts import OrpheusModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

hf_secret = os.getenv('HF_TOKEN')

WAV_SAMPLE_RATE = 24000
WAV_BITS_PER_SAMPLE = 16
WAV_CHANNELS = 1
WAV_BYTE_RATE = WAV_SAMPLE_RATE * WAV_CHANNELS * WAV_BITS_PER_SAMPLE // 8
WAV_BLOCK_ALIGN = WAV_CHANNELS * WAV_BITS_PER_SAMPLE // 8

def create_wav_header_template():
    """Create WAV header template with placeholder data size"""
    data_size = 0  
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        WAV_CHANNELS,
        WAV_SAMPLE_RATE,
        WAV_BYTE_RATE,
        WAV_BLOCK_ALIGN,
        WAV_BITS_PER_SAMPLE,
        b"data",
        data_size,
    )


WAV_HEADER_TEMPLATE = create_wav_header_template()


app = FastAPI()

engine = OrpheusModel(model_name="unsloth/orpheus-3b-0.1-ft")
        

@app.get("/tts")
def tts(prompt: str = "Hey there, looks like you forgot to provide a prompt!", format: str = "wav"):

    def generate_streaming_audio():
        """Generator function that streams audio chunks directly as they're generated - optimized for TTFB"""
        start_time = time.time()
        first_chunk_time = None

        syn_tokens = engine.generate_speech(
            prompt=prompt,
            voice="tara",
            repetition_penalty=1.1,
            stop_token_ids=[128258],
            max_tokens=1500, 
            temperature=0.4,
            top_p=0.9,
        )
        
        if format == "pcm":
            try:
                for chunk in syn_tokens:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    yield chunk
            except Exception as e:
                raise RuntimeError(f"Error in PCM streaming: {e}")
        
        else:
            yield WAV_HEADER_TEMPLATE
            
            try:
                for chunk in syn_tokens:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    yield chunk
            except Exception as e:
                raise RuntimeError(f"Error in audio generation: {e}")
        
        if first_chunk_time:
            ttfb = (first_chunk_time - start_time) * 1000
            total_time = (time.time() - start_time) * 1000
            print(f"TTFB: {ttfb:.2f}ms, Total: {total_time:.2f}ms", flush=True)

    if format == "pcm":
        media_type = "audio/pcm"
        filename = "audio.pcm"
    else:  
        media_type = "audio/wav"
        filename = "audio.wav"
    
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Cache-Control": "no-cache",
        "Transfer-Encoding": "chunked"
    }
    
    return StreamingResponse(
        generate_streaming_audio(),
        media_type=media_type,
        headers=headers
    )

@app.post("/openai_speech")
def openai_speech(self, request_body: dict):
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse
    
    try:
        text_input = request_body.get("input", "")
        model = request_body.get("model", "gpt-4o-mini-tts")
        voice = request_body.get("voice", "coral")
        instructions = request_body.get("instructions")
        response_format = request_body.get("response_format", "pcm")
        
        if not text_input:
            raise HTTPException(status_code=400, detail="input field is required")
        
        voice_mapping = {
            "coral": "tara",
            "dave": "tara",  
        }
        orpheus_voice = voice_mapping.get(voice, "tara")
        
        def get_media_type_and_filename(response_format: str):
            format_map = {
                "mp3": ("audio/mpeg", "speech.mp3"),
                "opus": ("audio/opus", "speech.opus"),
                "aac": ("audio/aac", "speech.aac"),
                "flac": ("audio/flac", "speech.flac"),
                "pcm": ("audio/pcm", "speech.pcm"),
                "wav": ("audio/wav", "speech.wav"),
            }
            return format_map.get(response_format, ("audio/pcm", "speech.pcm"))
        
        def convert_audio_format(wav_data: bytes, target_format: str) -> bytes:
            if target_format == "wav":
                return wav_data
            
            try:
                from pydub import AudioSegment
                
                audio_segment = AudioSegment.from_wav(io.BytesIO(wav_data))
                
                output_data = io.BytesIO()
                
                if target_format == "mp3":
                    audio_segment.export(output_data, format="mp3", bitrate="128k")
                elif target_format == "opus":
                    audio_segment.export(output_data, format="opus", bitrate="128k")
                elif target_format == "aac":
                    audio_segment.export(output_data, format="aac", bitrate="128k")
                elif target_format == "flac":
                    audio_segment.export(output_data, format="flac")
                else:
                    return wav_data  
                
                output_data.seek(0)
                return output_data.read()
                
            except Exception as e:
                print(f"Error converting audio format: {e}")
                return wav_data  
        
        media_type, filename = get_media_type_and_filename(response_format)
        
        def generate_streaming_audio():
            syn_tokens = self.engine.generate_speech(
                prompt=text_input,
                voice=orpheus_voice,
                repetition_penalty=1.1,
                stop_token_ids=[128258],
                max_tokens=1500,  
                temperature=0.4,
                top_p=0.9,
            )
            
            if response_format == "pcm":
                for chunk in syn_tokens:
                    yield chunk
            
            else:
                audio_chunks = []
                for chunk in syn_tokens:
                    audio_chunks.append(chunk)
                
                if not audio_chunks:
                    raise HTTPException(status_code=500, detail="No audio generated")
                
                full_audio = b"".join(audio_chunks)
                
                data_size = len(full_audio)
                wav_header = struct.pack(
                    "<4sI4s4sIHHIIHH4sI",
                    b"RIFF",
                    36 + data_size,
                    b"WAVE",
                    b"fmt ",
                    16,
                    1,
                    WAV_CHANNELS,
                    WAV_SAMPLE_RATE,
                    WAV_BYTE_RATE,
                    WAV_BLOCK_ALIGN,
                    WAV_BITS_PER_SAMPLE,
                    b"data",
                    data_size,
                )
                wav_data = wav_header + full_audio
                
                final_audio_data = convert_audio_format(wav_data, response_format)
                
                audio_stream = io.BytesIO(final_audio_data)
                while True:
                    chunk = audio_stream.read(8192)
                    if not chunk:
                        break
                    yield chunk
        
        return StreamingResponse(
            generate_streaming_audio(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in OpenAI API speech synthesis: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")