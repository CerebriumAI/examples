import torch
from TTS.api import TTS
import base64
import os

os.environ["COQUI_TOS_AGREED"] = "1"
api = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

def run(prompt: str, run_id):  # run_id is optional, injected by Cerebrium at runtime
    file_path = f"{run_id}.wav"
    api.tts_to_file(text=prompt,
                file_path=file_path,
                speaker_wav="female.wav",
                language="en")
    
    # Read the file and encode it to base64
    with open(file_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    
    # Optionally, remove the temporary file
    os.remove(file_path)
    
    return {
        "message": "Audio generated successfully",
        "status_code": 200,
        "audio_base64": encoded_audio
    }