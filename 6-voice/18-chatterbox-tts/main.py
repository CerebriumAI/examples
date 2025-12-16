from chatterbox.tts_turbo import ChatterboxTurboTTS
import io
import torchaudio as ta
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

# Load the Turbo model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

@app.post("/run")
def run(prompt: str):
    wav = model.generate(prompt)

    buffer = io.BytesIO()
    ta.save(buffer, wav, model.sr, format="wav")
    buffer.seek(0)

    return StreamingResponse(
        io.BytesIO(buffer.read()),
        media_type="audio/wav",
    )
