import base64
import http.client
import io
import urllib.request

import torchaudio
from chatterbox.tts import ChatterboxTTS


model = ChatterboxTTS.from_pretrained(device="cuda")

try:
    req = urllib.request.Request("http://169.254.169.253:8234/checkpoint", method="POST")
    urllib.request.urlopen(req)
    print("Checkpointed successfully")
except http.client.RemoteDisconnected:
    # TCP connections disconnect on restore and throw remote
    pass


def run(
    prompt: str = "Hello from Cerebrium. This audio was generated right after a GPU snapshot restore.",
):
    wav = model.generate(prompt)

    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, model.sr, format="wav")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"audio_b64": audio_b64, "sample_rate": model.sr}
