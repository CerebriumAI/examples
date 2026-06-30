"""Baseten (Truss) variant of the Chatterbox TTS snapshot benchmark.

No GPU snapshot on Baseten, so no checkpoint trigger. Weights load in `load()`;
generation happens in `predict()`.
"""

import base64
import io

import torchaudio
from chatterbox.tts import ChatterboxTTS


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self._model = ChatterboxTTS.from_pretrained(device="cuda")

    def predict(self, request: dict):
        prompt = request.get(
            "prompt",
            "Hello from Baseten. This audio was generated on a cold start.",
        )
        wav = self._model.generate(prompt)

        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, self._model.sr, format="wav")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"audio_b64": audio_b64, "sample_rate": self._model.sr}
