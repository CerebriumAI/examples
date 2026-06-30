"""Baseten (Truss) variant of the WhisperX snapshot benchmark.

No GPU snapshot on Baseten, so no checkpoint trigger. WhisperX (large-v2) loads
in `load()`; transcription happens in `predict()`.
"""

import os
import tempfile

# Authenticate HF downloads (pyannote VAD weights) from the Baseten secret,
# before whisperx/huggingface_hub import and read the env.
try:
    with open("/secrets/hf_access_token") as _f:
        _tok = _f.read().strip()
        os.environ.setdefault("HF_TOKEN", _tok)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _tok)
except FileNotFoundError:
    pass

import requests
import torch

# PyTorch 2.6 flipped torch.load's default to weights_only=True, which breaks
# pyannote's VAD checkpoint (it pickles omegaconf objects). Force it back to
# False — lightning passes weights_only=True explicitly, so we override rather
# than setdefault.
_orig_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_full

import whisperx  # noqa: E402  (imported after the torch.load patch is in place)

DEVICE = "cuda"
COMPUTE_TYPE = "float16"
DEFAULT_AUDIO_URL = (
    "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
)


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self._model = whisperx.load_model(
            "large-v2", device=DEVICE, compute_type=COMPUTE_TYPE
        )

    def predict(self, request: dict):
        audio_url = request.get("audio_url", DEFAULT_AUDIO_URL)
        batch_size = request.get("batch_size", 16)

        resp = requests.get(audio_url, timeout=60)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(resp.content)
            path = f.name

        try:
            audio = whisperx.load_audio(path)
            result = self._model.transcribe(audio, batch_size=batch_size)
        finally:
            os.remove(path)

        return {"language": result["language"], "segments": result["segments"]}
