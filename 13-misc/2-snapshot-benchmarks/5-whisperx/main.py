import http.client
import os
import tempfile
import urllib.request

import requests
import torch

# PyTorch 2.6 flipped torch.load's default to weights_only=True, which breaks
# pyannote's VAD checkpoint (it pickles omegaconf objects). We trust these
# weights (pulled from HF), so force weights_only=False for the internal loads
# whisperx/pyannote/lightning perform.
_orig_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    # Force (not setdefault) — lightning's pl_load passes weights_only=True
    # explicitly, so a setdefault wouldn't override it.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_full

import whisperx  # noqa: E402  (imported after the torch.load patch is in place)

DEVICE = "cuda"
COMPUTE_TYPE = "float16"

# --- Cold-start work that we want the snapshot to capture ----------------------
# Loading WhisperX (large-v2) onto the GPU is the expensive part of the cold
# start. We run it at import time so it happens once, before the checkpoint.
model = whisperx.load_model("large-v2", device=DEVICE, compute_type=COMPUTE_TYPE)

try:
    req = urllib.request.Request("http://169.254.169.253:8234/checkpoint", method="POST")
    urllib.request.urlopen(req)
    print("Checkpointed successfully")
except http.client.RemoteDisconnected:
    # TCP connections disconnect on restore and throw remote
    pass

DEFAULT_AUDIO_URL = (
    "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
)


def run(audio_url: str = DEFAULT_AUDIO_URL, batch_size: int = 16):
    resp = requests.get(audio_url)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(resp.content)
        path = f.name

    try:
        audio = whisperx.load_audio(path)
        result = model.transcribe(audio, batch_size=batch_size)
    finally:
        os.remove(path)

    return {"language": result["language"], "segments": result["segments"]}
