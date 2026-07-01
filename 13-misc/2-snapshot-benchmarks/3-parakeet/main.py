import http.client
import os
import tempfile
import urllib.request
import requests
import nemo.collections.asr as nemo_asr

# --- Cold-start work that we want the snapshot to capture ----------------------
# Loading the Parakeet NeMo checkpoint onto the GPU dominates the cold start.
# We do it once at import time, before the checkpoint is taken.
model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/parakeet-tdt-0.6b-v2"
)

try:
    req = urllib.request.Request("http://169.254.169.253:8234/checkpoint", method="POST")
    urllib.request.urlopen(req)
    print("Checkpointed successfully")
except http.client.RemoteDisconnected:
    print("Checkpointed failed")
    # TCP connections disconnect on restore and throw remote
    pass

# A short public sample clip used when no audio is supplied.
DEFAULT_AUDIO_URL = (
    "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
)


def run(audio_url: str = DEFAULT_AUDIO_URL):
    resp = requests.get(audio_url)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(resp.content)
        path = f.name

    try:
        output = model.transcribe([path])
    finally:
        os.remove(path)

    return {"text": output[0].text}
