"""Baseten (Truss) variant of the NVIDIA Parakeet snapshot benchmark.

No GPU snapshot on Baseten, so no checkpoint trigger. The NeMo .nemo checkpoint
is cached on a Baseten volume (model_cache); we block on that download, then
restore from the local path in `load()`. Transcription happens in `predict()`.
"""

import os
import tempfile

import requests
import nemo.collections.asr as nemo_asr

# model_cache volume_folder mount path; the repo holds a single .nemo file.
NEMO_PATH = "/app/model_cache/parakeet-tdt/parakeet-tdt-0.6b-v2.nemo"
DEFAULT_AUDIO_URL = (
    "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
)


class Model:
    def __init__(self, **kwargs):
        self._data_resolver = kwargs.get("lazy_data_resolver")
        self._model = None

    def load(self):
        if self._data_resolver:
            self._data_resolver.block_until_download_complete()
        self._model = nemo_asr.models.ASRModel.restore_from(NEMO_PATH)

    def predict(self, request: dict):
        audio_url = request.get("audio_url", DEFAULT_AUDIO_URL)

        resp = requests.get(audio_url, timeout=60)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(resp.content)
            path = f.name

        try:
            output = self._model.transcribe([path])
        finally:
            os.remove(path)

        return {"text": output[0].text}
