"""Modal port of the Parakeet (NeMo ASR) snapshot benchmark.

Mirror of ../main.py using Modal's `enable_memory_snapshot` + experimental
`enable_gpu_snapshot`. Loading the Parakeet NeMo checkpoint onto the GPU happens
inside `@modal.enter(snap=True)` so it is captured in the snapshot.

Deploy:  modal deploy app.py
"""

import modal

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    # nemo_toolkit[asr] pulls `texterrors`, which compiles a pybind11 C++
    # extension from source. Modal's `add_python` standalone interpreter ships a
    # clang that can't find a C++ stdlib ("Unsupported compiler -- at least C++11
    # support is needed!"), so force the base image's system g++ for the build.
    .apt_install("build-essential", "g++")
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install(
        "nemo_toolkit[asr]==2.3.2",
        "cuda-python==12.8.0",
        "requests",
    )
)

app = modal.App("snapshot-bench-parakeet", image=image)

DEFAULT_AUDIO_URL = (
    "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
)


@app.cls(
    gpu="A10G",
    memory=24576,
    scaledown_window=2,
    timeout=900,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class Model:
    @modal.enter(snap=True)
    def load(self):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

    @modal.method()
    def infer(self, audio_url: str = DEFAULT_AUDIO_URL):
        import os
        import tempfile

        import requests

        resp = requests.get(audio_url)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(resp.content)
            path = f.name

        try:
            output = self.model.transcribe([path])
        finally:
            os.remove(path)

        return {"text": output[0].text}
