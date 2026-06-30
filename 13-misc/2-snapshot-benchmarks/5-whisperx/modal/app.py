"""Modal port of the WhisperX snapshot benchmark.

Mirror of ../main.py using Modal's `enable_memory_snapshot` + experimental
`enable_gpu_snapshot`. WhisperX (large-v2) is loaded onto the GPU inside
`@modal.enter(snap=True)` so the load is captured in the snapshot.

Deploy:  modal deploy app.py
"""

import modal

# cuDNN 8 base: CTranslate2 (faster-whisper backend, pinned <4.5 by whisperx)
# needs libcudnn_ops_infer.so.8. The cuDNN-9 images renamed that lib, so we use
# a cuDNN-8 image. torch still loads its own bundled cuDNN 9 via RPATH.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg")
    .pip_install(
        # Pin torch/torchaudio/ctranslate2 so the resolver doesn't backtrack
        # across every torch release.
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "ctranslate2==4.4.0",
        "whisperx==3.4.0",
        "numpy==2.0.2",
        "scipy==1.15.0",
        "requests",
    )
)

app = modal.App("snapshot-bench-whisperx", image=image)

DEVICE = "cuda"
COMPUTE_TYPE = "float16"
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
        import torch

        # PyTorch 2.6 flipped torch.load's default to weights_only=True, which
        # breaks pyannote's VAD checkpoint (it pickles omegaconf objects). We
        # trust these HF weights, so force weights_only=False.
        _orig_torch_load = torch.load

        def _torch_load_full(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

        torch.load = _torch_load_full

        import whisperx

        self.model = whisperx.load_model(
            "large-v2", device=DEVICE, compute_type=COMPUTE_TYPE
        )

    @modal.method()
    def infer(self, audio_url: str = DEFAULT_AUDIO_URL, batch_size: int = 16):
        import os
        import tempfile

        import requests
        import whisperx

        resp = requests.get(audio_url)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(resp.content)
            path = f.name

        try:
            audio = whisperx.load_audio(path)
            result = self.model.transcribe(audio, batch_size=batch_size)
        finally:
            os.remove(path)

        return {"language": result["language"], "segments": result["segments"]}
