"""Modal port of the Chatterbox-TTS snapshot benchmark.

Mirror of ../main.py using Modal's `enable_memory_snapshot` + experimental
`enable_gpu_snapshot`. ChatterboxTTS is loaded onto the GPU inside
`@modal.enter(snap=True)` so the load is captured in the snapshot.

Deploy:  modal deploy app.py
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "chatterbox-tts",
    "torchaudio",
)

app = modal.App("snapshot-bench-chatterbox-tts", image=image)


@app.cls(
    gpu="A10G",
    # No explicit `memory=` — pinning it appeared to trigger a SIGSEGV (exit 139)
    # on snapshot restore; let Modal use its default memory allocation.
    scaledown_window=2,
    timeout=900,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class Model:
    @modal.enter(snap=True)
    def load(self):
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    @modal.method()
    def infer(
        self,
        prompt: str = "Hello from Modal. This audio was generated right after a GPU snapshot restore.",
    ):
        import base64
        import io

        import torchaudio

        wav = self.model.generate(prompt)

        buffer = io.BytesIO()
        torchaudio.save(buffer, wav, self.model.sr, format="wav")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"audio_b64": audio_b64, "sample_rate": self.model.sr}
