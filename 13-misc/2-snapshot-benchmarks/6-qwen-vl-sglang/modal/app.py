"""Modal port of the Qwen2-VL-7B (SGLang) snapshot benchmark.

Mirror of ../main.py using Modal's `enable_memory_snapshot` + experimental
`enable_gpu_snapshot`. The SGLang offline engine (7B vision weights + CUDA
graphs) is spun up and warmed inside `@modal.enter(snap=True)` so the engine
state is captured in the snapshot.

Deploy:  modal deploy app.py
"""

import modal

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("libnuma-dev")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install(
        "transformers",
        "huggingface_hub",
        "hf_transfer",
        "pillow",
        "requests",
        # Pin to the CUDA 12.8 generation. sglang 0.5.5 pins torch==2.8.0 (cu128)
        # + sgl-kernel 0.3.16 (cu128) — all aligned with this base.
        "torch==2.8.0",
        "sglang[all]==0.5.5",
    )
)

app = modal.App("snapshot-bench-qwen-vl-sglang", image=image)

DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)


@app.cls(
    gpu="L40S",
    memory=61440,
    scaledown_window=2,
    timeout=900,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class Model:
    @modal.enter(snap=True)
    def load(self):
        import os
        import signal
        import tempfile

        # SGLang's engine launch calls signal.signal(SIGQUIT, ...), which Python
        # only allows on the main thread. Make signal registration a no-op when
        # it can't run so the import-time launch doesn't crash off-main-thread.
        _orig_signal = signal.signal

        def _safe_signal(signalnum, handler):
            try:
                return _orig_signal(signalnum, handler)
            except ValueError:
                return None

        signal.signal = _safe_signal

        import sglang as sgl
        from PIL import Image
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(MODEL_PATH)
        self.engine = sgl.Engine(model_path=MODEL_PATH, mem_fraction_static=0.8)

        # Warm up generation with a tiny image BEFORE the snapshot so CUDA graphs
        # are captured and we verify the engine actually generates.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as wf:
            Image.new("RGB", (64, 64), (127, 127, 127)).save(wf, format="JPEG")
            warmup_path = wf.name
        try:
            self._infer(warmup_path, "Warmup.", max_tokens=2)
            print("Warmup generation OK", flush=True)
        finally:
            os.remove(warmup_path)

    def _build_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": ""},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _infer(self, image_path: str, prompt: str, max_tokens: int):
        return self.engine.generate(
            prompt=self._build_prompt(prompt),
            image_data=[image_path],
            sampling_params={"max_new_tokens": max_tokens, "temperature": 0.2},
        )

    @modal.method()
    def infer(
        self,
        image_url: str = DEFAULT_IMAGE_URL,
        image_base64: str | None = None,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 256,
    ):
        import base64
        import os
        import tempfile

        import requests

        if image_base64:
            data = base64.b64decode(image_base64)
        else:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            data = resp.content

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(data)
            path = f.name

        try:
            output = self._infer(path, prompt, max_tokens)
        finally:
            os.remove(path)

        return {"text": output["text"]}
