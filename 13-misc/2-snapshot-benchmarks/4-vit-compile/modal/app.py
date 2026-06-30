"""Modal port of the ViT (torch.compile) snapshot benchmark.

Mirror of ../main.py using Modal's `enable_memory_snapshot` + experimental
`enable_gpu_snapshot`. The model is loaded, `torch.compile`d and warmed inside
`@modal.enter(snap=True)`, so the compiled CUDA graph is captured in the
snapshot and restores skip recompilation entirely.

Deploy:  modal deploy app.py
"""

import modal

MODEL_ID = "google/vit-base-patch16-224"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "pillow",
        "requests",
    )
)

app = modal.App("snapshot-bench-vit-compile", image=image)

DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


@app.cls(
    gpu="L40S",
    memory=16384,
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
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        # Enables TF32 matmuls (clears the inductor warning, small speedup).
        torch.set_float32_matmul_precision("high")

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to("cuda")
        self.model = torch.compile(model)

        # Warm the compiled graph BEFORE the snapshot so restores skip recompile.
        warmup = self.processor(Image.new("RGB", (224, 224)), return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            _ = self.model(**warmup)

    @modal.method()
    def infer(self, image_url: str = DEFAULT_IMAGE_URL, image_base64: str | None = None):
        import base64
        import io

        import requests
        import torch
        from PIL import Image

        if image_base64:
            image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        else:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        processed = self.processor(image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = self.model(**processed).logits

        predicted_id = logits.argmax(-1).item()
        return {"label": self.model.config.id2label[predicted_id]}
