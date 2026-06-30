"""Baseten (Truss) variant of the ViT + torch.compile snapshot benchmark.

No GPU snapshot on Baseten — so the compiled graph is NOT preserved across cold
starts. The model loads and compiles (with a warmup forward) in `load()`, which
on Baseten runs on every scale-from-zero. Inference happens in `predict()`.
"""

import base64
import io

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# model_cache volume_folder mount path.
CACHE_DIR = "/app/model_cache/vit-base"


class Model:
    def __init__(self, **kwargs):
        self._data_resolver = kwargs.get("lazy_data_resolver")
        self.processor = None
        self.model = None

    def load(self):
        if self._data_resolver:
            self._data_resolver.block_until_download_complete()
        torch.set_float32_matmul_precision("high")
        self.processor = AutoImageProcessor.from_pretrained(CACHE_DIR)
        self.model = AutoModelForImageClassification.from_pretrained(CACHE_DIR).to(
            "cuda"
        )
        self.model = torch.compile(self.model)

        # Warm up the compiled graph so the first real request isn't a recompile.
        warmup = self.processor(
            Image.new("RGB", (224, 224)), return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            _ = self.model(**warmup)

    def predict(self, request: dict):
        image_base64 = request.get("image_base64")
        if image_base64:
            image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert(
                "RGB"
            )
        else:
            resp = requests.get(request.get("image_url", DEFAULT_IMAGE_URL), timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        processed_input = self.processor(image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = self.model(**processed_input).logits

        predicted_id = logits.argmax(-1).item()
        return {"label": self.model.config.id2label[predicted_id]}
