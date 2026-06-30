"""Baseten (Truss) variant of the Qwen2-VL 7B (SGLang) snapshot benchmark.

No GPU snapshot on Baseten, so no checkpoint trigger. Weights are cached on a
Baseten volume (model_cache); we block on that download, then build the
in-process SGLang offline engine from the local path in `load()`. A vision
request is served in `predict()`.
"""

import sglang as sgl
from transformers import AutoProcessor

# model_cache volume_folder mount path.
CACHE_DIR = "/app/model_cache/qwen2-vl-7b"
DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


class Model:
    def __init__(self, **kwargs):
        self._data_resolver = kwargs.get("lazy_data_resolver")
        self._processor = None
        self._engine = None

    def load(self):
        if self._data_resolver:
            self._data_resolver.block_until_download_complete()
        self._processor = AutoProcessor.from_pretrained(CACHE_DIR)
        self._engine = sgl.Engine(model_path=CACHE_DIR, mem_fraction_static=0.8)

    def predict(self, request: dict):
        image_url = request.get("image_url", DEFAULT_IMAGE_URL)
        prompt = request.get("prompt", "Describe this image in detail.")
        max_tokens = request.get("max_tokens", 256)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        output = self._engine.generate(
            prompt=text,
            image_data=[image_url],
            sampling_params={"max_new_tokens": max_tokens, "temperature": 0.2},
        )

        return {"text": output["text"]}
