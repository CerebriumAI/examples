import base64
import http.client
import io
import urllib.request

import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Enables TF32 matmuls (clears the inductor warning, small speedup on Ada/L40).
torch.set_float32_matmul_precision("high")

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
).to("cuda")
model = torch.compile(model)


_warmup = processor(Image.new("RGB", (224, 224)), return_tensors="pt").to("cuda")
with torch.no_grad():
    _ = model(**_warmup)

try:
    req = urllib.request.Request("http://169.254.169.253:8234/checkpoint", method="POST")
    urllib.request.urlopen(req)
    print("Checkpointed successfully")
except http.client.RemoteDisconnected:
    # TCP connections disconnect on restore and throw remote
    pass

DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


def run(image_url: str = DEFAULT_IMAGE_URL, image_base64: str = None):
    # Prefer an inline base64 image (no egress needed); otherwise download the
    # URL with a hard timeout so a slow/blocked host fails fast instead of
    # hanging the request indefinitely.
    if image_base64:
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    else:
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    processed_input = processor(image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**processed_input).logits

    predicted_id = logits.argmax(-1).item()
    return {"label": model.config.id2label[predicted_id]}
