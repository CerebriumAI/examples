from typing import Optional
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import base64 

from transformers import CLIPProcessor, CLIPModel


class Item(BaseModel):
    image: Optional[str]
    file_url: Optional[str]


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

classes = [
    "drugs", 
    "nudity", 
    "women naked", 
    "guns",
    "violence",
    "killing"
]

def download_file_from_url(logger, url: str, filename: str):
    logger.info("Downloading file...")

    import requests

    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Download was successful")

        with open(f"/persistent-storage/{filename}", "wb") as f:
            f.write(response.content)

        return f"/persistent-storage/{filename}"

    else:
        logger.info(response)
        raise Exception("Download failed")

def predict(item, run_id, logger):
    item = Item(**item)

    if not item.image and not item.file_url:
        return "image or file_url field is required."

    if item.image:
        image = Image.open(BytesIO(base64.b64decode(item.image)))
    elif item.file_url:
        image = Image.open(download_file_from_url(logger, item.file_url, run_id))

    inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1).tolist()
    max_value = max(probs)
    index_of_max = probs.index(max_value)

    if (index_of_max < 6 and max_value[0] > 0.5):
        return {"safe": False, "probability": max_value[0]}
    else:
        return {"safe": True}