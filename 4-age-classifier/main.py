from io import BytesIO

import requests
from PIL import Image
from pydantic import BaseModel
from transformers import ViTFeatureExtractor, ViTForImageClassification


class Item(BaseModel):
    image: str


# Init model, transforms
model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier")
transforms = ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")


def download_image(image):
    if image:
        r = requests.get(image)
    else:
        r = requests.get(
            "https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true"
        )

    return Image.open(BytesIO(r.content))


def predict(item, run_id, logger):
    # Get example image from official fairface repo + read it in as an image
    image = download_image(item["image"])

    # Transform our image and pass it through the model
    inputs = transforms(image, return_tensors="pt")
    output = model(**inputs)

    # Predicted Class probabilities
    proba = output.logits.softmax(1)

    # Predicted Classes
    pred = proba.argmax(1).item()

    labels = {
        0: "0-2",
        1: "3-9",
        2: "10-19",
        3: "20-29",
        4: "30-39",
        5: "40-49",
        6: "50-59",
        7: "60-69",
        8: "more than 70",
    }
    predicted_class = labels[pred]

    return {"prediction": f"This person is in the {predicted_class} age category"}
