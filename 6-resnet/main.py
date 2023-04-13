import torch
from datasets import load_dataset
from fastapi import FastAPI
from transformers import AutoImageProcessor, ResNetForImageClassification

app = FastAPI()


@app.post("/predict")
def generate(item, run_id, logger):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    label = model.config.id2label[predicted_label]

    print("Label: " + label)
    return label
