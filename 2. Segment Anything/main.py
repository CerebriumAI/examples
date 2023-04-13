import base64
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from transformers import ViTImageProcessor, ViTForImageClassification


class Item(BaseModel):
    image: Optional[str]
    file_url: Optional[str]
    cursor: list


sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.96,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')


def find_annotation_by_coordinates(annotations, x, y):
    for ann in annotations:
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        if bbox_x <= x <= bbox_x + bbox_w and bbox_y <= y <= bbox_y + bbox_h:
            return ann
    return None


def create_image(image, ann):
    m = ann['segmentation']
    resized_original_image = cv2.resize(image, (m.shape[1], m.shape[0]))
    mask = np.ones((m.shape[0], m.shape[1], 3), dtype=np.uint8) * 255
    mask[m] = resized_original_image[m]  # Set the segmented area to white
    x, y, w, h = ann['bbox']
    cropped_image = mask[y:y + h, x:x + w]

    return cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)


def classify(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]


def download_image(url):
    if url:
        r = requests.get(url)
    else:
        return ValueError("No valid url passed")

    return Image.open(BytesIO(r.content))


def predict(item: Item):
    if not item.image and not item.file_url:
        return "image or file_url field is required."

    if item.image:
        image = Image.open(BytesIO(base64.b64decode(item.image)))
    elif item.file_url:
        image = download_image(item.file_url)

    masks = mask_generator.generate(image)
    selected_annotation = find_annotation_by_coordinates(masks, item.cursor[0], item.cursor[1])

    if not selected_annotation:
        return {"message": "No annotation found at the given coordinates."}
    segmented_image = create_image(image, selected_annotation)
    result = classify(segmented_image)

    return {"result": result}

# first about creating image from mask then downloading image then classifying it
