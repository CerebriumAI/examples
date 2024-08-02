import base64
from io import BytesIO
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from pydantic import BaseModel

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)
pipe = pipe.to("cuda")


class Item(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = 4
    guidance_scale: Optional[float] = 7.5


def predict(prompt: str, num_inference_steps: Optional[int] = 4, guidance_scale: Optional[float] = 7.5) -> dict:
    item = Item(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    images = pipe(
        item.prompt,
        num_inference_steps=item.num_inference_steps,
        guidance_scale=item.guidance_scale,
    ).images

    finished_images = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return {"results": finished_images}
