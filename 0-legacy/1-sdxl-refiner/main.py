import base64
import io
from typing import Optional

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from pydantic import BaseModel


class Item(BaseModel):
    prompt: str
    url: str
    negative_prompt: Optional[str] = None
    conditioning_scale: Optional[float] = 0.5
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    num_images_per_prompt: Optional[int] = 1


pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe = pipe.to("cuda")


def predict(item, run_id, logger):
    item = Item(**item)

    init_image = load_image(item.url).convert("RGB")
    images = pipe(
        item.prompt,
        negative_prompt=item.negative_prompt,
        controlnet_conditioning_scale=item.conditioning_scale,
        height=item.height,
        width=item.width,
        num_inference_steps=item.num_inference_steps,
        guidance_scale=item.guidance_scale,
        num_images_per_prompt=item.num_images_per_prompt,
        image=init_image,
    ).images

    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return {"images": finished_images}
