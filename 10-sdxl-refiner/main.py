import base64
import io
import tempfile
from typing import Optional

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from pydantic import BaseModel

temp_dir = tempfile.TemporaryDirectory()


class Item(BaseModel):
    prompt: str
    url: str
    negative_prompt: Optional[str]
    conditioning_scale: float
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    num_images_per_prompt: int


pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    cache_dir=temp_dir.name,
)
pipe = pipe.to("cuda")


def predict(
    prompt,
    url,
    negative_prompt=None,
    conditioning_scale=0.5,
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    num_images_per_prompt=1,
):
    item = Item(
        prompt=prompt,
        url=url,
        negative_prompt=negative_prompt,
        conditioning_scale=conditioning_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    )

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
