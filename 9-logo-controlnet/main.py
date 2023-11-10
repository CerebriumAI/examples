from typing import Optional
from pydantic import BaseModel
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2
import io
import base64

class Item(BaseModel):
    prompt: str
    image_url: str
    negative_prompt: Optional[str] = None
    conditioning_scale: Optional[float] = 0.5
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    num_images_per_prompt: Optional[int] = 1

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()


def predict(item, run_id, logger):
    item = Item(**item)

    init_image = load_image(item.image_url)
    image = np.array(init_image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    images = pipe(
        item.prompt,
        negative_prompt=item.negative_prompt,
        image=image,
        controlnet_conditioning_scale=item.conditioning_scale,
        height=item.height,
        width=item.width,
        num_inference_steps=item.num_inference_steps,
        guidance_scale=item.guidance_scale,
        num_images_per_prompt=item.num_images_per_prompt
    ).images

    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return {"images": finished_images}