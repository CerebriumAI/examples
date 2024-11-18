import base64
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
    "cuda", torch.float16
)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(
    base, unet=unet, torch_dtype=torch.float16, variant="fp16"
).to("cuda")

pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing="trailing"
)


class Item(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = 4
    guidance_scale: Optional[float] = 0


def convert_to_b64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64


def predict(prompt, num_inference_steps, guidance_scale):
    item = Item(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    image = pipe(
        item.prompt,
        num_inference_steps=item.num_inference_steps,
        guidance_scale=item.guidance_scale,
    ).images[0]
    results = convert_to_b64(image)

    return {"results": results}  # return your results
