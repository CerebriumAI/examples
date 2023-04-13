import base64
import io
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pydantic import BaseModel


class Item(BaseModel):
    prompt: str
    height: Optional[int]
    width: Optional[int]
    num_inference_steps: Optional[int]
    num_images_per_prompt: Optional[int]


model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe = pipe.to("cuda")


def predict(item, run_id, logger):
    images = pipe(
        prompt=item["prompt"],
        height=getattr(item, "height", 512),
        width=getattr(item, "width", 512),
        num_images_per_prompt=getattr(item, "num_images_per_prompt", 1),
        num_inference_steps=getattr(item, "num_inference_steps", 25)
    ).images

    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return finished_images
