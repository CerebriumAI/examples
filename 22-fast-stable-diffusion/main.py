import base64
import io
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pydantic import BaseModel


class Item(BaseModel):
    prompt: str
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 25
    num_images_per_prompt: Optional[int] = 1


model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)
print("Loaded.")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
print("Loaded scheduler.")
pipe.enable_xformers_memory_efficient_attention()
print("Enabled memory efficient attention.")
pipe = pipe.to("cuda")
print("Moved to GPU.")

print(pipe.device)


def predict(item):
    item = Item(**item)

    images = pipe(
        prompt=item.prompt,
        height=item.height,
        width=item.width,
        num_images_per_prompt=item.num_images_per_prompt,
        num_inference_steps=item.num_inference_steps,
    ).images
    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
    return finished_images
