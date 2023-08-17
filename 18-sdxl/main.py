from typing import Optional
from pydantic import BaseModel
import torch
import base64
import io
from diffusers import DiffusionPipeline

class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    steps: Optional[int] = 40
    high_noise_frac: Optional[float] = 0.8

base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

def predict(item, run_id, logger):
    item = Item(**item)

    # run both experts
    image = base(
        prompt=item.prompt,
        num_inference_steps=item.steps,
        denoising_end=item.high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=item.prompt,
        num_inference_steps=item.steps,
        denoising_start=item.high_noise_frac,
        image=image,
    ).images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    return base64.b64encode(buffered.getvalue()).decode("utf-8")

