
from typing import Optional
from pydantic import BaseModel
import os
import base64
import io
import torch
from diffusers import StableDiffusionXLPipeline
from onediffx import compile_pipe, save_pipe, load_pipe

pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
pipe.to("cuda")
pipe = compile_pipe(pipe)


if not os.path.isdir("/persistent-storage/cached_pipe"):
    ##Run before saving
    image = pipe(
        prompt="street style, detailed, raw photo, woman, face, shot on CineStill 800T",
        height=1024,
        width=1024,
        num_inference_steps=30,
        output_type="pil",
    ).images
    print("Pipe compiled:", pipe)
    save_pipe(pipe, dir="/persistent-storage/cached_pipe")
else:
    print("Loading:", pipe)
    pipe = load_pipe(pipe, dir="/persistent-storage/cached_pipe")
    print("Pipe loaded:", pipe)



class Item(BaseModel):
    prompt: str
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 30

def predict(prompt, height=512, width=512, num_inference_steps=30):
    item = Item(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps
    )

    # run once to trigger compilation
    images = pipe(
        prompt=item.prompt,
        height=item.height,
        width=item.width,
        num_inference_steps=item.num_inference_steps,
        output_type="pil",
    ).images

    finished_images = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

    return {"images": finished_images}
