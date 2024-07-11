import base64
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from diffusers import AutoencoderKL, DiffusionPipeline
from pydantic import BaseModel

torch.backends.cuda.matmul.allow_tf32 = True

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

pipe.unet.to(memory_format=torch.channels_last)
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")
refiner.enable_xformers_memory_efficient_attention()


class Item(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    num_inference_steps: Optional[int] = 30
    guidance_scale: Optional[float] = 7.5
    num_images_per_prompt: Optional[int] = 1
    use_refiner: Optional[bool] = True
    denoising_frac: Optional[float] = 0.8
    end_cfg_frac: Optional[float] = 0.4
    seed: Optional[int] = None


def convert_to_b64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64


def predict(
    prompt,
    negative_prompt=None,
    height=512,
    width=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    num_images_per_prompt=1,
    use_refiner=True,
    denoising_frac=0.8,
    end_cfg_frac=0.4,
    seed=None,
):
    item = Item(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        use_refiner=use_refiner,
        denoising_frac=denoising_frac,
        end_cfg_frac=end_cfg_frac,
        seed=seed,
    )

    generator = None
    if item.seed is not None:
        torch.manual_seed(item.seed)
        generator = [torch.Generator(device="cuda").manual_seed(item.seed)]

    if not item.use_refiner:
        item.denoising_frac = 1.0

    image = pipe(
        prompt=item.prompt,
        negative_prompt=item.negative_prompt,
        width=item.width,
        height=item.height,
        generator=generator,
        end_cfg=item.end_cfg_frac,
        num_inference_steps=item.num_inference_steps,
        denoising_end=item.denoising_frac,
        guidance_scale=item.guidance_scale,
        output_type="latent" if item.use_refiner else "pil",
    ).images[0]

    scheduler = pipe.scheduler
    if item.use_refiner:
        refiner.scheduler = scheduler
        image = refiner(
            prompt=item.prompt,
            negative_prompt=item.negative_prompt,
            generator=generator,
            end_cfg=item.end_cfg_frac,
            num_inference_steps=item.num_inference_steps,
            denoising_start=item.denoising_frac,
            guidance_scale=item.guidance_scale,
            image=image[None, :],
        ).images[0]

    b64_results = convert_to_b64(image)

    return {"status": "success", "data": b64_results}
