import os
import time
import torch
import subprocess
import numpy as np
from typing import List
from transformers import CLIPImageProcessor
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from io import BytesIO  # Correct import for BytesIO
import base64

UNET = "sdxl_lightning_4step_unet.pth"
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
UNET_CACHE = "/persistent-storage/unet-cache"
BASE_CACHE = "/persistent-storage/checkpoints"
SAFETY_CACHE = "/persistent-storage/safety-cache"
FEATURE_EXTRACTOR = "feature-extractor"
MODEL_URL = "https://weights.replicate.delivery/default/sdxl-lightning/sdxl-1.0-base-lightning.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
UNET_URL = "https://weights.replicate.delivery/default/comfy-ui/unet/sdxl_lightning_4step_unet.pth.tar"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DPM++2MSDE": KDPM2AncestralDiscreteScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


"""Load the model into memory to make running multiple predictions efficient"""
start = time.time()
print("Loading safety checker...")
if not os.path.exists(SAFETY_CACHE):
    download_weights(SAFETY_URL, SAFETY_CACHE)
print("Loading model")
if not os.path.exists(BASE_CACHE):
    download_weights(MODEL_URL, BASE_CACHE)
print("Loading Unet")
if not os.path.exists(UNET_CACHE):
    download_weights(UNET_URL, UNET_CACHE)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    SAFETY_CACHE, torch_dtype=torch.float16
).to("cuda")
feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)
print("Loading txt2img pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_BASE,
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir=BASE_CACHE,
    local_files_only=True,
).to("cuda")
unet_path = os.path.join(UNET_CACHE, UNET)
pipe.unet.load_state_dict(torch.load(unet_path, map_location="cuda"))
print("setup took: ", time.time() - start)


def run_safety_checker(image):
    safety_checker_input = feature_extractor(image, return_tensors="pt").to("cuda")
    np_image = [np.array(val) for val in image]
    image, has_nsfw_concept = safety_checker(
        images=np_image,
        clip_input=safety_checker_input.pixel_values.to(torch.float16),
    )
    return image, has_nsfw_concept


def predict(
    prompt: str = "A superhero smiling",
    negative_prompt: str = "worst quality, low quality",
    width: int = 1024,
    height: int = 1024,
    num_outputs: int = 1,
    scheduler: str = "K_EULER",
    num_inference_steps: int = 4,
    guidance_scale: float = 0,
    seed: int = None,
    disable_safety_checker: bool = False,
):
    """Run a single prediction on the model"""
    global pipe
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    print(f"Using seed: {seed}")
    generator = torch.Generator("cuda").manual_seed(seed)

    # OOMs can leave vae in bad state
    if pipe.vae.dtype == torch.float32:
        pipe.vae.to(dtype=torch.float16)

    sdxl_kwargs = {}
    print(f"Prompt: {prompt}")
    sdxl_kwargs["width"] = width
    sdxl_kwargs["height"] = height

    pipe.scheduler = SCHEDULERS[scheduler].from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    common_args = {
        "prompt": [prompt] * num_outputs,
        "negative_prompt": [negative_prompt] * num_outputs,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "num_inference_steps": num_inference_steps,
    }

    output = pipe(**common_args, **sdxl_kwargs)

    if not disable_safety_checker:
        _, has_nsfw_content = run_safety_checker(output.images)

    encoded_images = []
    for i, image in enumerate(output.images):
        if not disable_safety_checker:
            if has_nsfw_content[i]:
                print(f"NSFW content detected in image {i}")
                continue
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_images.append(img_b64)

    if len(encoded_images) == 0:
        raise Exception(
            "NSFW content detected. Try running it again, or try a different prompt."
        )

    return encoded_images
