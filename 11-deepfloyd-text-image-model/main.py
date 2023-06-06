import base64
from io import BytesIO
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

from pydantic import BaseModel



########################################################################
# Insert your parameters here.
########################################################################
class Item(BaseModel):
    prompt : str
    # image:  Optional[]
    # url:  Optional[str]


########################################################################
# Run once on first deployment to initialise the model
########################################################################

access_token = '<<<<Your HuggingFace Access Token Here>>>>'

# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16",
                                             torch_dtype=torch.float16, use_auth_token=access_token)
stage1 = stage_1.to("cuda")


# stage 2
stage_2 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0", text_encoder=None,
                                             variant="fp16", torch_dtype=torch.float16, use_auth_token=access_token)
stage2 = stage_2.to("cuda")

# stage 3
safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler",
                                             **safety_modules, torch_dtype=torch.float16, use_auth_token=access_token)
stage3 = stage_3.to("cuda")



########################################################################
# Run every time your model is called.
########################################################################

def predict(item, run_id, logger):
    item = Item(**item)
    ##Do something with parameters from item
    # Retrieve text embeddings
    prompt_embeds, negative_embeds = stage_1.encode_prompt(Item.prompt)

    logger.info(f'Starting with prompt: {Item.prompt}')

    # Running stage 1
    generator = torch.manual_seed(0)
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                     generator=generator, output_type="pt").images
    logger.info('Finished running stage 1')

    # Running stage 2
    image = stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                    generator=generator, output_type="pt").images
    logger.info('Finished running stage 2')

    # Running stage 3
    image = stage_3(prompt=Item.prompt, image=image, generator=generator, noise_level=100).images
    logger.info('Finished running stage 3')

    # Converting and returning the finished image.
    logger.info('Converting')
    buffered = BytesIO()
    image[0].save(buffered, format="PNG")
    finished_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    logger.info('Returning image')
    return finished_image