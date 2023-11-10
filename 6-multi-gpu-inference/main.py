"""
This main.py is a brief guide on how to implement multi-gpu using Cerebrium.
Some prerequisites:
    - you need to have a "hf_auth_token" secret added to your account containing your huggingface auth token with access to llama2-70b
    - You need to have the latest version of cerebrium installed (pip install cerebrium --upgrade)

Note:
    - Make sure you deploy using the --config-file flag and point it to the config.yaml file in this directory.
    - larger models by nature have more calculations that need to be performed and will take longer to run.
    - If you're using GPU's that do not have nvlink, inter-gpu communication will be slower but it will still work.
    - To speedup loading of model weights, take a look at the folder in this repo which shows you how to speedup loading of model weights.
"""
import os
from typing import Optional

import huggingface_hub
import torch
from accelerate import Accelerator
from cerebrium import get_secret
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, LlamaTokenizer, GenerationConfig

os.environ[
    "HF_HUB_ENABLE_HF_TRANSFER"
] = "true"  # to enable faster weights download on first time build

# initialise accelerator. We'll use this as an easier way to get some speedup
accelerator = Accelerator()

# Loading in base model and tokenizer
base_model_name = "meta-llama/Llama-2-70b-hf"  # Hugging Face Model Id
try:
    hf_auth_token = get_secret("hf_auth_token")
    if hf_auth_token == "":
        raise Exception(
            "hf_auth_token is empty. You need a hf_auth_token secret added to your account to access this model."
        )
except Exception as e:
    print("\n\n")
    print("=" * 60)
    print("Error: ", e)
    print("=" * 60)
    raise e

huggingface_hub.login(token=hf_auth_token)
print("Loading hf model... this could take a while. Sit tight!")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference

if hasattr(base_model, "to_bettertransformer"):
    print("Converting to BetterTransformer")
    base_model.to_bettertransformer()
base_model = accelerator.prepare(base_model)


########################################
# User-facing API Parameters
########################################
class Item(BaseModel):
    prompt: str
    cutoff_len: Optional[int] = 256
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.75
    top_k: Optional[float] = 40
    num_beams: Optional[int] = 4
    max_new_tokens: Optional[int] = 256


#######################################
# Initialize the model
#######################################
def tokenize(prompt, cutoff_len, add_eos_token=True):
    print("tokenizing: ", prompt)
    return tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors="pt",
    )


def generate(params: Item):
    print("Placing prompt in template")
    prompt_input = tokenize(prompt=params.prompt, cutoff_len=params.cutoff_len)
    input_ids = prompt_input["input_ids"]
    input_ids = input_ids.to(base_model.device)

    print("Setting up generation config")
    generation_config = GenerationConfig(
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        num_beams=params.num_beams,
        max_new_tokens=params.max_new_tokens,
    )
    with torch.no_grad():
        outputs = base_model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)


#######################################
# Prediction
#######################################
def predict(item, run_id, logger):
    item = Item(**item)
    result = generate(params=item)
    return {"Prediction": result}
