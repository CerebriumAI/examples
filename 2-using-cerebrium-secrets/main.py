from pydantic import BaseModel
from typing import Optional
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, LlamaTokenizer, GenerationConfig
import huggingface_hub
from cerebrium import get_secret # <-- import secrets from cerebrium

########################################
# Using a Cerebrium secret
########################################
try: 
    """
    To access your cerebrium secret:
    1. create a secret in your projects dashboard
    2. Take note of the name, 
        for this example, we'll use a secret called `hf_auth_token`
    3. add the code in the line below to access your secret
    """
    hf_auth_token = get_secret("hf_auth_token") # load your secret
    
    # And that's all! It's that easy.

    if hf_auth_token == "":
        raise Exception("hf_auth_token is empty. You need a hf_auth_token secret added to your account to access this model.")
except Exception as e:
    print("\n\n")
    print("="*60)
    print("Error: ", e)
    print("="*60)
    raise e

huggingface_hub.login(token=hf_auth_token)

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

base_model_name =  'meta-llama/Llama-2-7b-hf'  # Hugging Face Model Id
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference


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
