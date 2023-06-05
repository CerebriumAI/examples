import pandas as pd
from pydantic import BaseModel
import sys
import os.path as osp
import os
from typing import TypedDict, Optional
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from pathlib import Path

import torch

# Loading in base model and tokenizer
base_model_name = "decapoda-research/llama-7b-hf"  # Hugging Face Model Id
base_model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference


# Get input params
class Item(BaseModel):
    prompt: str
    cutoff_len: Optional[int] = 256
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.75
    top_k: Optional[float] = 40
    num_beams: Optional[int] = 4
    max_new_tokens: Optional[int] = 256



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


def predict(item, run_id, logger):
    item = Item(**item)
    result = generate(params=item)
    return {"Prediction": result}
