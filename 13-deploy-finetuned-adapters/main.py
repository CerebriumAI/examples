"""
This file serves as an example to using a finetuned adapter for inference with Llama.

In this example, the adapter is obtained through finetunining on Cerebrium using the`cerebrium train` command
Adapter files that have been obtained from the AdapterHub can be used in the same way.

Note: the cerebrium finetuning is still in beta testing. If you have any issues, please don't hesitate to contact the team.
"""
import json
from pydantic import BaseModel
from typing import Optional
import torch
from pydantic import BaseModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, __version__
from peft import PeftModel

import transformers
import peft


transformers.logging.set_verbosity_info()

print("need transformers>=4.29.0,  got :", transformers.__version__)
print("Peft version is :", peft.__version__)

# Loading in base model and tokenizer
base_model_name = "decapoda-research/llama-7b-hf"  # Hugging Face Model Id
base_model = LlamaForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference

adapter_path = "training-output/"
print("Loading in PEFT model")
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

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
    template =  "### Instruction:\n{instruction}\n\n### Response:\n"
    
    # read in contents of template.json
    with open("template.json", "r") as f:
        template = json.load(f)
    instruction = template["prompt_no_input"].format(instruction=params.prompt)

    prompt_input = tokenize(prompt=instruction, cutoff_len=params.cutoff_len)
    input_ids = prompt_input["input_ids"]
    input_ids = input_ids.to(peft_model.device)
    # input_ids = input_ids.to(base_model.device)

    print("Setting up generation config")
    generation_config = GenerationConfig(
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        num_beams=params.num_beams,
        max_new_tokens=params.max_new_tokens,
    )
    with torch.no_grad():
        outputs = peft_model.generate(
        # outputs = base_model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)



def predict(item, run_id, logger):
    item = Item(**item)

    print("Loading in PEFT model")

    print("Generating")
    result = generate(params=item)
    
    return {"Prediction": result}
