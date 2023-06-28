import textwrap
from typing import Optional
from pydantic import BaseModel
import sys
from pathlib import Path

import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

models_path = Path("..") / "models"
sys.path.append(str(models_path.resolve()))
from instruct_pipeline import InstructionTextGenerationPipeline
sys.path.remove(str(models_path.resolve()))


class Item(BaseModel):
    prompt: str

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b",
                                          padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b",
                                             load_in_8bit= True,
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16)

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)


def predict(item, run_id, logger):
    item = Item(**item)

    out = generate_text(item.prompt)

    return {"result": out}