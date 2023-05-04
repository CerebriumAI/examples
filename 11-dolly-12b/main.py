import textwrap
from typing import Optional
from pydantic import BaseModel
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

models_path = Path("..") / "models"
sys.path.append(str(models_path.resolve()))
from instruct_pipeline import InstructionTextGenerationPipeline
sys.path.remove(str(models_path.resolve()))


class Item(BaseModel):
    prompt: str
    num_beams: Optional[int] = 2
    min_new_tokens: Optional[int] = 10
    max_length: Optional[int] = 100
    repetition_penalty: Optional[float] = 2.0

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b", device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)
generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

def predict(item, run_id, logger):
    item = Item(**item)

    out = generate_text(item.prompt)

    return {"result": out}