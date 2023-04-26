import textwrap
from typing import Optional
from pydantic import BaseModel
import sys
from pathlib import Path

models_path = Path("..") / "models"
sys.path.append(str(models_path.resolve()))
from nomic.gpt4all.gpt4all import GPT4AllGPU

sys.path.remove(str(models_path.resolve()))

class Item(BaseModel):
    prompt: str
    num_beams: Optional[int] = 2
    min_new_tokens: Optional[int] = 10
    max_length: Optional[int] = 100
    repetition_penalty: Optional[float] = 2.0

model = GPT4AllGPU("decapoda-research/llama-7b-hf")

def predict(item, run_id, logger):
    item = Item(**item)

    config = {
        'num_beams': item.num_beams,
        'min_new_tokens': item.min_new_tokens,
        'max_length': item.max_length,
        'repetition_penalty': item.repetition_penalty
    }
    out = model.generate(item.prompt, config)

    return {"result": out}