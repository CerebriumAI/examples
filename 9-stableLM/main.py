from typing import Optional
from pydantic import BaseModel
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch

class Item(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.7
    do_sample: Optional[bool] = True,

tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-3b")
model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-tuned-alpha-3b")
model.half().to(torch.cuda.current_device())

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

def predict(item, run_id, logger):
    item = Item(**item)

    prompt = f"{system_prompt}{item.prompt}"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=item.max_new_tokens,
        temperature=item.temperature,
        do_sample=item.do_sample,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    return {"result": tokenizer.decode(tokens[0], skip_special_tokens=True)}
