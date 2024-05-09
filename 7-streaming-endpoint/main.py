from typing import Optional

import torch
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

modal_path = "tiiuae/falcon-7b-instruct"

# Loading in base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(modal_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    modal_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)


class Item(BaseModel):
    prompt: str
    cutoff_len: Optional[int] = 256
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.75
    top_k: Optional[float] = 40
    max_new_tokens: Optional[int] = 250


def predict(prompt, cutoff_len, temperature, top_p, top_k, max_new_tokens):
    item = Item(prompt=prompt, cutoff_len=cutoff_len, temperature=temperature, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens)
    inputs = tokenizer(
        item.prompt, return_tensors="pt", max_length=512, truncation=True, padding=True
    )
    input_ids = inputs["input_ids"].to("cuda")

    streamer = TextIteratorStreamer(tokenizer)
    generation_config = GenerationConfig(
        temperature=item.temperature,
        top_p=item.top_p,
        top_k=item.top_k,
    )
    with torch.no_grad():
        generation_kwargs = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": tokenizer.eos_token_id,
            "max_new_tokens": item.max_new_tokens,
            "streamer": streamer,
        }
        model.generate(**generation_kwargs)
        for text in streamer:
            yield text  # vital for streaming
