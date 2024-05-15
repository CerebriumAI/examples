from typing import Optional
from pydantic import BaseModel
from transformers_neuronx import NeuronAutoModelForCausalLM
import torch
import time
from transformers import AutoTokenizer, TextStreamer
from huggingface_hub import login
import os


class Item(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95


login(token="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")
os.environ["NEURON_COMPILE_CACHE_URL"] = (
    "/persistent-storage/plain-neuron-compile-cache"
)

name = "mistralai/Mistral-7B-Instruct-v0.2"

model = NeuronAutoModelForCausalLM.from_pretrained(
    name,  # The reference to the huggingface model
    tp_degree=1,  # The Number of NeuronCores to shard the model across. Using 8 means 3 replicas can be used on a inf2.48xlarge
    amp="bf16",  # Ensure the model weights/compute are bfloat16 for faster compute
)
model.to_neuron()

tokenizer = AutoTokenizer.from_pretrained(name)
streamer = TextStreamer(tokenizer)


def predict(prompt, temperature=0.8, top_p=0.95):
    item = Item(prompt=prompt, temperature=temperature, top_p=top_p)

    # Start timing
    start_time = time.time()

    input_ids = tokenizer.encode(item.prompt, return_tensors="pt")

    with torch.inference_mode():
        generated_sequences = model.sample(
            input_ids, sequence_length=2048, top_k=50, streamer=streamer
        )

    # End timing
    end_time = time.time()

    print(f"generated_sequences: {generated_sequences}")

    print(f"Processed in {end_time-start_time} seconds")

    return {"result": generated_sequences}
