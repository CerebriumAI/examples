import os
import time
from typing import Optional

from huggingface_hub import login
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["NEURON_COMPILE_CACHE_URL"] = "/persistent-storage/vllm-neuron-cache"


class Item(BaseModel):
    prompts: list[str] = []
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95


login(token=os.environ.get("HF_AUTH_TOKEN"))

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm = LLM(
    model=model_id,
    max_num_seqs=1,
    max_model_len=128,
    block_size=128,
    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    tensor_parallel_size=2,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)


def predict(prompts, temperature=0.8, top_p=0.95):
    item = Item(prompts=prompts, temperature=temperature, top_p=top_p)

    # Start timing
    start_time = time.time()

    sampling_params = SamplingParams(temperature=item.temperature, top_p=item.top_p)
    outputs = llm.generate(item.prompts, sampling_params)

    # End timing
    end_time = time.time()

    total_tokens = 0
    generated_texts = []  # List to store all generated texts
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        generated_texts.append(generated_text)

        # Assuming you have a way to count tokens, e.g., a function count_tokens(text)
        total_tokens += count_tokens(prompt) + count_tokens(generated_text)

    # Calculate duration and tokens per second
    duration = end_time - start_time
    tokens_per_second = total_tokens / duration if duration > 0 else 0

    print(
        f"Processed {total_tokens} tokens in {duration:.2f} seconds ({tokens_per_second:.2f} tokens/second)"
    )

    return {"outputs": generated_texts, "tokens_per_second": tokens_per_second}


def count_tokens(text):
    return len(tokenizer.tokenize(text))
