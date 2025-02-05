import os
import torch
from transformers import AutoTokenizer
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.config import NeuronConfig, QuantizationConfig, GenerationConfig
from transformers_neuronx.config import GenerationConfig 
from transformers_neuronx import GQA
from transformers import LlamaForCausalLM, TextIteratorStreamer
from huggingface_hub import login, logging
from threading import Thread
import time
from pydantic import BaseModel
from typing import List, Optional
from threading import Thread
import json
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import LLM, SamplingParams

logging.set_verbosity_info()

os.environ["HF_TRANSFER"] = "1"
os.environ["HF_HUB_VERBOSE"] = "1"
os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "1"

class Message(BaseModel):
    role: str
    content: str

def format_chat_prompt(messages: list) -> str:
    formatted_messages = []
    for msg in messages:
        msg_obj = Message(**msg)
        formatted_messages.append(
            f"<|start_header_id|>{msg_obj.role}<|end_header_id|>\n{msg_obj.content}<|eot_id|>"
        )
    return "<|begin_of_text|>" + "".join(formatted_messages) + "<|start_header_id|>assistant<|end_header_id|>"


login(token=os.environ.get("HF_TOKEN"))

llm = None
async def run(
    messages: list,
    model: str = "meta-llama/Llama-3.1-70B-Instruct",
    run_id: str = None,
    stream: bool = True,
    temperature: float = 0.8,
    max_tokens: int = 128,
):

    global llm  # Declare we're using the global llm variable
    if llm is None:
        print("Initializing LLM...") 
        llm = LLM(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct", 
            max_num_seqs=1,
            max_model_len=4092,
            block_size=4092,
            device="neuron",
            tensor_parallel_size=32,
            gpu_memory_utilization=0.9
        )
        print("Done initializing LLM...") 
    prompt = format_chat_prompt(messages)
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    outputs = llm.generate(prompt, sampling_params)

    previous_text = ""
    first_chunk = True

    async for output in outputs:
        prompt_output = output.outputs
        new_text = prompt_output[0].text[len(previous_text) :]
        previous_text = prompt_output[0].text

        # Construct OpenAI-compatible chunk
        chunk = {
            "id": run_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None,
                }
            ],
        }

        # Include the role in the first chunk
        if first_chunk:
            chunk["choices"][0]["delta"]["role"] = "assistant"
            first_chunk = False

        # Add new text to the delta if any
        if new_text:
            chunk["choices"][0]["delta"]["content"] = new_text

        # Capture a finish reason if it's provided
        finish_reason = prompt_output[0].finish_reason or None
        if finish_reason and finish_reason != "none":
            chunk["choices"][0]["finish_reason"] = finish_reason

        print(f"data: {json.dumps(chunk)}\n\n")
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send the final [DONE] message
    yield "data: [DONE]\n\n"