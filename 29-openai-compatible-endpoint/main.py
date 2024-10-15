import json
import os
import time
from typing import List, Dict, Any

from huggingface_hub import login
from pydantic import BaseModel
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

login(token=os.environ.get("HF_TOKEN"))

engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.9,  # Increase GPU memory utilization
    max_model_len=8192,  # Decrease max model length
)
engine = AsyncLLMEngine.from_engine_args(engine_args)


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]


async def run(
    messages: List[Message],
    model: str,
    run_id: str,
    stream: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
):
    prompt = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
    results_generator = engine.generate(prompt, sampling_params, run_id)
    previous_text = ""
    full_text = ""  # Collect all generated text here

    async for output in results_generator:
        prompt = output.outputs
        new_text = prompt[0].text[len(previous_text) :]
        previous_text = prompt[0].text
        full_text += new_text  # Append new text to full_text

        response = ChatCompletionResponse(
            id=run_id,
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "text": new_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": prompt[0].finish_reason or "stop",
                }
            ],
        )
        yield json.dumps(response.model_dump())
