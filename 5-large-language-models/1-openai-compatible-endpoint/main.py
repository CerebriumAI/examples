import json
import os
import time


from huggingface_hub import login
from pydantic import BaseModel
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

login(token=os.environ.get("HF_TOKEN"))

engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)


class Message(BaseModel):
    role: str
    content: str


async def run(
    messages: list,
    model: str,
    run_id: str,
    stream: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 4096,
):
    prompt = " ".join(
        [f"{Message(**msg).role}: {Message(**msg).content}" for msg in messages]
    )
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
    results_generator = engine.generate(prompt, sampling_params, run_id)

    previous_text = ""
    first_chunk = True

    async for output in results_generator:
        prompt_output = output.outputs
        new_text = prompt_output[0].text[len(previous_text) :]
        previous_text = prompt_output[0].text

        chunk = {
            "id": run_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }

        # Include role in the first chunk
        if first_chunk:
            chunk["choices"][0]["delta"]["role"] = "assistant"
            first_chunk = False

        # Add new text to delta if there is any
        if new_text:
            chunk["choices"][0]["delta"]["content"] = new_text

        # Check for a finish_reason
        finish_reason = prompt_output[0].finish_reason
        if finish_reason and finish_reason != "none":
            chunk["choices"][0]["finish_reason"] = finish_reason

        yield f"data: {json.dumps(chunk)}\n\n"

    # After all chunks, send [DONE]
    yield "data: [DONE]\n\n"
