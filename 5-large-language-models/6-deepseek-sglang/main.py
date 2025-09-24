# launch the offline engine
from huggingface_hub import login
from sglang import Runtime
from pydantic import BaseModel
import json
from typing import List, Dict, Any
import time
import os

os.environ["HF_TRANSFER"] = "1"
os.environ["HF_HUB_VERBOSE"] = "1"
os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "1"

login(token=os.environ.get("HF_TOKEN"))

# model_id = "deepseek-ai/DeepSeek-R1" ##uncomment for R1
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
runtime = Runtime(
    model_path=model_id, tp_size=1
)  # change tp_size=8 if serving R1 on H200


async def run(
    messages: list,
    model: str,
    run_id: str,
    stream: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 4096,
):

    sampling_params = {"temperature": temperature, "top_p": top_p}
    tokenizer = runtime.get_tokenizer()

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    stream = runtime.add_request(prompt, sampling_params)
    full_text = ""
    first_chunk = True

    async for output in stream:
        full_text += output

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

        if first_chunk:
            chunk["choices"][0]["delta"]["role"] = "assistant"
            first_chunk = False

        if output:
            chunk["choices"][0]["delta"]["content"] = output

        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"
