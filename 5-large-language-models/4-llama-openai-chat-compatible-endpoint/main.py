import http.client
import json
import os
import time
import urllib.request

from huggingface_hub import login
from pydantic import BaseModel
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

CHECKPOINT_URL = "http://169.254.169.253:8234/checkpoint"


def _trigger_snapshot() -> None:
    print("[init] requesting GPU snapshot", flush=True)
    try:
        req = urllib.request.Request(CHECKPOINT_URL, method="POST")
        urllib.request.urlopen(req, timeout=300)
        print("[init] snapshot complete", flush=True)
    except http.client.RemoteDisconnected:
        # TCP connections disconnect on restore and throw remote
        print("[init] snapshot complete (RemoteDisconnected)", flush=True)
    except Exception as exc:
        print(f"[init] snapshot failed: {type(exc).__name__}: {exc}", flush=True)


print("[init] starting", flush=True)
login(token=os.environ.get("HF_TOKEN"))

engine_args = AsyncEngineArgs(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    async_scheduling=False,
)

print("[init] building vLLM engine", flush=True)
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("[init] vLLM engine ready", flush=True)

_trigger_snapshot()
print("[init] handler ready", flush=True)


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
    return (
        "<|begin_of_text|>"
        + "".join(formatted_messages)
        + "<|start_header_id|>assistant<|end_header_id|>"
    )


async def run(
    messages: list,
    model: str,
    run_id: str,
    stream: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 4096,
):
    prompt = format_chat_prompt(messages)

    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
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

        if first_chunk:
            chunk["choices"][0]["delta"]["role"] = "assistant"
            first_chunk = False

        if new_text:
            chunk["choices"][0]["delta"]["content"] = new_text

        finish_reason = prompt_output[0].finish_reason
        if finish_reason and finish_reason != "none":
            chunk["choices"][0]["finish_reason"] = finish_reason

        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"
