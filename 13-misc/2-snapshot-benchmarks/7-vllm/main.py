import http.client
import json
import time
import urllib.request

from pydantic import BaseModel
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


engine_args = AsyncEngineArgs(
    model=MODEL,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)


try:
    req = urllib.request.Request("http://169.254.169.253:8234/checkpoint", method="POST")
    with urllib.request.urlopen(req) as response:
        response_body = response.read().decode("utf-8")
        print("Checkpointed successfully")
        print("Response:", response_body)
except http.client.RemoteDisconnected:
    # TCP connections disconnect on restore and throw remote
    pass


class Message(BaseModel):
    role: str
    content: str


async def run(
    messages: list,
    model: str = MODEL,
    run_id: str = "",
    stream: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 4096,
):
    prompt = " ".join(
        [f"{Message(**msg).role}: {Message(**msg).content}" for msg in messages]
    )
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
