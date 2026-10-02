import http.client
import json
import os
import time
import urllib.request

from huggingface_hub import login
from sglang import Runtime

CHECKPOINT_URL = "http://169.254.169.253:8234/checkpoint"

os.environ["HF_TRANSFER"] = "1"
os.environ["HF_HUB_VERBOSE"] = "1"
os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "1"


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

# model_id = "deepseek-ai/DeepSeek-R1"  # uncomment for R1
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

print("[init] building SGLang runtime", flush=True)
runtime = Runtime(
    model_path=model_id,
    tp_size=1,  # change tp_size=8 if serving R1 on H200
)
print("[init] SGLang runtime ready", flush=True)

_trigger_snapshot()
print("[init] handler ready", flush=True)


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
    first_chunk = True

    async for output in stream:
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

        if output:
            chunk["choices"][0]["delta"]["content"] = output

        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"
