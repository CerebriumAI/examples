import os

# # Ensure micromamba's libs come first in LD_LIBRARY_PATH so that spawn children
# # find the right libstdc++ before the system one.
ld = os.environ.get("LD_LIBRARY_PATH", "")
mm_lib = "/micromamba/envs/3.11/lib"
if mm_lib not in ld.split(":"):
    os.environ["LD_LIBRARY_PATH"] = mm_lib + ":" + ld

# # Bind NCCL TCPStore to localhost so it survives checkpoint/restore.
os.environ["VLLM_HOST_IP"] = "127.0.0.1"

# # Spawn mode: EngineCore starts fresh (fork+exec) so it doesn't inherit the
# # parent's nvidia FDs — fork-inherited FDs aren't tracked by any CUDA session,
# # so cuda-checkpoint --toggle can't close them, causing nvproxy.beforeSaveImpl
# # to panic with "frontendFD is not saveable".
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# # Disable flashinfer sampler — it JIT-compiles CUDA kernels and needs nvcc.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# # Disable NCCL heartbeat monitor — it spams "Broken pipe" on the TCPStore after
# # restore. Safe to skip for TP=1 (single-GPU).
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")

# # Put ZMQ IPC sockets on the gVisor-internal tmpfs so they survive restore.
# # /tmp is on overlayfs and gets remapped on restore, killing Unix-domain sockets.
os.environ.setdefault("VLLM_RPC_BASE_PATH", "/run/cuda-ckpt")
os.makedirs("/run/cuda-ckpt", exist_ok=True)

# # Cache weights / compiled artifacts on Cerebrium persistent storage so restarts
# # skip the re-download.
os.environ.setdefault("HF_HOME", "/persistent-storage/huggingface")
os.environ.setdefault("VLLM_CACHE_ROOT", "/persistent-storage/")


import asyncio
import http.client
import json
import threading
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
    # async_scheduling (default in v0.21+) spawns an extra worker holding CUDA
    # state that cuda-checkpoint can't round-trip. Keep the PID set stable.
    async_scheduling=False,
    # Required for sleep(1)/wake_up() — the cumem allocator offloads the weights
    # to CPU and frees the GPU on sleep, then restores them on wake.
    enable_sleep_mode=True,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# vLLM V1's AsyncLLM is async-only: sleep(), wake_up() and generate() are all
# coroutines. Its engine-core binds zmq.asyncio sockets + the reply-queue task
# to the FIRST event loop that touches them and never rebinds. Cortex imports
# main.py in an executor thread with no loop, then serves run() on a *different*
# loop/thread, so every engine call must land on ONE loop. Pin that loop to a
# dedicated thread and funnel sleep/wake_up (here) and generate (in run())
# through it.
_engine_loop = asyncio.new_event_loop()
threading.Thread(target=_engine_loop.run_forever, daemon=True).start()


def _engine_call(coro):
    """Run an engine coroutine on the engine loop from any thread and block."""
    return asyncio.run_coroutine_threadsafe(coro, _engine_loop).result()


# Warm up hot Triton kernels (e.g. _compute_slot_mapping_kernel) while the engine
# is still awake, so they're JIT-compiled and cached before the snapshot instead
# of on the first request after restore.
async def _warmup():
    async for _ in engine.generate(
        "warmup", SamplingParams(temperature=0, max_tokens=8), "warmup"
    ):
        pass


_engine_call(_warmup())

# --- Snapshot disabled for no-snapshot benchmark (re-enable later) ------------
# The sleep/checkpoint/wake cycle below only exists to take and restore the GPU
# snapshot. With it commented out the weights stay resident on the GPU and the
# replica serves straight after warmup — a true cold-start (no-snapshot) path.
#
# # Release ALL GPU memory before the checkpoint. vLLM's sleep(level=1) offloads
# # the weights to CPU RAM and discards the KV cache, so nothing holds the GPU when
# # Cerebrium freezes the sandbox (the same thing Modal does before its GPU
# # snapshot — see vllm_low_latency.py). The offloaded weights live in host RAM,
# # which the memory checkpoint captures; wake_up() copies them back to the GPU
# # after restore. Keeping weights resident on the GPU across the freeze leaves a
# # live GPU allocation that the checkpoint can't cleanly capture.
print("sleeping engine (offload weights to CPU, free GPU)", flush=True)
_engine_call(engine.sleep(level=1))
#
# # Trigger Cerebrium's checkpoint with a single POST. On the creation run the POST
# # returns the checkpoint response; on a restored replica the snapshotted TCP
# # connection is dead and the call raises RemoteDisconnected — either way we fall
# # through and wake the engine so the replica can serve.
print("calling checkpoint", flush=True)
try:
    req = urllib.request.Request("http://169.254.169.253:8234/checkpoint", method="POST")
    with urllib.request.urlopen(req) as response:
        print("Checkpointed successfully")
        print("Response:", response.read().decode("utf-8"))
except http.client.RemoteDisconnected:
    # TCP connections disconnect on restore and throw RemoteDisconnected.
    pass
#
print("waking engine (reload weights to GPU)", flush=True)
_engine_call(engine.wake_up())


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

    previous_text = ""
    first_chunk = True

    # generate() must run on the engine loop (see _engine_loop above), so drive
    # its async generator there and hand each item back to the server loop.
    agen = engine.generate(prompt, sampling_params, run_id)
    while True:
        try:
            output = await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(agen.__anext__(), _engine_loop)
            )
        except StopAsyncIteration:
            break
        prompt_output = output.outputs
        new_text = prompt_output[0].text[len(previous_text):]
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
