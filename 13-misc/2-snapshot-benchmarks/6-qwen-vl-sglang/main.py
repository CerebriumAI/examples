import multiprocessing
import os

multiprocessing.set_start_method("spawn", force=True)

import base64
import signal
import tempfile

# Faster HF weight download — the ~16GB Qwen2-VL-7B download is the bulk of the
# cold-start init, which must finish (load + checkpoint) within Cerebrium's 830s
# init cap. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

# --- gVisor checkpoint-friendliness ------------------------------------------
# Put IPC sockets on the gVisor-internal tmpfs (mounted by the shim) so they
# survive checkpoint/restore. /tmp is on overlayfs and gets remapped on restore.
os.environ.setdefault("VLLM_RPC_BASE_PATH", "/run/cuda-ckpt")
os.makedirs("/run/cuda-ckpt", exist_ok=True)
# NCCL HeartbeatMonitor spams "Broken pipe" on the TCPStore after restore.
# Disable monitoring (HEARTBEAT_TIMEOUT_SEC=0 means "fire immediately", so use a
# long timeout instead).
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
# torch._inductor eagerly forks N compile workers (N=os.cpu_count()) after CUDA
# init; each inherits nvidia FDs without a CUDA context, which cuda-checkpoint
# can't release and gVisor's nvproxy panics on. Compile synchronously (no fork).
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# Cerebrium's Cortex runtime imports main.py from a worker thread, but SGLang's
# engine launch calls signal.signal(SIGQUIT, ...), which Python only allows on
# the main thread ("signal only works in main thread"). That handler is just a
# launch-time debug aid, so make signal registration a no-op when it can't run.
_orig_signal = signal.signal


def _safe_signal(signalnum, handler):
    try:
        return _orig_signal(signalnum, handler)
    except ValueError:
        return None


signal.signal = _safe_signal

import http.client
import urllib.request

import requests
from PIL import Image
import sglang as sgl
from transformers import AutoProcessor

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

# --- Cold-start work that we want the snapshot to capture ----------------------
# Spinning up the SGLang engine (loading the 7B vision weights onto the GPU) is
# the expensive part of the cold start. We use the in-process offline
# `sgl.Engine` so the load happens at import time, before the checkpoint, rather
# than in a separate server process.
#
# disable_cuda_graph mirrors vLLM's enforce_eager: torch.compile / CUDA graph
# capture fail inside gVisor, so they must be turned off for the checkpoint to
# succeed.
processor = AutoProcessor.from_pretrained(MODEL_PATH)
engine = sgl.Engine(
    model_path=MODEL_PATH,
    mem_fraction_static=0.8,
    # Memory-saver lets us free the GPU before the checkpoint and reload after
    # restore (see release/resume below). This mirrors the Modal variant
    # (--enable-memory-saver --enable-weights-cpu-backup) and the vLLM Cerebrium
    # variant's sleep(1)/wake_up(): the snapshot then captures CPU-backed weights
    # instead of ~16GB of resident GPU state, so restore becomes a fast CPU->GPU
    # copy rather than a full GPU-memory checkpoint restore through gVisor.
    enable_memory_saver=True,
    # Without this, release discards the weights instead of backing them up to
    # CPU RAM, and the restored weights are garbage.
    enable_weights_cpu_backup=True,
)


def _build_prompt(prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ""},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _infer(image_path: str, prompt: str, max_tokens: int):
    return engine.generate(
        prompt=_build_prompt(prompt),
        image_data=[image_path],
        sampling_params={"max_new_tokens": max_tokens, "temperature": 0.2},
    )


# Warm up generation with a tiny in-memory image BEFORE the checkpoint. This
# populates the weights/KV cache and JIT-compiles hot kernels into the snapshot
# (so restores are fast) and verifies the engine actually generates — if
# generation hangs, init fails loudly here instead of silently hanging the first
# request.
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as _wf:
    Image.new("RGB", (64, 64), (127, 127, 127)).save(_wf, format="JPEG")
    _warmup_path = _wf.name
try:
    _infer(_warmup_path, "Warmup.", max_tokens=2)
    print("Warmup generation OK", flush=True)
finally:
    os.remove(_warmup_path)

# --- Snapshot disabled for no-snapshot benchmark (re-enable later) ------------
# The release/checkpoint/resume cycle below only exists to take and restore the
# GPU snapshot. With it commented out the weights stay resident on the GPU and
# the replica serves straight after warmup — a true cold-start (no-snapshot) path.
#
# # Free the GPU before the checkpoint: back the weights up to CPU RAM and release
# # the GPU allocation (KV cache + weights). The snapshot then captures the CPU
# # copy instead of resident GPU memory, so restore is a fast CPU->GPU reload
# # (resume below) rather than a full GPU-memory restore through gVisor.
print("releasing memory occupation before checkpoint", flush=True)
engine.release_memory_occupation()
print("released memory occupation", flush=True)
#
# # --- Trigger Cerebrium's memory checkpoint -----------------------------------
# # Cerebrium equivalent of Modal's `enable_memory_snapshot` + GPU snapshot. On the
# # creation run the POST returns the checkpoint response; on a restored replica
# # the snapshotted TCP connection is dead and the call raises RemoteDisconnected.
# print("calling checkpoint", flush=True)
try:
    req = urllib.request.Request("http://169.254.169.253:8234/checkpoint", method="POST")
    with urllib.request.urlopen(req) as response:
        print("Checkpointed successfully", flush=True)
        print("Response:", response.read().decode("utf-8"), flush=True)
except http.client.RemoteDisconnected:
    # TCP connections disconnect on restore and throw RemoteDisconnected.
    pass
#
# # Reload the weights from the CPU backup onto the GPU after restore (and on the
# # creation run, right after checkpointing) so the replica can serve immediately.
print("resuming memory occupation", flush=True)
engine.resume_memory_occupation()
print("resumed memory occupation", flush=True)

# Reachable https demo image (HF-hosted). The http COCO host and the Alibaba
# Beijing OSS host are slow/blocked from this region and stall the image fetch.
DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)


def run(
    image_url: str = DEFAULT_IMAGE_URL,
    image_base64: str = None,
    prompt: str = "Describe this image in detail.",
    max_tokens: int = 256,
):
    # Resolve the image to a local file ourselves (with a timeout) and hand
    # SGLang a path — most reliable input, and avoids SGLang's URL fetch hanging
    # indefinitely on an unreachable host.
    if image_base64:
        data = base64.b64decode(image_base64)
    else:
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        data = resp.content

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data)
        path = f.name

    try:
        output = _infer(path, prompt, max_tokens)
    finally:
        os.remove(path)

    return {"text": output["text"]}
