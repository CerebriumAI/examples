import base64
import io
import os
import signal
import tempfile

# Faster HF weight download — the ~16GB Qwen2-VL-7B download is the bulk of the
# cold-start init, which must finish (load + checkpoint) within Cerebrium's
# 830s init cap. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

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
import urllib.error
import urllib.request

import requests
from PIL import Image
import sglang as sgl
from transformers import AutoProcessor

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

# --- Cold-start work that we want the snapshot to capture ----------------------
# Spinning up the SGLang engine (loading the 7B vision weights onto the GPU and
# capturing CUDA graphs) is the expensive part of the cold start. We use the
# in-process offline `sgl.Engine` so the load happens at import time, before the
# checkpoint, rather than in a separate server process.
processor = AutoProcessor.from_pretrained(MODEL_PATH)
engine = sgl.Engine(model_path=MODEL_PATH, mem_fraction_static=0.8)


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
# captures the CUDA graphs into the snapshot (so restores are fast) and verifies
# the engine actually generates — if generation hangs, init fails loudly here
# instead of silently hanging the first request.
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as _wf:
    Image.new("RGB", (64, 64), (127, 127, 127)).save(_wf, format="JPEG")
    _warmup_path = _wf.name
try:
    _infer(_warmup_path, "Warmup.", max_tokens=2)
    print("Warmup generation OK")
finally:
    os.remove(_warmup_path)

# --- Trigger Cerebrium's memory checkpoint -------------------------------------
# Cerebrium equivalent of Modal's `enable_memory_snapshot` + GPU snapshot.
# Log the endpoint's actual response so we can see what it reports, and tell
# snapshot-creation (first run, returns HTTP body) apart from snapshot-restore
# (the snapshotted TCP socket is dead on resume -> RemoteDisconnected).
CHECKPOINT_BASE = "http://169.254.169.253:8234"


def _checkpoint_status():
    """Read the REAL checkpoint outcome. POST /checkpoint returns a hardcoded
    'in_progress' even after the (synchronous) checkpoint finishes, so the only
    way to know success/error is GET /checkpoint/status."""
    try:
        sreq = urllib.request.Request(CHECKPOINT_BASE + "/checkpoint/status", method="GET")
        with urllib.request.urlopen(sreq) as sresp:
            return sresp.read().decode("utf-8", "replace").strip()
    except Exception as e:
        return f"<status query failed: {type(e).__name__}: {e}>"


print("Calling checkpoint endpoint...", flush=True)
try:
    req = urllib.request.Request(CHECKPOINT_BASE + "/checkpoint", method="POST")
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8", "replace").strip()
        print(f"Checkpoint POST: HTTP {resp.status} body={body!r}", flush=True)
    # POST blocks until the runsc checkpoint finishes but always reports
    # 'in_progress'. Query the real status to learn checkpointed vs error.
    print(f"Checkpoint REAL status: {_checkpoint_status()}", flush=True)
except http.client.RemoteDisconnected:
    # Resumed from the snapshot: the checkpoint socket no longer exists.
    print("RESTORED FROM SNAPSHOT (RemoteDisconnected on checkpoint socket)", flush=True)
except urllib.error.HTTPError as e:
    print(f"Checkpoint HTTPError {e.code}: {e.read().decode('utf-8', 'replace')!r}", flush=True)
except Exception as e:
    print(f"Checkpoint endpoint error: {type(e).__name__}: {e}", flush=True)

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
