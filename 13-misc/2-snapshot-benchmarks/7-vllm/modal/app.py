"""Modal port of the vLLM (LLM serving) snapshot benchmark.

Mirror of ../main.py using Modal's `enable_memory_snapshot` + experimental
`enable_gpu_snapshot`. Following Modal's own vllm_low_latency.py reference, vLLM
runs as a `vllm serve` subprocess: it's started + warmed up + put to sleep inside
`@modal.enter(snap=True)` (so the snapshot captures clean CPU-only state with the
GPU freed), then woken in `@modal.enter(snap=False)` after restore. Driving the
engine over HTTP avoids vLLM V1's AsyncLLM event-loop pinning, which an in-process
engine would otherwise hit.

The Cerebrium original streams OpenAI-format SSE chunks; for a clean cold-start
measurement this port returns the full generated text from a single method call.

Deploy:  modal deploy app.py
Test:    modal run app.py
"""

import subprocess
import time

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    # Pin vLLM to match the Cerebrium variant; "latest" (0.24.0) crashes at init.
    .pip_install("vllm==0.11.2")
    .env({"VLLM_SERVER_DEV_MODE": "1", "TORCH_CPP_LOG_LEVEL": "FATAL"})
)

app = modal.App("snapshot-bench-vllm", image=image)

with image.imports():
    import requests


def _check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def _wait_ready(process: subprocess.Popen, timeout: int = 5 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            _check_running(process)
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(2)
    raise TimeoutError(f"vLLM server not ready within {timeout} seconds")


def _warmup():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 8,
    }
    for _ in range(2):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=30
        ).raise_for_status()


def _sleep(level: int = 1):
    requests.post(f"http://127.0.0.1:{PORT}/sleep?level={level}").raise_for_status()


def _wake_up():
    requests.post(f"http://127.0.0.1:{PORT}/wake_up").raise_for_status()


@app.cls(
    gpu="A10G",
    memory=16384,
    scaledown_window=2,
    timeout=15 * MINUTES,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class Model:
    @modal.enter(snap=True)
    def startup(self):
        """Start vLLM, wait until healthy, warm it up, then sleep so the snapshot
        captures a clean CPU-only state. sleep(1) offloads weights to host RAM and
        frees the GPU — without it, live GPU allocations break the snapshot
        ("Failed creating Function memory snapshot")."""
        cmd = [
            "vllm",
            "serve",
            MODEL,
            "--served-model-name",
            MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.9",
            "--enable-sleep-mode",
            "--uvicorn-log-level",
            "error",
            "--disable-uvicorn-access-log",
        ]
        self.process = subprocess.Popen(cmd)
        _wait_ready(self.process)
        _warmup()
        _sleep(1)

    @modal.enter(snap=False)
    def restore(self):
        """Reload weights onto the GPU after restoring from the snapshot."""
        _wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()

    @modal.method()
    def infer(
        self,
        messages: list | None = None,
        run_id: str = "bench",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 256,
    ):
        if messages is None:
            messages = [
                {
                    "role": "user",
                    "content": "Give me a short introduction to large language models.",
                }
            ]
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=120
        )
        resp.raise_for_status()
        return {"text": resp.json()["choices"][0]["message"]["content"]}


@app.local_entrypoint()
def test():
    t0 = time.time()
    res = Model().infer.remote(
        messages=[
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        max_tokens=128,
    )
    dt = time.time() - t0
    print(f"\n=== Modal inference result (took {dt:.1f}s incl. cold start) ===")
    print(res["text"])
