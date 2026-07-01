"""Modal port of the Qwen2-VL-7B (SGLang) snapshot benchmark.

Follows Modal's official sglang_snapshot.py recipe: run SGLang as a
`launch_server` subprocess started with `--enable-memory-saver
--enable-weights-cpu-backup`, warm it up, then hit `/release_memory_occupation`
BEFORE the snapshot and `/resume_memory_occupation` AFTER restore.

Two things are essential and were missing from the earlier in-process attempt:
  * Subprocess server (not in-process sgl.Engine) — Modal's *memory* snapshot
    can't capture sgl.Engine's scheduler/detokenizer child processes, so it
    failed outright. The launch_server subprocess snapshots cleanly.
  * --enable-weights-cpu-backup — release copies the weights to CPU RAM (which
    the snapshot captures) instead of discarding them. Without it the restored
    weights are garbage.

Built on the nvidia/cuda base + pip sglang 0.5.5 (the lmsysorg runtime image
fails Modal's client bootstrap on an aiohttp source build).

Deploy:  modal deploy app.py
"""

import subprocess
import time

import modal

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
PORT = 8000
N_GPUS = 1
MAX_INPUTS = 1  # matches the Cerebrium variant's replica_concurrency = 1

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("libnuma-dev")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Compile inductor kernels synchronously — forked compile workers
            # inherit GPU FDs without a CUDA context and break the snapshot.
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
    .pip_install(
        "huggingface_hub",
        "hf_transfer",
        "requests",
        # Pin to the CUDA 12.8 generation: sglang 0.5.5 pins torch==2.8.0 (cu128)
        # + sgl-kernel 0.3.16 (cu128), all aligned with this base.
        "torch==2.8.0",
        "sglang[all]==0.5.5",
    )
)

app = modal.App("snapshot-bench-qwen-vl-sglang", image=image)

DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)


@app.cls(
    image=image,
    gpu="L40S",
    memory=61440,
    timeout=900,
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=MAX_INPUTS)
class Model:
    @modal.enter(snap=True)
    def startup(self):
        """Launch the sglang server, warm it up, then release GPU memory (with
        weights backed up to CPU) so the snapshot can be taken."""
        import requests

        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--tp",
            str(N_GPUS),
            "--mem-fraction-static",
            "0.8",
            "--cuda-graph-max-bs",
            str(MAX_INPUTS),
            "--max-running-requests",
            str(MAX_INPUTS),
            "--enable-memory-saver",
            "--enable-weights-cpu-backup",
        ]
        self.proc = subprocess.Popen(cmd, start_new_session=True)

        # Block until the server is healthy (model download + load can be slow).
        for _ in range(900):
            try:
                requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("sglang server failed to become healthy")

        # Warm up the full vision+decode path before the snapshot.
        self._infer(DEFAULT_IMAGE_URL, "Warmup.", max_tokens=2)
        print("Warmup generation OK", flush=True)

        print("releasing memory occupation before snapshot", flush=True)
        requests.post(
            f"http://127.0.0.1:{PORT}/release_memory_occupation", json={}
        ).raise_for_status()
        print("released memory occupation", flush=True)

    @modal.enter(snap=False)
    def wake_up(self):
        """Reload the weights to the GPU after restoring from the snapshot."""
        import requests

        print("resuming memory occupation after restore", flush=True)
        requests.post(
            f"http://127.0.0.1:{PORT}/resume_memory_occupation", json={}
        ).raise_for_status()
        print("resumed memory occupation", flush=True)

    def _infer(self, image_url, prompt, max_tokens, image_base64=None):
        import requests

        url = (
            f"data:image/jpeg;base64,{image_base64}" if image_base64 else image_url
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        resp = requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @modal.method()
    def infer(
        self,
        image_url: str = DEFAULT_IMAGE_URL,
        image_base64: str | None = None,
        prompt: str = "Describe this image in detail.",
        max_tokens: int = 256,
    ):
        return {"text": self._infer(image_url, prompt, max_tokens, image_base64)}
