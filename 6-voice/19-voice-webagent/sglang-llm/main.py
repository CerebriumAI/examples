from __future__ import annotations

import os
import shlex
import subprocess
import sys

from autoinference_utils.endpoint import wait_ready, warmup_chat_completions

PORT = 8000
MODEL = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
DRAFT_MODEL = "z-lab/Qwen3.6-35B-A3B-DFlash"
STARTUP_TIMEOUT = 60 * 60

EXTRA_SERVER_ARGS = {
    "--revision": MODEL_REVISION,
    "--speculative-algorithm": "DFLASH",
    "--speculative-num-draft-tokens": "16",
    "--attention-backend": "trtllm_mha",
    "--speculative-draft-attention-backend": "fa4",
    "--mem-fraction-static": "0.75",
    "--mamba-scheduler-strategy": "extra_buffer",
    "--mamba-ssm-dtype": "float32",
    "--reasoning-parser": "qwen3",
    "--tool-call-parser": "qwen3_coder",
    "--trust-remote-code": "",
}

# DFLASH does not support grammar/json_schema constrained decoding yet.
WARMUP_PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Reply with one sentence about Tokyo."}],
    "max_tokens": 64,
    "temperature": 0,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}


def _find_sglang_python() -> str:
    for candidate in (
        os.environ.get("SGLANG_PYTHON"),
        "/usr/local/bin/python3",
        "/usr/bin/python3",
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return sys.executable


def _build_cmd(python: str) -> list[str]:
    cmd = [
        python,
        "-m",
        "sglang.launch_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--model-path",
        MODEL,
        "--served-model-name",
        MODEL,
        "--speculative-draft-model-path",
        DRAFT_MODEL,
        "--tp-size",
        "1",
    ]
    for key, value in EXTRA_SERVER_ARGS.items():
        if value == "":
            cmd.append(key)
        else:
            cmd.extend([key, value])
    return cmd


if __name__ == "__main__":
    env = os.environ.copy()
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    env.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    env.setdefault("SGLANG_CUDA_COREDUMP_BEFORE_CRASH", "0")
    env.setdefault("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "1")
    env.setdefault("SGLANG_PYSPY_DUMP_BEFORE_CRASH", "0")

    python = _find_sglang_python()
    cmd = _build_cmd(python)
    print(f"[sglang-llm] starting: {shlex.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, env=env)
    wait_ready(
        proc,
        port=PORT,
        timeout=STARTUP_TIMEOUT,
        poll_interval=5.0,
        request_timeout=5.0,
    )
    warmup_chat_completions(
        port=PORT,
        payload=WARMUP_PAYLOAD,
        successful_requests=2,
        request_timeout=60.0,
    )
    print(f"{MODEL} (1xB200) sglang deployment is ready.", flush=True)
    proc.wait()
