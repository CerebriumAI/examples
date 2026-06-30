"""Modal port of the vLLM (LLM serving) snapshot benchmark.

Mirror of ../main.py using Modal's `enable_memory_snapshot` + experimental
`enable_gpu_snapshot`. The vLLM async engine is constructed inside
`@modal.enter(snap=True)` so the loaded weights / CUDA state are captured in the
snapshot.

The Cerebrium original streams OpenAI-format SSE chunks; for a clean cold-start
measurement this port returns the full generated text from a single method call.

Deploy:  modal deploy app.py
"""

import modal

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install("vllm", "pydantic")
)

app = modal.App("snapshot-bench-vllm", image=image)


@app.cls(
    gpu="A10G",
    memory=16384,
    scaledown_window=2,
    timeout=900,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class Model:
    @modal.enter(snap=True)
    def load(self):
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def infer(
        self,
        messages: list | None = None,
        run_id: str = "bench",
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_tokens: int = 256,
    ):
        from vllm import SamplingParams

        if messages is None:
            messages = [
                {"role": "user", "content": "Give me a short introduction to large language models."}
            ]
        prompt = " ".join(f"{m['role']}: {m['content']}" for m in messages)

        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        results_generator = self.engine.generate(prompt, sampling_params, run_id)

        text = ""
        async for output in results_generator:
            text = output.outputs[0].text

        return {"text": text}
