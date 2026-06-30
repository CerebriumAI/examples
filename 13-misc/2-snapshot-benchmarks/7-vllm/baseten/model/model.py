"""Baseten (Truss) variant of the vLLM snapshot benchmark.

No GPU snapshot on Baseten, so no checkpoint trigger. The vLLM engine is built
in `load()` (full weight load + KV cache alloc + CUDA warmup on every cold
start). Uses the synchronous `LLM` engine since Truss `predict` is
request/response; the chat template is applied via the tokenizer.
"""

from vllm import LLM, SamplingParams

# model_cache volume_folder mount path.
CACHE_DIR = "/app/model_cache/qwen25-05b"


class Model:
    def __init__(self, **kwargs):
        self._data_resolver = kwargs.get("lazy_data_resolver")
        self._llm = None

    def load(self):
        if self._data_resolver:
            self._data_resolver.block_until_download_complete()
        self._llm = LLM(model=CACHE_DIR, gpu_memory_utilization=0.9, max_model_len=8192)

    def predict(self, request: dict):
        messages = request.get("messages") or [
            {"role": "user", "content": "Hi"}
        ]
        sampling_params = SamplingParams(
            temperature=request.get("temperature", 0.8),
            top_p=request.get("top_p", 0.95),
            max_tokens=request.get("max_tokens", 512),
        )
        outputs = self._llm.chat(messages, sampling_params)
        return {"text": outputs[0].outputs[0].text}
