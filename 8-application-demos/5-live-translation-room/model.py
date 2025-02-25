from pydantic import BaseModel
from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from transformers import AutoTokenizer
import time
import numpy as np
import librosa
from huggingface_hub import login
import os

login(token=os.environ.get("HF_TOKEN"))

class UltravoxModel:
    def __init__(self, model_name: str = "fixie-ai/ultravox-v0_4_1-llama-3_1-8b"):
        self.model_name = model_name
        self._initialize_engine()
        self._initialize_tokenizer()
        
    def _initialize_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
    def _initialize_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.messages = [{
            'role': 'user',
            'content': "<|audio|>\n"
        }]
        self.prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        self.stop_token_ids = None

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 12, audio: np.ndarray = None):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=self.stop_token_ids
        )
        
        mm_data = {
            "audio": librosa.load("./actual_speech.wav", sr=16000)[0]
        }
        inputs = {"prompt": prompt, "multi_modal_data": mm_data}
        results_generator = self.engine.generate(inputs, sampling_params, str(time.time()))
        
        async for output in results_generator:
            print(output.outputs)
            yield output.outputs[0]