from vllm import EngineArgs, AsyncLLMEngine, RequestOutput, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import os
import sounddevice as sd
import numpy as np
import time
from cartesia.tts import CartesiaTTS
import re
import time

voice = 'Barbershop Man'

api_key = "fc560914-1303-456a-9fde-ed95f4561604"
gen_cfg = dict(model_id="upbeat-moon", data_rtype='array', output_format='fp32')

client = CartesiaTTS(api_key=api_key)
voices = client.get_voices()
voice_id = voices[voice]["id"]
voice = client.get_voice_embedding(voice_id=voice_id)

def initialize_engine(model_id: str, quantization: str) -> AsyncLLMEngine:
    """Initialize the LLMEngine with the given model ID."""
    engine_args = AsyncEngineArgs(model=model_id, quantization=quantization)
    return AsyncLLMEngine.from_engine_args(engine_args)

engine = initialize_engine("casperhansen/llama-3-8b-instruct-awq", "AWQ")

async def run(prompt: str, run_id):  # run_id is optional, injected by Cerebrium at runtime
  global start_time
  start_time = time.time()
  sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
  results_generator = engine.generate(prompt, sampling_params, run_id)
  buffer = ""
  async for output in results_generator:
    buffer += output.outputs[0].text
    sentences = re.split(r'(?<=[.!?]) +', buffer)
    for sentence in sentences[:-1]:
        on_sentence_end(sentence)
    buffer = sentences[-1]
    asyncio.sleep(5)
    yield buffer

def on_sentence_end(sentence: str):
    """Function to be triggered after every sentence."""
    global start_time
    output = client.generate(transcript=sentence, voice=voice, stream=False, **gen_cfg)
    elapsed_time = (time.time() - start_time) * 1000  # Calculate elapsed time in milliseconds
    print(f"Sentence ended: {output}")
    print(f"Time elapsed: {elapsed_time:.2f} ms")

if __name__ == "__main__":
    prompt = "Please provide a prompt for the LLM engine."
    run_id = None  # Replace with actual run_id if available
    async def main():
        async for output in run(prompt, run_id):
            print(output)
    asyncio.run(main())

# # play audio
# buffer = output["audio"]
# rate = output["sampling_rate"]
# sd.play(buffer, rate, blocking=True)