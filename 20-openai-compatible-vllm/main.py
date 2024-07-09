from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


def initialize_engine(model_id: str) -> AsyncLLMEngine:
    """Initialize the LLMEngine with the given model ID."""
    engine_args = AsyncEngineArgs(model=model_id)
    return AsyncLLMEngine.from_engine_args(engine_args)


engine = initialize_engine("NousResearch/Meta-Llama-3-8B-Instruct")


async def main(prompt: str, run_id: str):
    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, min_tokens=100, max_tokens=120
    )
    results_generator = engine.generate("What is your name?", sampling_params, run_id)
    previous_text = ""
    async for output in results_generator:
        prompt = output.outputs
        new_text = prompt[0].text[len(previous_text) :]
        print(new_text)
        yield (new_text)
        previous_text = prompt[0].text
