from cerebrium import get_secret
from huggingface_hub import login
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Your huggingface token (HF_AUTH_TOKEN) should be stored in your project secrets on your dashboard
login(token=get_secret("HF_AUTH_TOKEN"))

# Initialize the model
llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    dtype="bfloat16",
    max_model_len=20000,
    gpu_memory_utilization=0.9,
)


# Set up pydantic model
class Item(BaseModel):
    prompt: str
    temperature: float
    top_p: float
    top_k: float
    max_tokens: int
    frequency_penalty: float


def predict(
    prompt, temperature=0.8, top_p=0.75, top_k=40, max_tokens=256, frequency_penalty=1
):
    item = Item(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )

    sampling_params = SamplingParams(
        temperature=item.temperature,
        top_p=item.top_p,
        top_k=item.top_k,
        max_tokens=item.max_tokens,
        frequency_penalty=item.frequency_penalty,
    )

    outputs = llm.generate([item.prompt], sampling_params)

    generated_text = []
    for output in outputs:
        generated_text.append(output.outputs[0].text)

    return {"result": generated_text}
