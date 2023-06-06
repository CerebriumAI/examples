from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Item(BaseModel):
    prompt: str


model_id = "PygmalionAI/pygmalion-6b"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    low_cpu_mem_usage=True,
    load_in_8bit=True,
    torch_dtype=torch.float16,
)


def predict(item, run_id, logger):
    item = Item(**item)

    # Encoding input tokens
    input_tokens = tokenizer.encode(item.prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        input_tokens,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=1,
    )

    # Decode output tokens
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return {"result": output_text}
