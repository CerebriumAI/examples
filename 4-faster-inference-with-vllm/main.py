from pydantic import BaseModel
from typing import Optional

import torch
from pydantic import BaseModel

# From vLLM, you need to import the following
from vllm import LLM, SamplingParams

# then simply initialise your vLLM model as follows:
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1", dtype="bfloat16")
"""
- vLLM will handle the setup of the model as well as the tokenization/de-tokenization automatically
- Some things to note:
    -  Please check to see if your model is supported on vLLM.
    -  we only support single gpu vllm models at this time.
"""



class Item(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.75
    top_k: Optional[float] = 40
    max_tokens: Optional[int] = 256
    frequency_penalty: Optional[float] = 1


def predict(item, run_id, logger):
    item = Item(**item)

    # Now jusst setup your sampling parameters for inference:
    sampling_params = SamplingParams(temperature=item.temperature, top_p=item.top_p, top_k=item.top_k, max_tokens=item.max_token, frequency_penalty=item.frequency_penalty)
    
    # And feed your prompt and sampling params into your LLM pipeline as follows. 
    outputs = llm.generate([item.prompt], sampling_params)

    # Extract your text outputs:
    generated_text = []
    for output in outputs:
        generated_text.append(output.outputs[0].text)

    # And return the result
    return {"result": generated_text}
