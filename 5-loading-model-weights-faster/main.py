import time
from typing import Optional
from pydantic import BaseModel
from huggingface_hub import hf_api
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from fast_load import fast_load
import os
import sys
os.environ["HUGGINGFACE_HUB_ENABLE_HF_TRANSFER"] = "1"

print(("\n\n"),  file=sys. stderr)
print(("#" * 40),  file=sys. stderr)
print(("Starting main.py"),  file=sys. stderr)
print(("#" * 40),  file=sys. stderr)
print(("\n\n"),  file=sys. stderr)

class Item(BaseModel):
    # Add your input parameters here
    prompt: str

# model_path = "EleutherAI/gpt-neox-20b"
model_path = "EleutherAI/pythia-14m"



#########################################################
# Start of the fast_load code
#########################################################
def setup_model():
    # Load your model here
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model

start = time.time()
# What the user would call.
model = fast_load(
    model_id=model_path, load_weights_func=setup_model,faster=True
)
# print out the timing
print((f"Tensoriser loaded model in: {time.time() - start} seconds"),  file=sys. stderr)
#########################################################
# The rest of the inference code
#########################################################


tokenizer = AutoTokenizer.from_pretrained(model_path)

class _SentinelTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, sentinel_token_ids: torch.LongTensor, starting_idx: int):
        StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx :]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                0, self.sentinel_token_ids.shape[-1], 1
            ):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False


def get_stopping_criteria_list(words: list, tokens, device):
    stopping_criteria_list = StoppingCriteriaList(
        [
            _SentinelTokenStoppingCriteria(
                sentinel_token_ids=tokenizer(
                    word,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.to(device=device),
                starting_idx=tokens.input_ids.shape[-1],
            )
            for word in words
        ]
    )

    return stopping_criteria_list


########################################
# Prediction
########################################
def predict(item, run_id, logger):
    params = Item(**item)
    prompt = params.prompt
    max_new_tokens = params.max_new_tokens
    temperature = params.temperature
    top_p = params.top_p
    typical_p = params.typical_p
    repetition_penalty = params.repetition_penalty
    top_k = params.top_k
    stopping_criteria = params.stopping_criteria
    pad_token_id = params.pad_token_id

    if stopping_criteria:
        stopping_criteria = get_stopping_criteria_list(
            stopping_criteria, tokenizer(prompt, return_tensors="pt"), model.device
        )

    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        input_tokens,
        max_length=100,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=1,
    )

    output = model.generate(
        input_tokens,
        do_sample=True,
        temperature=temperature,
        stopping_criteria=stopping_criteria,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        typical_p=typical_p,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        pad_token_id=pad_token_id,
    )

    result = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return result
