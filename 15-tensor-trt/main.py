
from typing import Optional
from pydantic import BaseModel
import huggingface_hub
from huggingface_hub import snapshot_download

from cerebrium import get_secret
import os
import torch
import subprocess

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer
import time

huggingface_hub.login(token=get_secret("HF_AUTH_TOKEN"))
MAX_INPUT_LEN, MAX_OUTPUT_LEN = 256, 256
MAX_BATCH_SIZE = (
    10 
)
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_DIR="/persistent-storage/model_input"
ENGINE_DIR="/persistent-storage/model_output"

class Item(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.95
    top_k: Optional[int] = 100
    top_p: Optional[float] = 1.0
    repetition_penalty: Optional[float] = 1.05
    num_tokens: Optional[int] = 250
    prompt_template: Optional[str] = "<start_of_turn>user\n{input_text}<end_of_turn>\n<start_of_turn>model\n"


if not os.path.exists(ENGINE_DIR):
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
    )

    # Build the TRT engine
    convert_checkpoint = f"""
    python convert_checkpoint.py --model_dir {MODEL_DIR} \
                                  --output_dir ./model_ckpt \
                                  --dtype float16
    """

    SIZE_ARGS = f"--max_batch_size={MAX_BATCH_SIZE} --max_input_len={MAX_INPUT_LEN} --max_output_len={MAX_OUTPUT_LEN}"
    build_engine = f"""
    trtllm-build --checkpoint_dir ./model_ckpt --output_dir {ENGINE_DIR} \
                --tp_size=1 --workers=1 \
                --max_batch_size={MAX_BATCH_SIZE} --max_input_len={MAX_INPUT_LEN} --max_output_len={MAX_OUTPUT_LEN} \
                --gemm_plugin=float16 --gpt_attention_plugin=float16
    """

    print("Building engine...")
    subprocess.run(convert_checkpoint, shell=True, check=True)
    subprocess.run(build_engine, shell=True, check=True)
    print("\nEngine built successfully! You can find it at: ", ENGINE_DIR)
else:
    print("Engine already exists at: ", ENGINE_DIR)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# LLaMA models do not have a padding token, so we use the EOS token
tokenizer.add_special_tokens(
    {"pad_token": tokenizer.eos_token}
)
# and then we add it from the left, to minimize impact on the output
tokenizer.padding_side = "left"
pad_id = tokenizer.pad_token_id
end_id = tokenizer.eos_token_id

runner_kwargs = dict(
    engine_dir=f"{ENGINE_DIR}",
    lora_dir=None,
    rank=tensorrt_llm.mpi_rank(),
)

model = ModelRunner.from_dir(**runner_kwargs)


def parse_input(
        tokenizer,
        input_text,
        prompt_template=None,
        add_special_tokens=True,
        max_input_length=923
    ):

    # Apply prompt template if provided
    if prompt_template is not None:
        input_text = prompt_template.format(input_text=input_text)

    # Encode the text to input IDs
    input_ids = tokenizer.encode(
        input_text,
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=max_input_length,
    )

    # Convert to tensor
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.int32)  # Add batch dimension

    return input_ids_tensor

def predict(item, run_id, logger):
    item = Item(**item)
    
    stop_words_list = None
    bad_words_list = None

    batch_input_ids = parse_input(
        tokenizer=tokenizer,
        input_text=item.prompt,
        prompt_template=item.prompt_template
    )
    input_length = batch_input_ids[0].size(0)

    time_begin = time.time()
    with torch.no_grad():
        outputs = model.generate(
            batch_input_ids,
            max_new_tokens=item.num_tokens,
            max_attention_window_size=None,
            sink_token_length=None,
            end_id=end_id,
            pad_id=pad_id,
            temperature=item.temperature,
            top_k=item.top_k,
            top_p=item.top_p,
            num_beams=1,
            repetition_penalty=item.repetition_penalty,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            output_sequence_lengths=True,
            return_dict=True,
        )
        torch.cuda.synchronize()

    time_total = time.time() - time_begin

    output_ids = outputs["output_ids"]
    sequence_lengths = outputs["sequence_lengths"]

    # Decode the output
    output_begin = input_length
    output_end = sequence_lengths
    output_text = tokenizer.decode(output_ids[0][0][output_begin:output_end].tolist())

    return {
        "response_txt": output_text,
        "latency_s": time_total,
    }