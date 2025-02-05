import os
import torch
from transformers import AutoTokenizer
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.config import NeuronConfig, QuantizationConfig, GenerationConfig
from transformers_neuronx.config import GenerationConfig 
from transformers_neuronx import GQA
from transformers import LlamaForCausalLM, TextIteratorStreamer
from huggingface_hub import login
from threading import Thread
import time
from pydantic import BaseModel
from typing import List, Optional
from threading import Thread
import json
from fastapi import FastAPI

# app = FastAPI()

# @app.get("/health")
# def health():
#     return "Ok"

class Message(BaseModel):
    role: str
    content: str

def format_chat_prompt(messages: list) -> str:
    formatted_messages = []
    for msg in messages:
        msg_obj = Message(**msg)
        formatted_messages.append(
            f"<|start_header_id|>{msg_obj.role}<|end_header_id|>\n{msg_obj.content}<|eot_id|>"
        )
    return "<|begin_of_text|>" + "".join(formatted_messages) + "<|start_header_id|>assistant<|end_header_id|>"


login(token=os.environ.get("HF_TOKEN"))

model_name = 'meta-llama/Llama-3.1-70B-Instruct'
# model = LlamaForCausalLM.from_pretrained(model_name)

# from transformers_neuronx.module import save_pretrained_split

## Comment out this in your second run
# if not os.path.exists('/persistent-storage/Llama-3-70b-split'):
#     print('Downloading model')
#     save_pretrained_split(model, '/persistent-storage/Llama-3-70b-split')
 
neuron_config = NeuronConfig(
    attention_layout='BSH',
    fuse_qkv=True,
    group_query_attention=GQA.REPLICATED_HEADS,
    sequence_parallel_norm=True,
    quant=QuantizationConfig(quant_dtype='s8', dequant_dtype='bf16'),
)

batch_size = 1
# neuron_model = LlamaForSampling.from_pretrained(
#     'meta-llama/Llama-3.1-70B-Instruct',
#     batch_size=1,
#     tp_degree=64, amp='bf16',
#     n_positions=4092,
#     neuron_config = neuron_config,
#     context_length_estimate=[4092]
# )


# if not os.path.exists('/persistent-storage/neuron_artifacts'):
#     print('Converting model to neuron')
#     neuron_model.to_neuron()
#     neuron_model.save('/persistent-storage/neuron_artifacts')
# else:
#     neuron_model.load('/persistent-storage/neuron_artifacts')
#     neuron_model.to_neuron()

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

neuron_model = None

# @app.post("/run/chat/completions")
async def run(
    messages: list,
    model: str = "meta-llama/Llama-3.1-70B-Instruct",
    run_id: str = None,
    stream: bool = True,
    temperature: float = 0.8,
    max_tokens: int = 128,
):
    prompt = format_chat_prompt(messages)
    
    global neuron_model  # Declare we're using the global llm variable
    if neuron_model is None:
        print("Initializing LLM...") 
        model = LlamaForCausalLM.from_pretrained(model_name)
        neuron_model = LlamaForSampling.from_pretrained(
            'meta-llama/Llama-3.1-70B-Instruct',
            batch_size=1,
            tp_degree=64, amp='bf16',
            n_positions=4096,
            neuron_config = neuron_config,
            context_length_estimate=[4096]
        )


        if not os.path.exists('/persistent-storage/neuron_artifacts'):
            print('Converting model to neuron')
            neuron_model.to_neuron()
            neuron_model.save('/persistent-storage/neuron_artifacts')
        else:
            neuron_model.load('/persistent-storage/neuron_artifacts')
            neuron_model.to_neuron()
        print("Done initializing LLM...") 

    with torch.inference_mode():
        start = time.time()
        input_ids = torch.as_tensor([tokenizer.encode(prompt)])
        

        streamer = TextIteratorStreamer(tokenizer)
        generation_kwargs = {
            "input_ids": input_ids,
            "sequence_length": max_tokens,
            "temperature": temperature,
            "streamer": streamer
        }
        
        thread = Thread(target=neuron_model.sample, kwargs=generation_kwargs)
        thread.start()
        
        previous_text = ""
        first_chunk = True
        
        for new_text in streamer:
            chunk = {
                "id": run_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None,
                    }
                ],
            }
            
            if first_chunk:
                chunk["choices"][0]["delta"]["role"] = "assistant"
                first_chunk = False
            
            if new_text:
                chunk["choices"][0]["delta"]["content"] = new_text
            
            yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"