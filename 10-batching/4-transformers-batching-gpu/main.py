import os
import threading
from typing import Callable

import torch
from huggingface_hub import login
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Log into Hugging Face Hub
login(token=os.environ.get("HF_AUTH_TOKEN"))

# Initialize the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically uses available GPUs
)

# Initialize the text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    # Set default values for parameters that might cause warnings
    do_sample=True,  # Enable sampling
)


# Define the data model for requests
class Item(BaseModel):
    prompt: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    frequency_penalty: float


class BatchProcessor:
    def __init__(self, max_batch_size: int, max_wait_time: float):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.lock = threading.Lock()
        self.current_batch = []
        self.timer = None

    def add_request(
        self, item: Item, callback: Callable[[str], None], result_event: threading.Event
    ):
        with self.lock:
            # Append the request with its callback and event
            self.current_batch.append((item, callback, result_event))

            # If batch size is reached, process immediately
            if len(self.current_batch) >= self.max_batch_size:
                self.process_batch()
            # Otherwise, start the timer if it's the first item in the batch
            elif len(self.current_batch) == 1:
                self.start_timer()

    def start_timer(self):
        # Cancel any existing timer and start a new one
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.max_wait_time, self.process_batch)
        self.timer.start()

    def process_batch(self):
        with self.lock:
            # Cancel the timer if running
            if self.timer:
                self.timer.cancel()
                self.timer = None

            # Process batch if there are items
            if self.current_batch:
                try:
                    # Extract prompts for each item
                    prompts = [item.prompt for item, _, _ in self.current_batch]
                    first_item = self.current_batch[0][
                        0
                    ]  # Use first item for generation params

                    # Generate results using the generator
                    outputs = generator(
                        prompts,
                        max_new_tokens=first_item.max_tokens,
                        temperature=first_item.temperature,
                        top_p=first_item.top_p,
                        top_k=first_item.top_k,
                        num_return_sequences=1,
                        return_full_text=False,
                        repetition_penalty=first_item.frequency_penalty,
                        do_sample=True,  # Ensure sampling is enabled
                    )

                    # Extract generated text for each prompt
                    results = [output[0]["generated_text"] for output in outputs]

                    # Call each request's callback with its corresponding result
                    for (_, callback, result_event), result_text in zip(
                        self.current_batch, results
                    ):
                        callback(result_text)
                        result_event.set()  # Unblock the waiting `predict` call

                except Exception as e:
                    # Handle exceptions and set the event to unblock waiting threads
                    for _, callback, result_event in self.current_batch:
                        callback(f"Error: {str(e)}")
                        result_event.set()
                    # Optionally, log the exception
                    print(f"Exception during batch processing: {e}")

                finally:
                    # Clear the batch
                    self.current_batch.clear()


# Initialize the batch processor with desired size and wait time
batch_processor = BatchProcessor(max_batch_size=10, max_wait_time=30.0)


def predict(
    prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.75,
    top_k: int = 40,
    max_tokens: int = 256,
    frequency_penalty: float = 1.0,
) -> str:
    item = Item(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
    )

    result_event = threading.Event()
    response = {}

    def handle_response(result):
        response["result"] = result  # Store result in a shared dict for retrieval

    # Add request to batch processor and wait for the event to complete
    batch_processor.add_request(item, handle_response, result_event)
    result_event.wait()  # Wait for the batch to process and for the event to be set

    return response["result"]  # Return the result stored in handle_response
