import os
import threading
from typing import Callable

from huggingface_hub import login
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Log into Hugging Face Hub
login(token=os.environ.get("HF_AUTH_TOKEN"))

# Initialize the model
llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    dtype="bfloat16",
    max_model_len=20000,
    gpu_memory_utilization=0.9,
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
                # Extract prompts and sampling parameters for each item
                prompts = [item.prompt for item, _, _ in self.current_batch]
                first_item = self.current_batch[0][
                    0
                ]  # Use first item for sampling params

                sampling_params = SamplingParams(
                    temperature=first_item.temperature,
                    top_p=first_item.top_p,
                    top_k=first_item.top_k,
                    max_tokens=first_item.max_tokens,
                    frequency_penalty=first_item.frequency_penalty,
                )

                # Generate results using the llm
                outputs = llm.generate(prompts, sampling_params)
                results = [output.outputs[0].text for output in outputs]

                # Call each request's callback with its corresponding result
                for (_, callback, result_event), result_text in zip(
                    self.current_batch, results
                ):
                    callback(result_text)
                    result_event.set()  # Unblock the waiting `predict` call

                # Clear the batch
                self.current_batch.clear()


# Initialize the batch processor with desired size and wait time
batch_processor = BatchProcessor(max_batch_size=4, max_wait_time=20.0)


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
