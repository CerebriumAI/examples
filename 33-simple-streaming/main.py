import time

from loguru import logger
from pydantic import BaseModel


class Item(BaseModel):
    prompt: str


def predict(prompt):
    params = Item(prompt=prompt)

    prompt = params.prompt
    logger.info(f"Received a prompt: `{prompt}`")

    # Initial message
    yield 'data: {"message":"hello"}\n\n'
    time.sleep(2)

    # Second message
    yield 'data: {"message":"1"}\n\n'
    time.sleep(2)

    # Third message
    yield 'data: {"message":"2"}\n\n'

    logger.info("Complete")
    # Optionally, yield a final response
    yield "complete"
