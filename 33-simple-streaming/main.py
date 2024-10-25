import time

from loguru import logger
from pydantic import BaseModel


class Item(BaseModel):
    prompt: str


def predict(prompt):
    params = Item(prompt=prompt)

    prompt = params.prompt
    logger.info(f"Received a prompt: `{prompt}`")

    # Example: Yield multiple responses over time
    for i in range(5):
        logger.info(f"Iteration {i}")
        # Simulate some processing or generate partial results
        response = "your_prompt " + params.prompt + "iteration " + str(i)

        logger.info("Sending response")
        yield response  # Use yield to send the response incrementally
        logger.info("Sleeping for 5s")
        time.sleep(5)  # Simulate a delay between responses

    logger.info("Complete")
    # Optionally, yield a final response
    yield "complete"
