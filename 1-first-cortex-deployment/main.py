from typing import Optional

from loguru import logger
from pydantic import BaseModel


class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    your_optional_param: Optional[str] = None  # an example optional parameter


# Set up your model here.
"""
This code will be run once when your replica starts. 
Load your model, setup datasets, pull in secrets, etc. here
"""


def predict(prompt, your_optional_param):
    params = Item(prompt=prompt, your_optional_param=your_optional_param)
    # This code is run on every inference request.

    # Access the parameters from your inference request
    prompt = params.prompt
    logger.info(f"Received a prompt of: `{prompt}`")

    if params.your_optional_param is not None:
        logger.info(f"You sent an optional param of: {params.your_optional_param}")

    # ADD YOUR CODE HERE

    return {
        "your_prompt": params.prompt,
        "your_other_return": "success",
    }  # return your results


"""
To deploy your model, run:
cerebrium deploy
"""
