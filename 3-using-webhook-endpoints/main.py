import time
from typing import Optional

from loguru import logger
from pydantic import BaseModel


class Item(BaseModel):
    # Add your input parameters here
    delayMins: Optional[int] = 3


# Set up your model here.
"""
This code will be run once when your replica starts. 
Load your model, setup datasets, pull in secrets, etc. here
"""


def predict(item):
    params = Item(**item)

    for i in range(params.delayMins * 10):
        logger.info(f"Waited for: {i * 10} seconds...")  # Some prints for your run logs
        time.sleep(10)

    return {"result": f"Delayed for {params.delayMins} mins"}  # return your results


"""
To deploy your model, run:
cerebrium deploy <<YOUR DEPLOYMENT NAME>> --config-file config.yaml
"""
