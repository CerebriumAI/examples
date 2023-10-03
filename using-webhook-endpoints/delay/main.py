from typing import Optional
from pydantic import BaseModel
import time

class Item(BaseModel):
    # Add your input parameters here
    delayMins: Optional[int] = 3

### Setup your model here. 
"""
This code will be run once when your replica starts. 
Load your model, setup datasets, pull in secrets, etc. here
"""


def predict(item, run_id, logger):
    params = Item(**item)

    for i in range(params.delayMins*10):
        logger.info(f"Waited for: {i*10} seconds...") # Some prints for your run logs
        time.sleep(10)

    return {"result":f"Delayed for {params.delayMins} mins"} # return your results 

    """
    To deploy your model, run:
    cerebrium deploy <<YOUR DEPLOYMEN TNAME>> --config-file config.yaml
    """