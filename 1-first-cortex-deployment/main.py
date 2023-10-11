from typing import Optional
from pydantic import BaseModel
import numpy as np # import a library you've entered in your  requirements.txt

class Item(BaseModel):
    # Add your input parameters here
    prompt: str
    your_optional_param: Optional[str] = None # an example optional parameter

### Setup your model here. 
"""
This code will be run once when your replica starts. 
Load your model, setup datasets, pull in secrets, etc. here
"""


def predict(item, run_id, logger):
    params = Item(**item)
    # This code is run on every inference inference request.

    # Access the parameters from your inference request
    prompt = params.prompt
    logger.info(f"Received a prompt of: `{prompt}`")
    
    if params.your_optional_param is not None:
        logger.info(f"You sent an optional param of: {params.your_optional_param}")
    
    
    ### ADD YOUR CODE HERE

    return {"your_prompt": params.prompt, "your_other_return": "success"} # return your results 

    """
    To deploy your model, run:
    cerebrium deploy <<YOUR DEPLOYMEN TNAME>> --config-file config.yaml
    """
