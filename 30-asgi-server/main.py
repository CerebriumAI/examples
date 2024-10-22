from fastapi import FastAPI, Body
from loguru import logger
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    # Add your input parameters here
    prompt: str


@app.post("/predict")
def predict(item: Item = Body(...)):
    # Access the parameters from your inference request
    prompt = item.prompt
    logger.info(f"Received a prompt of: `{prompt}`")

    return {
        "your_prompt": prompt,
        "your_other_return": "success",
    }  # return your results


#  health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


"""
To deploy your model, run:
cerebrium deploy
"""
