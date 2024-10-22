from fastapi import FastAPI, WebSocket
from loguru import logger
from pydantic import BaseModel

app = FastAPI()


class Item(BaseModel):
    # Add your input parameters here
    prompt: str


@app.post("/predict")
def predict(prompt):
    params = Item(prompt=prompt)
    # This code is run on every inference request.

    # Access the parameters from your inference request
    prompt = params.prompt
    logger.info(f"Received a prompt of: `{prompt}`")

    return {
        "your_prompt": params.prompt,
        "your_other_return": "success",
    }  # return your results


#  health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        logger.info(f"Received data: {data}")
        await websocket.send_text(f"Message text was: {data}")
    await websocket.close()


"""
To deploy your model, run:
cerebrium deploy
"""
