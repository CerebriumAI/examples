from fastapi import FastAPI, WebSocket
from loguru import logger

app = FastAPI()


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
