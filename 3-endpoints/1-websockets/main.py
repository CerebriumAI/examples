import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Configure JSON logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    counter = 0
    await websocket.accept()

    # Extract x-request-id from headers
    x_request_id = websocket.headers.get("x-request-id", "unknown")

    async def ping_loop():
        """Periodically sends ping messages."""
        try:
            while True:
                await asyncio.sleep(5)  # Send ping every 5 seconds
                await websocket.send_bytes(b'')  # Empty ping frame
                logger.info("Sent ping")
        except Exception as e:
            logger.error(f"Ping failed: {e}")

    # Start the ping loop
    ping_task = asyncio.create_task(ping_loop())

    try:
        while True:
            message = f"Message {counter}"
            log_entry = json.dumps({"runId": x_request_id, "message": message})
            logger.info(log_entry)

            await websocket.send_text(message)
            await asyncio.sleep(3)
            counter += 1
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        ping_task.cancel()  # Stop the ping loop
        await websocket.close()
        logger.info("WebSocket connection properly closed")