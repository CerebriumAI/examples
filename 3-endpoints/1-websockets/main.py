import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from anyio import Lock
from contextlib import asynccontextmanager

# Configure JSON logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

pendingRequestsLock = Lock()
pendingRequests = 0
timeout = 60  # seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Clean up the ML models and release the resources
    global pendingRequests, timeout
    logger.info("Shutting down the application")
    while pendingRequests > 0:
        logger.info(f"Waiting for {pendingRequests} pending requests to finish")
        await asyncio.sleep(1)
        timeout -= 1
        if timeout == 0:
            logger.error("Timeout reached. Forcing shutdown")
            break


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global pendingRequests, pendingRequestsLock
    async with pendingRequestsLock:
        pendingRequests += 1
    await websocket.accept()

    # Extract x-request-id from headers
    x_request_id = websocket.headers.get("x-request-id", "unknown")

    counter = 1
    try:
        while True:
            ping_message = await websocket.receive_text()
            if ping_message != "alive?":
                logger.error(f"Unexpected message: {ping_message}")
                break
            message = f"{x_request_id} is alive! Message {counter}"
            logger.info(message)
            counter += 1
            await websocket.send_text(message)
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        async with pendingRequestsLock:
            pendingRequests -= 1
        await websocket.close()
        logger.info("WebSocket connection closed")
