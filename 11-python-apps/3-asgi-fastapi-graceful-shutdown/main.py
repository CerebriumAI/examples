from fastapi import FastAPI, Body
from loguru import logger
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager

app = FastAPI()
shutdown_event = asyncio.Event()
active_requests = 0
active_requests_lock = asyncio.Lock()

class Item(BaseModel):
    # Add your input parameters here
    prompt: str


@app.post("/predict")
async def predict(item: Item = Body(...)):
    # Access the parameters from your inference request
    prompt = item.prompt
    logger.info(f"Received a prompt of: `{prompt}`")
    # Simulate long-running task
    await asyncio.sleep(120)
    return {
        "your_prompt": prompt,
        "your_other_return": "success",
    }  # return your results


#  health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}

@asynccontextmanager
async def track_active_requests():
    global active_requests
    async with active_requests_lock:
        active_requests += 1
    try:
        yield
    finally:
        async with active_requests_lock:
            active_requests -= 1


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutdown initiated. Waiting for ongoing requests to complete...")
    # Wait for active requests to finish
    while True:
        async with active_requests_lock:
            if active_requests == 0:
                break
        logger.info(f"Waiting for {active_requests} active requests to complete...")
        await asyncio.sleep(0.5)

    logger.info("All requests completed. Proceeding with shutdown.")
    shutdown_event.set()

"""
To deploy your model, run:
cerebrium deploy
"""
