import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI


async def periodic_hang():
    """Background task that hangs for 30 seconds every 30 seconds"""
    while True:
        await asyncio.sleep(30)
        print("Starting 30-second hang...")
        start = time.time()
        await asyncio.sleep(30)
        print(f"Hang completed after {time.time() - start:.2f} seconds")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the background task
    task = asyncio.create_task(periodic_hang())
    yield
    # Cancel the task on shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


@app.get("/hello")
def hello():
    return {"message": "Hello Cerebrium!"}


@app.post("/predict")
def predict():
    # Simulate processing time
    import time
    for i in range(10):
        print(f"Sleeping for {i * 1} seconds...")
        time.sleep(1)

    return {
        "your_other_return": "success",
    }


# return healthy 50% of the time
@app.get("/health")
def health():
    return "OK"


# return ready 50% of the time
@app.get("/ready")
def ready():
    return "OK"
