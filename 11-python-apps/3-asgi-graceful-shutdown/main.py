from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import asyncio
import logging
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AppState:
    active_requests: int = 0
    shutting_down: bool = False
    shutdown_event: asyncio.Event = None
    lock: asyncio.Lock = None
    max_concurrency: int = 1  # Should match replica_concurrency in cerebrium.toml

    def __post_init__(self):
        if self.shutdown_event is None:
            self.shutdown_event = asyncio.Event()
        if self.lock is None:
            self.lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase
    logger.info("Application startup initiated")
    app.state.app_state = AppState()
    logger.info("Initializing resources...")
    # Add any startup tasks here
    logger.info("Application startup complete - ready to accept requests")

    yield  # Application is running

    # Shutdown phase
    logger.info("Shutdown signal received - beginning graceful shutdown")
    state = app.state.app_state
    state.shutting_down = True

    # Wait for active requests to complete with timeout
    if state.active_requests > 0:
        logger.info(f"Waiting for {state.active_requests} active request(s) to complete")
        try:
            # Wait up to 30 seconds for requests to complete
            await asyncio.wait_for(state.shutdown_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Shutdown timeout reached - {state.active_requests} request(s) still active")
        else:
            logger.info("All requests completed gracefully")

    logger.info("Shutdown complete")

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def track_requests(request: Request, call_next):
    state: AppState = request.app.state.app_state

    async with state.lock:
        # Check if we're shutting down
        if state.shutting_down:
            logger.warning(f"Rejecting new request to {request.url.path} - server is shutting down")
            raise HTTPException(status_code=503, detail="Service shutting down")

        # Increment active request counter
        state.active_requests += 1
        logger.debug(f"Request started: {request.method} {request.url.path} (active: {state.active_requests})")

    try:
        response = await call_next(request)
        logger.debug(f"Request completed: {request.method} {request.url.path} (status: {response.status_code})")
        return response
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path} - {str(e)}")
        raise
    finally:
        async with state.lock:
            # Decrement active request counter
            state.active_requests -= 1
            logger.debug(f"Request finished: {request.method} {request.url.path} (active: {state.active_requests})")

            # Signal shutdown event if all requests completed during shutdown
            if state.shutting_down and state.active_requests == 0:
                state.shutdown_event.set()

@app.get("/health")
async def health(request: Request):
    """Health check endpoint - returns healthy if app is running"""
    return {"status": "healthy"}

@app.get("/ready")
async def ready(request: Request):
    """
    Readiness check - returns ready only if not shutting down and has capacity.

    IMPORTANT: Without a proper /ready endpoint, Cerebrium uses TCP ping to check readiness,
    which only verifies the port is open. This can cause traffic to be routed to replicas
    that are shutting down or at max capacity. This endpoint ensures true application readiness.
    """
    state: AppState = request.app.state.app_state

    async with state.lock:
        if state.shutting_down:
            raise HTTPException(status_code=503, detail="Not ready - shutting down")

        # Check if we have capacity for new requests
        if state.active_requests >= state.max_concurrency:
            raise HTTPException(
                status_code=503,
                detail=f"Not ready - at max concurrency ({state.active_requests}/{state.max_concurrency})"
            )

    return {
        "status": "ready",
        "active_requests": state.active_requests,
        "max_concurrency": state.max_concurrency
    }

@app.post("/hello")
def hello():
    logger.info("Hello endpoint called")
    return {"message": "Hello Cerebrium!"}