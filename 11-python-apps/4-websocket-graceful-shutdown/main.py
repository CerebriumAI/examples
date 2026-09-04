import asyncio
import contextlib
import os
import signal

import uvicorn
from fastapi import FastAPI, WebSocket
from uvicorn import Server

DRAIN_TIMEOUT_SECONDS = float(os.getenv("DRAIN_TIMEOUT_SECONDS", "600"))

active_handlers: set[asyncio.Task] = set()
drained = asyncio.Event()
drained.set()
shutting_down = False

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {
        "ready": not shutting_down,
        "active_connections": len(active_handlers),
    }


@app.websocket("/ws")
async def echo(websocket: WebSocket):
    await websocket.accept()
    if shutting_down:
        await websocket.close(code=1013, reason="Server shutting down")
        return
    task = asyncio.current_task()
    active_handlers.add(task)
    drained.clear()
    try:
        while True:
            await websocket.send_text(await websocket.receive_text())
    finally:
        active_handlers.discard(task)
        if not active_handlers:
            drained.set()


class GracefulWebSocketServer(Server):
    """Leaves signal handling to the caller.

    Uvicorn's own SIGTERM handling closes every open connection before it waits
    for anything, which drops in-flight WebSockets with code 1012.
    """

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def serve() -> None:
    port = int(os.getenv("PORT", "8000"))
    server = GracefulWebSocketServer(
        uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    )
    loop = asyncio.get_running_loop()
    drain_task = None

    async def drain_then_exit():
        global shutting_down
        shutting_down = True
        if active_handlers:
            try:
                await asyncio.wait_for(drained.wait(), timeout=DRAIN_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                pass
        server.should_exit = True

    def on_signal():
        nonlocal drain_task
        if drain_task is None:
            drain_task = loop.create_task(drain_then_exit())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, on_signal)

    await server.serve()


if __name__ == "__main__":
    asyncio.run(serve())
