# WebSocket Graceful Shutdown

Keep in-flight WebSocket connections alive while a container shuts down.

WebSockets need a custom runtime: the default Cerebrium runtime serves HTTP
functions only, so this app brings its own ASGI server via
`[cerebrium.runtime.custom]`.

## Why this is different to HTTP

Draining from FastAPI's `lifespan` is enough for HTTP requests, but not for
WebSockets. On SIGTERM, Uvicorn closes every open connection *before* it waits
for anything and before lifespan shutdown runs, so a WebSocket is dropped with
close code `1012` no matter what your drain logic does. Raising
`--timeout-graceful-shutdown` does not help, since it only bounds the wait
*after* connections are closed.

## How it works

1. `GracefulWebSocketServer` overrides `capture_signals` so Uvicorn never
   installs its own signal handlers and never closes connections itself.
2. The app installs its own SIGTERM handler, which marks the container not-ready
   and waits for active WebSocket handlers to finish.
3. Only once they have finished does it set `should_exit`, letting Uvicorn shut
   down normally.

Keep `entrypoint` in exec form, as above. Your process is started under an init
shim and must receive SIGTERM directly. Wrapping it in a shell
(`["sh", "-c", "python main.py"]`) means the shell gets the signal, exits
immediately, and takes the container down before your handler can run.

## Configuration

`response_grace_period` is the hard deadline before the container is killed. Keep
`DRAIN_TIMEOUT_SECONDS` comfortably below it so cleanup can finish.

## Try it

```bash
cerebrium deploy
```

```python
import asyncio, websockets

async def main():
    url = "wss://api.cortex.cerebrium.ai/v4/<project-id>/4-websocket-graceful-shutdown/ws"
    async with websockets.connect(url, additional_headers={"Authorization": "Bearer <token>"}) as ws:
        await ws.send("hello")
        print(await ws.recv())

asyncio.run(main())
```

Scale the app down while the socket is open: the connection stays up until it
closes on its own, instead of being cut immediately.
