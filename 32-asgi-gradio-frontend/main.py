import gradio as gr
from fastapi import FastAPI, Request
from starlette.responses import Response as StarletteResponse
import httpx
import os
import multiprocessing
import time
import requests
from typing import Optional
import sys

# Initialize FastAPI
app = FastAPI()

# Global variable for the Gradio server
gradio_server = None

# Get the Gradio app URL (when running on Cerebrium)
GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_URL = os.getenv("GRADIO_SERVER_URL", f"http://{GRADIO_HOST}:{GRADIO_PORT}")


class GradioServer:
    def __init__(self):
        self.host = GRADIO_HOST
        self.port = GRADIO_PORT
        self.process: Optional[multiprocessing.Process] = None
        self.url = GRADIO_URL

    def run_server(self):
        def greet(name):
            return f"Hello, {name}!"

        interface = gr.Interface(
            fn=greet,
            inputs="text",
            outputs="text",
            title="Greeting App"
        )
        interface.launch(
            server_name=self.host,
            server_port=self.port,
            quiet=True
        )

    def start(self):
        # Start Gradio in a separate process
        self.process = multiprocessing.Process(target=self.run_server)
        self.process.start()

        # Wait for Gradio to become ready
        max_retries = 30
        retry_delay = 1.0

        for _ in range(max_retries):
            try:
                response = requests.get(f"{self.url}/")
                if response.status_code == 200:
                    print(f"Gradio server is ready at {self.url}")
                    return True
            except requests.exceptions.ConnectionError:
                time.sleep(retry_delay)

        print("Failed to start Gradio server")
        self.stop()
        return False

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.join()
            self.process = None


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Catchall proxy endpoint for Gradio
@app.route("/{path:path}", include_in_schema=False, methods=["GET", "POST"])
async def gradio(request: Request):
    print(f"Proxying request to Gradio: {request.url.path}")
    headers = dict(request.headers)

    async with httpx.AsyncClient() as client:
        response = await client.request(
            request.method,
            f"{GRADIO_URL}",
            headers=headers,
            data=await request.body(),
            params=request.query_params,
        )

        content = await response.aread()
        response_headers = dict(response.headers)
        return StarletteResponse(
            content=content,
            status_code=response.status_code,
            headers=response_headers,
        )

@app.on_event("startup")
async def startup_event():
    global gradio_server
    if not os.getenv("GRADIO_SERVER_URL"):  # Only start local server if no external URL provided
        gradio_server = GradioServer()
        if not gradio_server.start():
            sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    global gradio_server
    if gradio_server:
        gradio_server.stop()
