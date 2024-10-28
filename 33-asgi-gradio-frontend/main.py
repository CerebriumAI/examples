import multiprocessing
import os
import sys
import time
from typing import Optional, List

import gradio as gr
import httpx
import requests
from fastapi import FastAPI, Request
from starlette.responses import Response as StarletteResponse

# Initialize FastAPI
app = FastAPI()

# Global variable for the Gradio server
gradio_server = None

# Get the Gradio app URL (when running on Cerebrium)
GRADIO_HOST = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_URL = os.getenv("GRADIO_SERVER_URL", f"http://{GRADIO_HOST}:{GRADIO_PORT}")

# Configure the Llama endpoint URL
LLAMA_ENDPOINT = os.getenv("LLAMA_ENDPOINT", "<YOUR_MODEL_API_ENDPOINT>")  # Update with your endpoint


class GradioServer:
    def __init__(self):
        self.host = GRADIO_HOST
        self.port = GRADIO_PORT
        self.process: Optional[multiprocessing.Process] = None
        self.url = GRADIO_URL

    async def chat_with_llama(self, message: str, history: List[List[str]]) -> str:
        """Make a request to the Llama endpoint"""
        # Convert history and new message into OpenAI chat format
        messages = []
        for h in history:
            messages.extend([
                {"role": "user", "content": h[0]},
                {"role": "assistant", "content": h[1]}
            ])
        messages.append({"role": "user", "content": message})

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{LLAMA_ENDPOINT}/v1/chat/completions",
                    json={
                        "messages": messages,
                        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                        "stream": False,
                        "temperature": 0.7,
                        "top_p": 0.95
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    response_data = response.json()
                    return response_data['choices'][0]['text']
                else:
                    return f"Error: Received status code {response.status_code} from Llama endpoint"
            except Exception as e:
                return f"Error communicating with Llama endpoint: {str(e)}"

    def run_server(self):
        interface = gr.ChatInterface(
            fn=self.chat_with_llama,
            type="messages",
            title="Chat with Llama",
            description="This is a chat interface powered by Llama 3.1 8B Instruct",
            examples=[
                ["What is the capital of France?"],
                ["Explain quantum computing in simple terms"],
                ["Write a short poem about technology"]
            ],
        )
        interface.launch(
            server_name=self.host,
            server_port=self.port,
            root_path=f"https://dev-api.cortex.cerebrium.ai/v4/{os.getenv('PROJECT_ID')}/{os.getenv('APP_NAME')}/",
            quiet=True
        )

    def start(self):
        print(f"Starting Gradio server at {self.url} port {self.port}")

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
    print(f"Forwarding request path: {request.url.path}")

    headers = dict(request.headers)

    # Construct the full URL to Gradio, preserving the original path
    target_url = f"{GRADIO_URL}{request.url.path}"

    async with httpx.AsyncClient() as client:
        response = await client.request(
            request.method,
            target_url,
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
