import requests
import asyncio
import aiohttp
import json
from typing import Dict, Any

# Endpoint configuration
ENDPOINT_URL = "https://api.cortex.cerebrium.ai/v4/p-xxxxxx/6-deepseek-sglang/run"
API_KEY = "<AUTH_TOKEN>"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

from openai import Client
openai_client = Client(
    api_key=API_KEY,
    base_url=ENDPOINT_URL
)

async def send_openai_request(session: aiohttp.ClientSession, i: int) -> None:
    try:
        stream = openai_client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            messages=[
                {
                    "role": "system", 
                    "content": "You will be given question answer tasks."
                },
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
            temperature=0.8,
            top_p=0.95,
            stream=True
        )
        
        for chunk in stream:  # The async iteration is where we actually await
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end='', flush=True)
        print("\nRequest {i} completed.\n", flush=True)  # Add newline at the end
    except Exception as e:
        print(f"OpenAI Request {i} failed: {str(e)}")


async def main():
    """Send 200 concurrent requests"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(2):
            print(f"Making request {i}")
            # tasks.append(send_request(session, i))
            tasks.append(send_openai_request(session,i))
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
