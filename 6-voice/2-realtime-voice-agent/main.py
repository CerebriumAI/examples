import asyncio
import os
import subprocess
import sys
import time
from multiprocessing import Process

import aiohttp
import requests
from huggingface_hub import login
from loguru import logger
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams

from helpers import (
    ClearableDeepgramTTSService,
    AudioVolumeTimer,
    TranscriptionTimingLogger,
)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

os.environ["SSL_CERT"] = ""
os.environ["SSL_KEY"] = ""
os.environ["OUTLINES_CACHE_DIR"] = "/tmp/.outlines"

deepgram_voice: str = "aura-asteria-en"

login(token=os.environ.get("HF_TOKEN"))


# Run vllM Server in background process
def start_server():
    while True:
        process = subprocess.Popen(
            f"python -m vllm.entrypoints.openai.api_server --port 5000 --model meta-llama/Llama-3.2-3B-Instruct --dtype auto --max_model_len 60000 --gpu-memory-utilization 0.9 --api-key {os.environ.get('HF_TOKEN')}",
            shell=True,
        )
        process.wait()  # Wait for the process to complete
        logger.error("Server process ended unexpectedly. Restarting in 5 seconds...")
        time.sleep(7)  # Wait before restarting


# Start the server in a separate process
server_process = Process(target=start_server, daemon=True)
server_process.start()


async def main(room_url: str, token: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(
            name="STT", api_key=None, url="ws://127.0.0.1:8082/v1/listen"
        )

        tts = ClearableDeepgramTTSService(
            name="Voice",
            aiohttp_session=session,
            api_key=None,
            voice=deepgram_voice,
            base_url="http://127.0.0.1:8082/v1/speak",
        )

        llm = OpenAILLMService(
            name="LLM",
            api_key=os.environ.get("HF_TOKEN"),
            model="meta-llama/Llama-3.2-3B-Instruct",
            base_url="http://127.0.0.1:5000/v1",
        )

        messages = [
            {
                "role": "system",
                "content": "You are a fast, low-latency chatbot. Your goal is to demonstrate voice-driven AI capabilities at human-like speeds. The technology powering you is Daily for transport, Cerebrium for serverless infrastructure, Llama 3 (8-B version) LLM, and Deepgram for speech-to-text and text-to-speech. You are hosted on the east coast of the United States. Respond to what the user said in a creative and helpful way, but keep responses short and legible. Ensure responses contain only words. Check again that you have not included special characters other than '?' or '!'.",
            },
        ]

        avt = AudioVolumeTimer()
        tl = TranscriptionTimingLogger(avt)

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                avt,  # Audio volume timer
                stt,  # Speech-to-text
                tl,  # Transcription timing logger
                tma_in,  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                tma_out,  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        # When the first participant joins, the bot should introduce itself.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Kick off the conversation.
            time.sleep(1.5)
            messages.append(
                {
                    "role": "system",
                    "content": "Introduce yourself by saying 'hello, I'm FastBot, how can I help you today?'",
                }
            )
            await task.queue_frame(LLMMessagesFrame(messages))

        # When the participant leaves, we exit the bot.
        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        # If the call is ended make sure we quit as well.
        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)
        await session.close()


async def check_deepgram_model_status():
    url = "http://127.0.0.1:8082/v1/status/engine"
    headers = {"Content-Type": "application/json"}
    max_retries = 5
    async with aiohttp.ClientSession() as session:
        for _ in range(max_retries):
            print("Trying Deepgram local server")
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        print(json_response)
                        if json_response.get("engine_connection_status") == "Connected":
                            print("Connected to deepgram local server")
                            return True
            except aiohttp.ClientConnectionError:
                print("Connection refused, retrying...")
            await asyncio.sleep(10)
    return False


async def check_vllm_model_status():
    url = "http://127.0.0.1:5000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}",
    }
    data = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working?"},
        ],
    }
    max_retries = 5
    async with aiohttp.ClientSession() as session:
        for _ in range(max_retries):
            print("Trying vLLM local server")
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        print("vLLM server is ready and responding correctly")
                        return True
                    else:
                        print(f"Unexpected status code: {response.status}")
                        response_text = await response.text()
                        print(f"Response: {response_text}")
            except aiohttp.ClientConnectionError:
                print("vLLM Connection refused, retrying...")
            await asyncio.sleep(10)
    print("Failed to connect to vLLM server after multiple attempts")
    return False


async def start_bot(room_url: str, token: str = None):
    await check_vllm_model_status()
    await check_deepgram_model_status()

    try:
        await main(room_url, token)
    except Exception as e:
        logger.error(f"Exception in main: {e}")
        sys.exit(1)  # Exit with a non-zero status code

    return {"message": "session finished"}


def create_room():
    url = "https://api.daily.co/v1/rooms/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('DAILY_TOKEN')}",
    }
    data = {
        "properties": {
            "exp": int(time.time()) + 60 * 5,  ##5 mins
            "eject_at_room_exp": True,
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        room_info = response.json()
        token = create_token(room_info["name"])
        if token and "token" in token:
            room_info["token"] = token["token"]
        else:
            logger.error("Failed to create token")
            return {
                "message": "There was an error creating your room",
                "status_code": 500,
            }
        return room_info
    else:
        data = response.json()
        if data.get("error") == "invalid-request-error" and "rooms reached" in data.get(
            "info", ""
        ):
            logger.error(
                "We are currently at capacity for this demo. Please try again later."
            )
            return {
                "message": "We are currently at capacity for this demo. Please try again later.",
                "status_code": 429,
            }
        logger.error(f"Failed to create room: {response.status_code}")
        return {"message": "There was an error creating your room", "status_code": 500}


def create_token(room_name: str):
    url = "https://api.daily.co/v1/meeting-tokens"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('DAILY_TOKEN')}",
    }
    data = {"properties": {"room_name": room_name}}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        token_info = response.json()
        return token_info
    else:
        logger.error(f"Failed to create token: {response.status_code}")
        return None
