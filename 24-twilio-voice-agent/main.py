import asyncio
import os
import subprocess
import sys
import time
from multiprocessing import Process

import aiohttp
import requests
from cerebrium import get_secret
from fastapi import HTTPException
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
from pipecat.transports.services.daily import (
    DailyParams,
    DailyTransport,
)
from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomObject,
    DailyRoomParams,
    DailyRoomProperties,
    DailyRoomSipParams,
)
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

from helpers import (
    ClearableDeepgramTTSService,
)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

os.environ["SSL_CERT"] = ""
os.environ["SSL_KEY"] = ""
os.environ["OUTLINES_CACHE_DIR"] = "/tmp/.outlines"

deepgram_voice: str = "aura-asteria-en"

twilio = Client(get_secret("TWILIO_ACCOUNT_SID"), get_secret("TWILIO_AUTH_TOKEN"))


def start_server():
    while True:
        process = subprocess.Popen(
            f"python -m vllm.entrypoints.openai.api_server --port 5000 --model NousResearch/Meta-Llama-3-8B-Instruct --dtype bfloat16 --api-key {get_secret('HF_TOKEN')}",
            shell=True,
        )
        process.wait()  # Wait for the process to complete
        logger.error("Server process ended unexpectedly. Restarting in 5 seconds...")
        time.sleep(7)  # Wait before restarting


# Start the server in a separate process
server_process = Process(target=start_server, daemon=True)
server_process.start()


async def main(room_url: str, token: str, callId: str, sipUri: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                api_key=get_secret("DAILY_TOKEN"),
                dialin_settings=None,
                audio_in_enabled=True,
                audio_out_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
                transcription_enabled=False,
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

        # tts = ElevenLabsTurbo(
        #     aiohttp_session=session,
        #     api_key=get_secret("ELEVENLABS_API_KEY"),
        #     voice_id=get_secret("ELEVENLABS_VOICE_ID"),
        # )

        # llm = OpenAILLMService(
        #     api_key=get_secret("OPENAI_API_KEY"),
        #     model="gpt-4o-mini"
        # )

        llm = OpenAILLMService(
            name="LLM",
            api_key=get_secret("HF_TOKEN"),
            model="NousResearch/Meta-Llama-3-8B-Instruct",
            base_url="http://127.0.0.1:5000/v1",
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in an audio call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                tma_in,
                llm,
                tts,
                transport.output(),
                tma_out,
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

        @transport.event_handler("on_dialin_ready")
        async def on_dialin_ready(transport, cdata):
            # For Twilio, Telnyx, etc. You need to update the state of the call
            # and forward it to the sip_uri..
            print(f"Forwarding call: {callId} {sipUri}")

            try:
                # The TwiML is updated using Twilio's client library
                call = twilio.calls(callId).update(
                    twiml=f"<Response><Dial><Sip>{sipUri}</Sip></Dial></Response>"
                )
            except Exception as e:
                raise Exception(f"Failed to forward call: {str(e)}")

        # If the call is ended make sure we quit as well.
        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)
        await session.close()


async def check_vllm_model_status():
    url = "http://127.0.0.1:5000/v1/models"
    headers = {"Authorization": f"Bearer {get_secret('HF_TOKEN')}"}

    max_retries = 5
    async with aiohttp.ClientSession() as session:
        for _ in range(max_retries):
            print("Trying vllm server")
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return True
            except aiohttp.ClientConnectionError:
                print("vLLM connection refused, retrying...")
            await asyncio.sleep(10)
    return False


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
                print("Deepgram Connection refused, retrying...")
            await asyncio.sleep(10)
    return False


async def send_request(url, data, headers):
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, json=data, headers=headers)
        print("Request sent successfully")
    except Exception as e:
        print(f"Error sending request: {str(e)}")


async def start_bot(request_data: dict):
    print(request_data)
    callId = request_data["CallSid"]

    # Create a room and spawn bot
    room = await create_room()
    # Make a request to an endpoint sending JSON data without waiting for the response
    endpoint_url = "https://api.cortex.cerebrium.ai/v4/p-c6754f15/twilio-agent/main"  # Replace with your actual endpoint URL
    payload = {
        "room_url": room["room_url"],
        "token": room["token"],
        "sipUri": room["sip_endpoint"],
        "callId": callId,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_secret('CEREBRIUM_BEARER_TOKEN')}",
        # Replace 'API_TOKEN' with your actual secret name if needed
    }

    # Use asyncio.create_task to run the request asynchronously without waiting
    asyncio.create_task(send_request(endpoint_url, payload, headers))

    # We have the room and the SIP URI,
    # but we do not know if the Daily SIP Worker and the Bot have joined the call
    # put the call on hold until the 'on_dialin_ready' fires.
    # The bot will call forward_twilio_call when it ready.
    resp = VoiceResponse()
    resp.play(
        url="http://com.twilio.sounds.music.s3.amazonaws.com/MARKOVICHAMP-Borghestral.mp3",
        loop=10,
    )
    return str(resp)


async def create_room():
    params = DailyRoomParams(
        properties=DailyRoomProperties(
            sip=DailyRoomSipParams(
                display_name="sip-dialin",
                video=False,
                sip_mode="dial-in",
                num_endpoints=1,
            )
        )
    )

    # Create sip-enabled Daily room via REST
    try:
        daily_helper = DailyRESTHelper(
            daily_api_key=get_secret("DAILY_TOKEN"),
            daily_api_url="https://api.daily.co/v1",
        )
        room: DailyRoomObject = daily_helper.create_room(params=params)

        token = daily_helper.get_token(room.url, 300)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to provision room {e}")

    print(f"Daily room returned {room.url} {room.config.sip_endpoint}")

    return {
        "room_url": room.url,
        "sip_endpoint": room.config.sip_endpoint,
        "token": token,
    }


def create_token(room_name: str):
    url = "https://api.daily.co/v1/meeting-tokens"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_secret('DAILY_TOKEN')}",
    }
    data = {"properties": {"room_name": room_name}}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        token_info = response.json()
        return token_info
    else:
        logger.error(f"Failed to create token: {response.status_code}")
        return None
