import os
import sys

import requests
from fastapi import HTTPException
from loguru import logger
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.services.openai import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.services.helpers.daily_rest import (
    DailyRESTHelper,
    DailyRoomObject,
    DailyRoomParams,
    DailyRoomProperties,
    DailyRoomSipParams,
)
from pipecat.vad.silero import SileroVADAnalyzer
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.twilio import TwilioFrameSerializer

from pipecat.services.cartesia import CartesiaTTSService

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

twilio = Client(
    os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN")
)


async def main(websocket_client, stream_sid):
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(stream_sid),
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(
        name="LLM",
        api_key=os.environ.get("OPENAI_API_KEY"),
        model="gpt-4",
    )
    

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    )   

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in an audio call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]
    print('here', flush=True)
    context = OpenAILLMContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)

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
            daily_api_key=os.environ.get("DAILY_TOKEN"),
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
