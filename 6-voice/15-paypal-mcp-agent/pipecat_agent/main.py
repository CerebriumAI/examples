import asyncio
import os
import subprocess
import sys
import time
from multiprocessing import Process
import shutil

import aiohttp
import requests
from loguru import logger
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner

from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from deepgram import LiveOptions
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from mcp import StdioServerParameters
from pipecat.services.mcp_service import MCPClient
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

from dotenv import load_dotenv

load_dotenv()


logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


deepgram_voice: str = "aura-asteria-en"


async def main(room_url: str, token: str, paypal_access_token: str, paypal_environment: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=False,
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.15)),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(
            api_key=os.environ.get("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                model="nova-3-general",
                language="en-US",
                smart_format=True,
                vad_events=True
            )
        )

        tts = CartesiaTTSService(
            api_key=os.environ.get("CARTESIA_API_KEY"),
            voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
        )

        print(f"Paypal access token: {paypal_access_token}")
        mcp = MCPClient(
            server_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "@paypal/mcp",
                    "--tools=all"
                ],
                env={"PAYPAL_ACCESS_TOKEN": paypal_access_token, "PAYPAL_ENVIRONMENT": paypal_environment},
            )
        )
        llm = OpenAILLMService(
            name="LLM",
            model="gpt-4.1",
        )

    # Create tools schema from the MCP server and register them with llm
        tools = await mcp.register_tools(llm)

        context = OpenAILLMContext(
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to PayPal tools. You have access to MCP tools. Before doing a tool call, please say 'Sure, give me a moment'",
                },
            ],
            tools=tools,
        )

        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # Speech-to-text
                context_aggregator.user(),
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True
            ),
        )

        # When the first participant joins, the bot should introduce itself.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Kick off the conversation.
            time.sleep(1.5)
            context.messages.append(
                {
                    "role": "system",
                    "content": "Introduce yourself by saying 'hello, I'm FastBot, how can I help you today?'",
                }
            )
            await task.queue_frame(LLMMessagesFrame(context.messages))

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


async def start_bot(paypal_access_token: str, paypal_environment: str):
    try:
        room_info = create_room()
        if "status_code" in room_info and room_info["status_code"] != 200:
            logger.error(f"Failed to create room: {room_info}")
            return {"message": "Failed to create room", "status_code": 500}

        room_url = room_info["url"]
        room_token = room_info["token"]
        
        # Start main() in background task so we can return room info immediately
        asyncio.create_task(main(room_url, room_token, paypal_access_token, paypal_environment))
        
        return {
            "message": "Room created successfully",
            "status_code": 200,
            "room_url": room_url
        }
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
    data = {"properties": {"room_name": room_name, "is_owner": True}}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        token_info = response.json()
        return token_info
    else:
        logger.error(f"Failed to create token: {response.status_code}")
        return None

##uncomment this to run locally
# if __name__ == "__main__":
#     """Initialize main function by creating room and token"""
#     room_info = create_room()
#     if "status_code" in room_info and room_info["status_code"] != 200:
#         logger.error(f"Failed to create room: {room_info}")
#         print(room_info)

#     room_url = room_info["url"]
#     room_token = room_info["token"]

#     print(f"Access token: {os.environ.get('PAYPAL_ACCESS_TOKEN')}")
#     asyncio.run(main(room_url=room_url, token=room_token, paypal_access_token=os.environ.get("PAYPAL_ACCESS_TOKEN"), paypal_environment="PRODUCTION"))

