import os
import sys
import time

import aiohttp
import requests
from loguru import logger
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.services.openai import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)
from openai.types.chat import ChatCompletionToolParam
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from rime import RimeTTSService  # Make sure to import the RimeTTSService
from pipecat.services.openai_realtime_beta import (
    OpenAILLMServiceRealtimeBeta,
    SessionProperties,
    InputAudioTranscription,
    TurnDetection,
)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

os.environ["SSL_CERT"] = ""
os.environ["SSL_KEY"] = ""
os.environ["OUTLINES_CACHE_DIR"] = "/tmp/.outlines"

current_service = "openai_realtime"


async def switch_service(
    function_name, tool_call_id, args, llm, context, result_callback
):
    global current_service
    print("Switching....!!!")
    current_service = args["service"]
    await result_callback(
        {
            "voice": f"Switching to {current_service} service. Future voice responses will be using the new service"
        }
    )


async def openai_realtime_filter(frame) -> bool:
    return current_service == "openai_realtime"


async def custom_filter(frame) -> bool:
    return current_service == "custom"


async def combined_main(room_url: str, token: str):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Friendly bot",
            DailyParams(
                audio_in_enabled=True,
                audio_in_sample_rate=24000,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6)),
                vad_audio_passthrough=True,
            ),
        )

        # OpenAI Realtime service
        session_properties = SessionProperties(
            input_audio_transcription=InputAudioTranscription(),
            # turn_detection=False,
            turn_detection=TurnDetection(silence_duration_ms=1000),
            instructions="""
                You are a helpful and friendly AI assistant. Keep your responses concise and to the point.
                You can switch between two services: 'openai_realtime' and 'custom'. Use the switch_service
                function when asked to change services.
                """,
        )
        openai_realtime = OpenAILLMServiceRealtimeBeta(
            api_key=os.environ.get("OPENAI_API_KEY"),
            session_properties=session_properties,
            start_audio_paused=False,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that can switch between two services to showcase the difference in performance and cost: 'openai_realtime' and 'custom'. Respond to user queries and switch services when asked.",
            },
        ]
        openai_realtime.register_function("switch_service", switch_service)
        tools_realtime = [
            {
                "type": "function",
                "name": "switch_service",
                "description": "Switch to the service when the user asks you to",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {
                            "type": "string",
                            "description": "The service the user wants you to switch to",
                        },
                    },
                    "required": ["service"],
                },
            }
        ]

        openai_realtime_context = OpenAILLMContext(
            messages=messages, tools=tools_realtime
        )
        context_aggregator = openai_realtime.create_context_aggregator(
            openai_realtime_context
        )

        # OpenAI LLM + Rime TTS service
        openai_llm = OpenAILLMService(
            name="LLM",
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4",
        )
        openai_llm.register_function("switch_service", switch_service)

        rime_tts = RimeTTSService(
            name="Voice",
            api_key=os.environ.get("RIME_API_KEY"),
            voice="grove",
            modelId="mist",
            sample_rate=24000,
            encoding="linear16",
        )

        tools_2 = [
            ChatCompletionToolParam(
                name="switch_service",
                type="function",
                function={
                    "type": "function",
                    "name": "switch_service",
                    "description": "Switch to the service when the user asks you to",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {
                                "type": "string",
                                "description": "The service the user wants you to switch to",
                            },
                        },
                        "required": ["service"],
                    },
                },
            )
        ]

        custom_context = OpenAILLMContext(messages=messages, tools=tools_2)
        context_aggregator_custom = openai_llm.create_context_aggregator(custom_context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                ParallelPipeline(
                    [
                        # openai_realtime_beta
                        FunctionFilter(openai_realtime_filter),
                        Pipeline(
                            [
                                context_aggregator.user(),
                                openai_realtime,  # LLM
                                context_aggregator.assistant(),
                            ]
                        ),
                    ],
                    # local inference
                    [
                        FunctionFilter(custom_filter),
                        Pipeline(
                            [
                                # stt,
                                context_aggregator_custom.user(),
                                openai_llm,
                                rime_tts,
                                context_aggregator_custom.assistant(),
                            ]
                        ),
                    ],
                ),
                transport.output(),  # Transport bot output
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            time.sleep(1.5)
            messages.append(
                {
                    "role": "system",
                    "content": "Introduce yourself and explain that you can switch between 'openai_realtime' and 'openai_rime' services.",
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])
            # await task.queue_frame(LLMMessagesFrame(messages)) ##if you start with custom first

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await runner.run(task)
        await session.close()


async def start_bot(room_url: str, token: str = None):
    try:
        await combined_main(room_url, token)

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
