from fastapi import FastAPI
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    metrics,
)
from livekit import api
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero, cartesia
import os
import asyncio
import sys
import logging

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def create_sip_participant(phone_number, room_name):
    LIVEKIT_URL = os.getenv("LIVEKIT_URL")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
    SIP_TRUNK_ID = os.getenv("SIP_TRUNK_ID")

    livekit_api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    sip_trunk_id = SIP_TRUNK_ID
    try:
        await livekit_api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=sip_trunk_id,
                sip_call_to=phone_number,
                room_name=room_name,
                participant_identity=f"sip_{phone_number}",
                participant_name="SIP Caller",
            )
        )
        await livekit_api.aclose()
        return f"Call initiated to {phone_number}"
    except Exception as e:
        await livekit_api.aclose()
        return f"Error: {str(e)}"


async def entrypoint(ctx: JobContext):

    initial_ctx = llm.ChatContext().append(
        role="system",
        text="You are a voice assistant created by LiveKit. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation.",
    )

    fnc_ctx = llm.FunctionContext()

    @fnc_ctx.ai_callable()
    async def transfer_call(
        # by using the Annotated type, arg description and type are available to the LLM
    ):
        """Called when the receiver would like to be transferred to a real person. This function will add another participant to the call."""
        await create_sip_participant("<phone number to call>", "Test SIP Room")
        await agent.say(
            "Connecting you to my colleague - please hold on", allow_interruptions=True
        )

        await ctx.room.disconnect()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        # flexibility to use any models
        stt=deepgram.STT(model="nova-2-general"),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.5,
        ),
        tts=cartesia.TTS(),
        # intial ChatContext with system prompt
        chat_ctx=initial_ctx,
        # whether the agent can be interrupted
        allow_interruptions=True,
        # sensitivity of when to interrupt
        interrupt_speech_duration=0.5,
        interrupt_min_words=0,
        # minimal silence duration to consider end of turn
        min_endpointing_delay=0.3,
        fnc_ctx=fnc_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)
        usage_collector.collect(mtrcs)

    async def log_usage():
        agent.say("connecting you to a real person")
        summary = usage_collector.get_summary()
        print(f"Usage: ${summary}")

    ctx.add_shutdown_callback(log_usage)

    agent.start(ctx.room)
    await asyncio.sleep(1.2)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("start")
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM, port=8600)
    )
