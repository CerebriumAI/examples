from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions,cli, WorkerOptions, WorkerType
from livekit.plugins import (
    openai,
    cartesia,
    rime,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents import metrics, MetricsCollectedEvent
import sys
import os
load_dotenv()
usage_collector = metrics.UsageCollector()

os.environ["HF_HOME"] = "/cortex/.cache/"

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(base_url="ws://p-xxxxxx-deepgram.tenant-cerebrium-prod.svc.cluster.local/v1/listen", model="nova-3",),
        llm=openai.LLM(api_key=os.environ.get("CEREBRIUM_API_KEY"), model="RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8", base_url="http://p-xxxxxx-llama-llm.tenant-cerebrium-prod.svc.cluster.local/run"),
        tts=deepgram.TTS(base_url="http://p-xxxxx-deepgram.tenant-cerebrium-prod.svc.cluster.local/v1/speak", model="aura-stella-en"),
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        print(f"Usage: {summary}")

    # At shutdown, generate and log the summary from the usage collector
    ctx.add_shutdown_callback(log_usage)


if __name__ == '__main__':
    if len(sys.argv) == 1:
            sys.argv.append('dev')
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM, port=8600))
