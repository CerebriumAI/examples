
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import aiohttp
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    WorkerType,
    cli,
    function_tool,
    inference,
    metrics,
)
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.english import EnglishModel

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = "/cortex/.cache/"

LINKUP_API_KEY = os.getenv("LINKUP_API_KEY")
LINKUP_BASE_URL = "https://api.linkup.so/v1"
FETCH_MAX_CHARS = 2000

SEARCH_FILLERS = [
    "Still searching, one moment.",
    "Almost there.",
    "Hang tight, pulling that up now.",
]

INSTRUCTIONS = (
    "You are a friendly voice assistant. "
    "Answer greetings, small talk, explanations, opinions, and general knowledge "
    "directly — do NOT use tools for those. "
    "Use search_web only when the user needs live or current information: "
    "news, prices, weather, who holds a role today, sports scores, recent events, "
    "or fact-checking a specific claim about the real world. "
    "Use fetch_webpage only when the user gives you a specific URL to read. "
    "Call search_web at most once per question — never retry if results are returned. "
    "When the tool returns an answer, state the key facts from it in one concise reply. "
    "If the tool returns NO_RESULTS, say you couldn't find current information. "
    "Keep spoken answers under three sentences unless the user asks for more. "
    "Never read raw URLs aloud; cite sources by site or publication name."
)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"[*_#`]", "", text).strip()


def _source_label(source: dict) -> str:
    return (
        source.get("name")
        or source.get("title")
        or urlparse(source.get("url", "")).netloc.replace("www.", "")
        or "an online source"
    )


def _format_sourced_answer(data: dict) -> str:
    answer = (data.get("answer") or "").strip()
    sources = data.get("sources") or data.get("results") or []

    if not answer and sources:
        snippets = []
        for source in sources[:3]:
            snippet = source.get("snippet") or source.get("content") or ""
            if snippet:
                snippets.append(f"{_source_label(source)}: {_strip_markdown(snippet)}")
        answer = " ".join(snippets)

    if not answer:
        return "NO_RESULTS: Web search returned no useful information for this query."

    answer = _strip_markdown(answer)
    source_lines = []
    for source in sources[:3]:
        snippet = source.get("snippet") or source.get("content") or ""
        if snippet:
            source_lines.append(
                f"- {_source_label(source)}: {_strip_markdown(snippet)}"
            )

    if source_lines:
        return f"Answer: {answer}\n\nSources:\n" + "\n".join(source_lines)
    return f"Answer: {answer}"


def _normalize_llm_base_url(url: str) -> str:
    """OpenAI clients expect .../v1; strip a trailing /chat/completions if present."""
    return url.rstrip("/").removesuffix("/chat/completions")


def _build_llm() -> inference.LLM | openai.LLM:
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        base_url = _normalize_llm_base_url(base_url)
        logger.info("Using custom LLM at %s", base_url)
        return openai.LLM(
            model=os.getenv("LLM_MODEL", "Qwen/Qwen3.6-35B-A3B"),
            base_url=base_url,
            api_key=os.getenv("CEREBRIUM_API_KEY"),
            temperature=0.4,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    logger.info("Using LiveKit Inference LLM (%s)", os.getenv("LLM_MODEL", "google/gemma-4-31b-it"))
    return inference.LLM(
        model=os.getenv("LLM_MODEL", "google/gemma-4-31b-it"),
        extra_kwargs={"temperature": 0.4, "parallel_tool_calls": False},
    )


class WebSearchAgent(Agent):
    def __init__(self, http_session: aiohttp.ClientSession) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._http = http_session

    async def _linkup_request(self, endpoint: str, payload: dict) -> dict | None:
        headers = {
            "Authorization": f"Bearer {LINKUP_API_KEY}",
            "Content-Type": "application/json",
        }
        async with self._http.post(
            f"{LINKUP_BASE_URL}/{endpoint}",
            headers=headers,
            json=payload,
        ) as response:
            if response.status != 200:
                body = await response.text()
                logger.error("Linkup %s error %s: %s", endpoint, response.status, body)
                return None
            return await response.json()

    @function_tool()
    async def search_web(
        self,
        ctx: RunContext,
        query: Annotated[str, "Focused web search query for one specific fact or topic"],
    ) -> str:
        """Search the web for real-time information using Linkup fast search."""
        ctx.session.say(
            "I'm checking the web for you right now.",
            allow_interruptions=True,
        )

        try:
            async with ctx.with_filler(
                lambda step: SEARCH_FILLERS[step],
                delay=5,
                interval=10,
                max_steps=len(SEARCH_FILLERS),
            ):
                search_start = time.perf_counter()
                data = await self._linkup_request(
                    "search",
                    {
                        "q": query,
                        "depth": "fast",
                        "outputType": "searchResults",
                        "maxResults": 3,
                        "includeImages": False,
                    },
                )
                search_elapsed = time.perf_counter() - search_start
        except Exception:
            logger.exception("search_web failed")
            return "NO_RESULTS: Search failed due to a technical error."

        logger.info("Linkup search %r took %.3fs", query, search_elapsed)

        if not data:
            return "NO_RESULTS: Could not reach the search service."

        result = _format_sourced_answer(data)
        logger.info("Linkup search %r -> %s", query, result[:300])
        return result

    @function_tool()
    async def fetch_webpage(
        self,
        ctx: RunContext,
        url: Annotated[str, "The full URL of the webpage to read"],
    ) -> str:
        """Fetch and summarize content from a specific webpage URL."""
        ctx.session.say(
            "Let me pull up that page for you.",
            allow_interruptions=True,
        )

        try:
            async with ctx.with_filler(
                lambda step: SEARCH_FILLERS[step],
                delay=5,
                interval=10,
                max_steps=len(SEARCH_FILLERS),
            ):
                data = await self._linkup_request(
                    "fetch",
                    {
                        "url": url,
                        "outputFormat": "markdown",
                        "renderJS": False,
                    },
                )
        except Exception:
            logger.exception("fetch_webpage failed")
            return "I hit an error while fetching that page. Please try again."

        if not data:
            return "I couldn't fetch that page. Please check the URL and try again."

        content = _strip_markdown(data.get("content", ""))
        if not content:
            return "That page didn't return any readable content."

        if len(content) > FETCH_MAX_CHARS:
            content = content[:FETCH_MAX_CHARS] + "..."
        return content


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    usage_collector = metrics.UsageCollector()

    http_session = aiohttp.ClientSession()

    async def cleanup():
        if not http_session.closed:
            await http_session.close()
        logger.info("Usage: %s", usage_collector.get_summary())

    ctx.add_shutdown_callback(cleanup)

    agent = WebSearchAgent(http_session)
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3-general",
            language="en",
            smart_format=True,
            filler_words=True,
            punctuate=True,
        ),
        llm=_build_llm(),
        tts=cartesia.TTS(
            model="sonic-2",
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",
            speed="normal",
            emotion=["positivity:high", "curiosity:high"],
        ),
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
        turn_handling={
            "endpointing": {
                "mode": "dynamic",
                "min_delay": 0.3,
                "max_delay": 3.0,
            },
            "interruption": {
                "mode": "adaptive",
                "resume_false_interruption": True,
            },
            "preemptive_generation": {"enabled": False},
        },
    )

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)
        metrics.log_metrics(ev.metrics)

    await session.start(
        agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions=(
            "Greet the user briefly. Mention you can chat normally "
            "or search the web for live information when needed."
        )
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("start")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            worker_type=WorkerType.ROOM,
            port=8600,
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
            ws_url=os.getenv("LIVEKIT_URL"),
        )
    )
