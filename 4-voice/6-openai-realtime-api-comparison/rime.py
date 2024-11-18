from typing import AsyncGenerator
import aiohttp
from aiohttp import ClientTimeout

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService


class RimeTTSService(TTSService):
    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-helios-en",
        modelId: str,
        sample_rate: int = 16000,
        encoding: str = "linear16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._modelId = modelId
        self._settings = {
            "sample_rate": sample_rate,
            "encoding": encoding,
        }
        self.set_voice(voice)

    def can_generate_metrics(self) -> bool:
        return True

    def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        async def generator():
            logger.debug(f"Generating TTS: [{text}]")

            url = "https://model-7qkpx4ew.api.baseten.co/production/predict"
            payload = {
                "speaker": "grove",
                "text": text,
                "modelId": self._modelId,
                "samplingRate": self._settings["sample_rate"],
                "speedAlpha": 1.0,
                "reduceLatency": True,
                "pauseBetweenBrackets": False,
                "phonemizeBetweenBrackets": False,
                "respondStreaming": True,
            }
            print(payload)
            headers = {
                "Accept": "audio/pcm",
                "Authorization": f"Api-Key {self._api_key}",
                "Content-Type": "application/json",
            }

            try:
                await self.start_ttfb_metrics()

                async with aiohttp.ClientSession(
                    timeout=ClientTimeout(total=300)
                ) as session:
                    async with session.post(
                        url, json=payload, headers=headers
                    ) as response:
                        if response.status != 200:
                            raise ValueError(f"Rime API error: {response.status}")

                        await self.start_tts_usage_metrics(text)
                        yield TTSStartedFrame()

                        await self.stop_ttfb_metrics()

                        chunk_size = 8192  # Use a fixed buffer size
                        async for chunk in response.content.iter_any():
                            if chunk:
                                frame = TTSAudioRawFrame(
                                    audio=chunk,
                                    sample_rate=self._settings["sample_rate"],
                                    num_channels=1,
                                )
                                yield frame

                yield TTSStoppedFrame()

            except Exception as e:
                logger.exception(f"{self} exception: {e}")
                yield ErrorFrame(f"Error getting audio: {str(e)}")

        return (frame async for frame in generator())
