import asyncio
import math
import struct
import time
from typing import List

from loguru import logger
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.stt import AsyncListenWebSocketClient
from pipecat.services.deepgram.stt import LiveTranscriptionEvents


class CustomDeepgramSTTService(DeepgramSTTService):
    """Custom Deepgram STT service that uses a specific websocket URL"""

    def __init__(self, websocket_url=None, **kwargs):
        super().__init__(**kwargs)
        self._custom_websocket_url = websocket_url

    async def _connect(self):
        """Override the _connect method to set a custom websocket URL"""
        logger.debug("Connecting to Deepgram with custom service")

        self._connection: AsyncListenWebSocketClient = (
            self._client.listen.asyncwebsocket.v("1")
        )

        self._connection.on(
            LiveTranscriptionEvents(LiveTranscriptionEvents.Transcript),
            self._on_message,
        )
        self._connection.on(
            LiveTranscriptionEvents(LiveTranscriptionEvents.Error), self._on_error
        )

        if self.vad_enabled:
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.SpeechStarted),
                self._on_speech_started,
            )
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.UtteranceEnd),
                self._on_utterance_end,
            )

        # Set custom websocket URL if provided
        if self._custom_websocket_url:
            logger.debug(f"Original URL: {self._connection._websocket_url}")
            logger.debug(f"Setting custom websocket URL: {self._custom_websocket_url}")
            self._connection._websocket_url = self._custom_websocket_url

        if not await self._connection.start(
            options=self._settings, addons=self._addons
        ):
            logger.error(f"{self}: unable to connect to Deepgram")
