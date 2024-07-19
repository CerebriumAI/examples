from loguru import logger
import asyncio
import math
import struct
import time
from dataclasses import dataclass, field
from typing import List
from typing import Union
import aiohttp

from pipecat.processors.frameworks.langchain import LangchainProcessor

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TextFrame,
    StartInterruptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TTSStoppedFrame,
    MetricsFrame
)

from pipecat.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMContextFrame

from pipecat.services.elevenlabs import ElevenLabsTTSService

class GreedyLLMAggregator(FrameProcessor):
    def __init__(self, context: OpenAILLMContext = None, **kwargs):
        super().__init__(**kwargs)
        self.context: OpenAILLMContext = context if context else OpenAILLMContext()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        logger.debug(f"{frame}")

        try:
            if isinstance(frame, InterimTranscriptionFrame):
                return

            if isinstance(frame, TranscriptionFrame):
                # append transcribed text to last "user" frame
                if self.context.messages and self.context.messages[-1]["role"] == "user":
                    last_frame = self.context.messages.pop()
                else:
                    last_frame = {"role": "user", "content": ""}

                last_frame["content"] += " " + frame.text
                self.context.messages.append(last_frame)

                oai_context_frame = OpenAILLMContextFrame(context=self.context)
                logger.debug(f"pushing frame {oai_context_frame}")
                await self.push_frame(oai_context_frame)
                return

            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"error: {e}")


class ClearableElevenLabsTTSService(ElevenLabsTTSService):
    def __init___(self, **kwargs):
        super().__init(**kwargs)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            self._current_sentence = ""

class LangchainRAGProcessor(LangchainProcessor):
    def __init__(self, chain: Runnable, transcript_key: str = "input"):
        super().__init__(chain, transcript_key)  
        self._chain = chain
        self._transcript_key = transcript_key

    @staticmethod
    def __get_token_value(text: Union[str, AIMessageChunk]) -> str:
        match text:
            case str():
                return text
            case AIMessageChunk():
                return text.content
            case dict() as d if 'answer' in d:
                return d['answer']
            case _:
                return ""
            
    async def _ainvoke(self, text: str):
        logger.debug(f"Invoking chain with {text}")
        targetPhrases = [
          "you can continue with the lecture",
          "continue with the lecture",
          "you can continue with lecture",
          "continue with lecture",
          "play the video",
          "continue with the video"
        ]

        ##Simple fuzzy matching by checking if the target phrase is included in the transcript text
        matchFound = any(phrase in text for phrase in targetPhrases)
        if matchFound:
            print("Fuzzy match found for the phrase: 'You can continue with the lecture'")
            return
        
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for token in self._chain.astream(
                {self._transcript_key: text},
                config={"configurable": {"session_id": self._participant_id}},
            ):
                await self.push_frame(LLMResponseStartFrame())
                await self.push_frame(TextFrame(self.__get_token_value(token)))
                await self.push_frame(LLMResponseEndFrame())
        except GeneratorExit:
            logger.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logger.exception(f"{self} an unknown error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

class ElevenLabsTurbo(ElevenLabsTTSService):
    def __init__(
            self,
            *,
            aiohttp_session: aiohttp.ClientSession,
            api_key: str,
            voice_id: str,
            model: str = "eleven_turbo_v2_5",
            **kwargs):
        super().__init__(aiohttp_session=aiohttp_session, api_key=api_key, voice_id=voice_id, **kwargs)

        self._api_key = api_key
        self._voice_id = voice_id
        self._aiohttp_session = aiohttp_session
        self._model = model


@dataclass
class BufferedSentence:
    audio_frames: List[AudioRawFrame] = field(default_factory=list)
    text_frame: TextFrame = None


class VADGate(FrameProcessor):

    def __init__(
            self,
            vad_analyzer: VADAnalyzer = None,
            context: OpenAILLMContext = None,
            **kwargs):
        super().__init__(**kwargs)
        self.vad_analyzer = vad_analyzer
        self.context = context

        self._audio_pusher_task = None
        self._expect_text_frame_next = False
        self._sentences: List[BufferedSentence] = []

    # queue output from tts one sentence at a time. associate a buffer of audio frames with the content of
    # each text frame.
    #
    # start a coroutine to service the queue and send sentences down the pipeline when possible.
    # 1. do not send anything when we are not in VADState.QUIET
    # 2. if we are in VADState.QUIET, send a sentence, estimate how long it will take for that sentence
    #    to output, sleep until it's time to send another sentence
    # 3. each time we send a sentence, append it to the conversation context
    # 3. when the sentence buffer becomes empty, cancel the coroutine
    # 4. if we get a new LLMFullResponse, treat that as a cancellation, too

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        try:

            # A TTSService will emit a series of AudioRawFrame objects, then a TTSStoppedFrame,
            # then a TextFrame.

            if self._expect_text_frame_next:
                self._expect_text_frame_next = False
                if isinstance(frame, TextFrame):
                    self._sentences[-1].text_frame = frame
                else:
                    logger.debug(f"expected a text frame, but received {frame}")
                    await self.push_frame(frame, direction)
                return
            else:
                if isinstance(frame, TextFrame):
                    logger.error("XXXXXXXXXXXXXXXXXXX received a text frame, wasn't expecting it.")

            if isinstance(frame, AudioRawFrame):
                # if our buffer is empty or has a "finished" sentence at the end,
                # then we need to start buffering a new sentence
                if not self._sentences or self._sentences[-1].text_frame:
                    self._sentences.append(BufferedSentence())
                self._sentences[-1].audio_frames.append(frame)
                await self.maybe_start_audio_pusher_task()
                return

            if isinstance(frame, TTSStoppedFrame):
                self._expect_text_frame_next = True
                await self.push_frame(frame, direction)
                return

            # There are two ways we can be interrupted. During greedy inference, a new
            # LLM response can start. Or, during playout, we can get a traditional
            # user interruption frame.
            if (isinstance(frame, LLMFullResponseStartFrame) or
                    isinstance(frame, StartInterruptionFrame)):
                logger.debug(f"{frame} - Handle interruption in VADGate")
                self._sentences = []
                if self._audio_pusher_task:
                    self._audio_pusher_task.cancel()
                    self._audio_pusher_task = None
                await self.push_frame(frame, direction)
                return

            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"error: {e}")

    async def maybe_start_audio_pusher_task(self):
        try:
            if self._audio_pusher_task:
                return
            self._audio_pusher_task = self.get_event_loop().create_task(self.push_audio())

        except Exception as e:
            logger.debug(f"Exception {e}")

    async def push_audio(self):
        try:
            while True:
                if not self._sentences:
                    await asyncio.sleep(0.01)
                    continue

                if self.vad_analyzer._vad_state != VADState.QUIET:
                    await asyncio.sleep(0.01)
                    continue

                # we only want to push completed sentence buffers
                if not self._sentences[0].text_frame:
                    await asyncio.sleep(0.01)
                    continue

                s = self._sentences.pop(0)
                if not s.audio_frames:
                    continue
                sample_rate = s.audio_frames[0].sample_rate
                duration = 0
                logger.debug(f"Pushing {len(s.audio_frames)} audio frames")
                for frame in s.audio_frames:
                    await self.push_frame(frame)
                    # assume linear16 encoding (2 bytes per sample). todo: add some more
                    # metadata to AudioRawFrame, maybe
                    duration += (len(frame.audio) / 2 / frame.num_channels) / sample_rate
                await asyncio.sleep(duration - 20 / 1000)
                if self.context:
                    logger.debug(f"Appending assistant message to context: [{s.text_frame.text}]")
                    self.context.messages.append(
                        {"role": "assistant", "content": s.text_frame.text}
                    )
                await self.push_frame(s.text_frame)

        except Exception as e:
            logger.debug(f"Exception {e}")


class TranscriptionTimingLogger(FrameProcessor):
    def __init__(self, avt):
        super().__init__()
        self.name = "Transcription"
        self._avt = avt

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                elapsed = time.time() - self._avt.last_transition_ts
                logger.debug(f"Transcription TTF: {elapsed}")
                await self.push_frame(MetricsFrame(ttfb={self.name: elapsed}))

            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"Exception {e}")


class AudioVolumeTimer(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.last_transition_ts = 0
        self._prev_volume = -80
        self._speech_volume_threshold = -50

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            volume = self.calculate_volume(frame)
            # print(f"Audio volume: {volume:.2f} dB")
            if (volume >= self._speech_volume_threshold and
                    self._prev_volume < self._speech_volume_threshold):
                # logger.debug("transition above speech volume threshold")
                self.last_transition_ts = time.time()
            elif (volume < self._speech_volume_threshold and
                    self._prev_volume >= self._speech_volume_threshold):
                # logger.debug("transition below non-speech volume threshold")
                self.last_transition_ts = time.time()
            self._prev_volume = volume

        await self.push_frame(frame, direction)

    def calculate_volume(self, frame: AudioRawFrame) -> float:
        if frame.num_channels != 1:
            raise ValueError(f"Expected 1 channel, got {frame.num_channels}")

        # Unpack audio data into 16-bit integers
        fmt = f"{len(frame.audio) // 2}h"
        audio_samples = struct.unpack(fmt, frame.audio)

        # Calculate RMS
        sum_squares = sum(sample**2 for sample in audio_samples)
        rms = math.sqrt(sum_squares / len(audio_samples))

        # Convert RMS to decibels (dB)
        # Reference: maximum value for 16-bit audio is 32767
        if rms > 0:
            db = 20 * math.log10(rms / 32767)
        else:
            db = -96  # Minimum value (almost silent)

        return db