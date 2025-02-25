import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import librosa
import soundfile as sf
import time
from loguru import logger

from pipecat.frames.frames import (
    Frame, 
    AudioRawFrame,
    TranscriptionFrame,
    TextFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.vad.vad_analyzer import VADAnalyzer, VADState
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.transports.local.audio import LocalAudioTransport
from select_audio_device import AudioDevice, run_device_selector

# from model import UltravoxModel

# Add TTSFrame definition since it's missing from the package
@dataclass
class TTSFrame(Frame):
    audio: bytes

@dataclass 
class AudioBuffer:
    frames: List[AudioRawFrame] = field(default_factory=list)
    started_at: Optional[float] = None
    
class UltravoxProcessor(FrameProcessor):
    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        self.model = ""#model
        self.vad = VADAnalyzer()
        self.buffer = AudioBuffer()
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioRawFrame):
            vad_state = self.vad.process_frame(frame)
            
            if vad_state == VADState.SPEECH_START:
                self.buffer = AudioBuffer(frames=[frame], started_at=time.time())
                logger.info("Speech started")
            
            elif vad_state == VADState.SPEECH:
                if self.buffer.frames:
                    self.buffer.frames.append(frame)
                    
            elif vad_state == VADState.SPEECH_END:
                if self.buffer.frames:
                    logger.info("Speech ended, processing buffer...")
                    # Process buffered audio
                    audio_data = np.concatenate([f.audio for f in self.buffer.frames])
                    audio_16k = librosa.resample(audio_data, orig_sr=48000, target_sr=16000)
                    
                    # Generate transcription
                    async for response in self.model.generate(
                        prompt="<|audio|>\n",
                        temperature=0.7,
                        max_tokens=100,
                        audio=audio_16k
                    ):
                        logger.info(f"Generated response: {response.text}")
                        await self.push_frame(
                            TranscriptionFrame(text=response.text)
                        )
                        
                    self.buffer = AudioBuffer()

async def main():
    # Get audio devices
    res = await run_device_selector()
    input_device, output_device, _ = res
    
    # Initialize audio transport with direct parameters
    transport = LocalAudioTransport(
        input_device_index=input_device.index,
        output_device_index=output_device.index,
        input_enabled=True,
        output_enabled=True,
        sample_rate=48000,
        channels=1,
    )
    
    processor = UltravoxProcessor(model=None)
    tts = CartesiaTTSService(
        api_key=os.environ.get("CARTESIA_API_KEY"),
        voice_id='97f4b8fb-f2fe-444b-bb9a-c109783a857a',
        model="sonic",
        sample_rate=44100,
        encoding="pcm_f32le",
        container="raw"
    )

    # Create pipeline using transport.input() and transport.output()
    pipeline = Pipeline([transport.input(), processor, tts, transport.output()])
    task = PipelineTask(pipeline)
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)
    
    logger.info("Starting pipeline...")
    await asyncio.gather(runner.run(task))

if __name__ == "__main__":
    # Set up logging
    logger.remove(0)  # Remove default handler
    logger.add(sys.stderr, level="INFO")
    logger.add("pipeline.log", rotation="1 MB")
    
    # Run the pipeline
    asyncio.run(main())

