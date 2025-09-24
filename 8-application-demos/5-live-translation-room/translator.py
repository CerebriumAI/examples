from daily import EventHandler, CallClient, Daily
import numpy as np
import threading
import time
import asyncio
import logging
import os
from scipy import signal
from pipecat.services.whisper import WhisperSTTService, Model
from pipecat.frames.frames import (
    AudioRawFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    ErrorFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from cartesia import Cartesia

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("translator")

cartesia_client = Cartesia(api_key=os.environ.get("CARTESIA_API_KEY"))
ws = cartesia_client.tts.websocket()

# TTS output format
OUTPUT_FORMAT = {
    "container": "raw",
    "encoding": "pcm_f32le",
    "sample_rate": 44100,
}

VOICE_CONFIG = {
    "en": {
        "voice_id": "71a7ad14-091c-4e8e-a314-022ece01c121",  # English voice
        "model_id": "sonic-2",
    },
    "es": {
        "voice_id": "5c5ad5e7-1020-476b-8b91-fdcbe9cc313c",  # Spanish voice
        "model_id": "sonic-2",
    },
    "default": {
        "voice_id": "71a7ad14-091c-4e8e-a314-022ece01c121",  # Default to English
        "model_id": "sonic-2",
    },
}


class AudioBuffer:
    def __init__(self):
        self.frames = []
        self.audio_arrays = []
        self.started_at = None
        self.last_updated_at = None
        self.is_processing = False
        self.speaker_id = None
        self.speech_start_frame = None
        self.detected_language = None


class TranslationService(EventHandler):
    CHANNELS = 1
    SAMPLE_RATE = 48000
    SPEECH_THRESHOLD = 0.3
    SILENCE_THRESHOLD_MS = 300
    PROCESSING_CHUNK_MS = 1000
    PROCESSING_INTERVAL_MS = 750
    VAD_RESET_PERIOD_MS = 1000
    VOLUME_REDUCTION = 0.4
    MAX_BUFFER_DURATION_SEC = 10

    def __init__(self, room_url, target_language="es", user_name="Guest", user_id=None):
        """Initialize the translation service"""
        logger.info(
            f"Initializing Translation Service for user_name={user_name}, room_url={room_url}, "
            f"target_language={target_language}, user_id={user_id}"
        )

        # Daily.js client setup
        self.client = CallClient(event_handler=self)
        self._running = True
        self.room_url = room_url
        self.target_language = target_language
        self.user_name = user_name
        self.user_id = user_id
        self.local_participant_id = None

        # Audio processing state
        self.audio_buffer = AudioBuffer()
        self.volume_levels = {}
        self.is_translator_speaking = False
        self.active_participants = set()
        self.participant_languages = {}  # Track language per participant
        self.translation_context = []
        self.max_context_sentences = 3

        # Translation queue
        self.translation_queue = asyncio.Queue(maxsize=10)
        self.active_translation_process = False

        # Initialize VAD
        self.vad = Daily.create_native_vad(
            reset_period_ms=self.VAD_RESET_PERIOD_MS,
            sample_rate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
        )

        # Initialize microphone device
        try:
            self.mic_device = Daily.create_microphone_device(
                "translator-mic",
                sample_rate=16000,
                channels=self.CHANNELS,
                non_blocking=True,
            )
            logger.info("Microphone device created")
        except Exception as e:
            logger.error(f"Error creating microphone: {e}")
            self.mic_device = None

        # Set client inputs
        self.client.update_inputs(
            {
                "camera": False,
                "microphone": {
                    "isEnabled": True,
                    "settings": {"deviceId": "translator-mic"},
                },
            }
        )

        # Initialize models
        self._init_models()

        # Start streaming processor thread
        self.streaming_thread = threading.Thread(
            target=self._streaming_processor, daemon=True
        )
        self.streaming_thread.start()

        # Start queue processor with dedicated event loop
        self.queue_event_loop = asyncio.new_event_loop()

        def run_queue_processor():
            asyncio.set_event_loop(self.queue_event_loop)
            self.queue_event_loop.run_until_complete(
                self._translation_queue_processor()
            )

        self.queue_thread = threading.Thread(target=run_queue_processor, daemon=True)
        self.queue_thread.start()

    def _init_models(self):
        """Initialize the STT and translation models"""
        try:
            # Initialize Whisper STT
            self.stt_service = WhisperSTTService(
                model=Model.LARGE,
                sample_rate=16000,
                no_speech_prob=0.6,
                beam_size=1,
                vad_parameters={"min_silence_duration_ms": self.SILENCE_THRESHOLD_MS},
            )

            # Initialize translation models for each supported language pair
            self.translation_models = {}

            # For now, initialize models for English and Spanish
            language_pairs = [
                ("en", "es"),  # English to Spanish
                ("es", "en"),  # Spanish to English
                # Add more language pairs as needed
            ]

            for source_lang, target_lang in language_pairs:
                model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    self.translation_models[(source_lang, target_lang)] = {
                        "tokenizer": tokenizer,
                        "model": model,
                        "pipeline": pipeline(
                            "translation", model=model, tokenizer=tokenizer
                        ),
                    }
                    logger.info(
                        f"Loaded translation model: {source_lang} to {target_lang}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading translation model {source_lang} to {target_lang}: {e}"
                    )

            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def on_participant_left(self, participant, reason):
        """Handle participant leaving the call"""
        if len(self.client.participant_counts()) <= 2:
            self._running = False

        if participant["id"] in self.active_participants:
            self.active_participants.remove(participant["id"])

        if participant["id"] in self.volume_levels:
            del self.volume_levels[participant["id"]]

        if participant["id"] in self.participant_languages:
            del self.participant_languages[participant["id"]]

    def on_participant_joined(self, participant):
        """Handle new participant joining the call"""
        if participant["info"]["isLocal"]:
            # Store the local participant ID
            self.local_participant_id = participant["id"]

            # If user_id not provided, default to local participant
            if self.user_id is None:
                self.user_id = participant["id"]

            logger.info(
                f"Local participant joined with ID: {self.local_participant_id}"
            )

        if (
            not participant["info"]["isLocal"]
            or participant["id"] != self.local_participant_id
        ):
            # Set up audio rendering for non-local participants
            self.client.set_audio_renderer(
                participant["id"],
                callback=self.on_audio_frame,
                audio_source="microphone",
            )

            # Initialize volume level
            self.volume_levels[participant["id"]] = 1.0

            # Add to active participants
            self.active_participants.add(participant["id"])

            # Default language detection will happen during speech processing
            self.participant_languages[participant["id"]] = None

            logger.info(f"Set up audio rendering for participant {participant['id']}")

    def on_audio_frame(self, participant, frame):
        """Process incoming audio frames and detect speech"""
        try:
            # Skip processing if this is from the user who needs translation (they speak, not listen)
            if participant == self.user_id:
                return

            current_time_ms = time.time() * 1000

            # Skip processing if we already have a large buffer being processed
            if (
                self.audio_buffer.is_processing
                and self.audio_buffer.started_at is not None
                and len(self.audio_buffer.frames) > 50
            ):
                return

            # Use VAD with dynamic threshold adjustment
            base_threshold = self.SPEECH_THRESHOLD

            # Lower threshold for continuing speech, higher for new speech
            adjusted_threshold = (
                base_threshold * 0.8
                if self.audio_buffer.started_at is not None
                else base_threshold * 1.2
            )

            confidence = self.vad.analyze_frames(frame.audio_frames)

            self._adjust_volume(participant, frame)

            if confidence > adjusted_threshold:
                audio_raw_frame = AudioRawFrame(
                    audio=frame.audio_frames,
                    sample_rate=16000,
                    num_channels=self.CHANNELS,
                )

                if self.audio_buffer.started_at is None:
                    speech_start_frame = UserStartedSpeakingFrame()
                    self.audio_buffer.speech_start_frame = speech_start_frame
                    self.audio_buffer.started_at = current_time_ms
                    self.audio_buffer.speaker_id = participant
                    self.audio_buffer.frames = []
                    self.audio_buffer.audio_arrays = []
                    self.audio_buffer.detected_language = None
                    logger.debug(f"New speech detected from participant {participant}")

                self.audio_buffer.last_updated_at = current_time_ms

                # Only collect frames from the same speaker
                if participant == self.audio_buffer.speaker_id:
                    frame_data = np.frombuffer(frame.audio_frames, dtype=np.int16)
                    if frame_data.size == 0:
                        return

                    # Process audio data (with optimized numpy operations)
                    frame_float = frame_data.astype(np.float32) / 32767.0

                    # Ensure mono audio
                    if len(frame_float.shape) > 1 and frame_float.shape[1] > 1:
                        frame_float = np.mean(frame_float, axis=1)

                    # Reshape and normalize
                    frame_float = np.reshape(frame_float, -1)

                    # Use inplace operations where possible to reduce memory usage
                    frame_float -= np.mean(frame_float)

                    # Apply noise gate (more efficiently)
                    noise_gate = 0.01
                    noise_mask = abs(frame_float) < noise_gate
                    frame_float[noise_mask] = 0

                    # Add to buffer if valid and not too large
                    if (
                        not np.isnan(frame_float).any()
                        and not np.isinf(frame_float).any()
                        and len(self.audio_buffer.frames) < 200
                    ):
                        self.audio_buffer.frames.append(audio_raw_frame)
                        self.audio_buffer.audio_arrays.append(frame_float)
            else:
                # Check for silence after speech
                if (
                    self.audio_buffer.started_at is not None
                    and self.audio_buffer.last_updated_at is not None
                ):
                    silence_duration = (
                        current_time_ms - self.audio_buffer.last_updated_at
                    )

                    adaptive_silence = self.SILENCE_THRESHOLD_MS
                    if len(self.audio_buffer.frames) > 50:
                        adaptive_silence = self.SILENCE_THRESHOLD_MS * 1.5

                    if silence_duration > adaptive_silence:
                        speech_stop_frame = UserStoppedSpeakingFrame()
                        self._finalize_speech_buffer(speech_stop_frame)
                        logger.debug(
                            f"Speech ended after {len(self.audio_buffer.frames)} frames"
                        )
        except Exception as e:
            logger.error(f"Audio frame error: {e}")

    def _adjust_volume(self, participant, frame):
        """
        Adjust the volume of the original speaker when translation is playing
        """
        try:
            if (
                self.is_translator_speaking
                and participant == self.audio_buffer.speaker_id
            ):
                # Lower the volume but don't mute completely
                new_volume = self.VOLUME_REDUCTION
                current_volume = self.volume_levels.get(participant, 1.0)

                if current_volume != new_volume:
                    self.volume_levels[participant] = new_volume

                    # Apply volume adjustment using Daily's API if available
                    if hasattr(self.client, "set_participant_volume"):
                        try:
                            self.client.set_participant_volume(participant, new_volume)
                            logger.debug(
                                f"Reduced volume for participant {participant} to {new_volume}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to adjust participant volume: {e}")
            elif (
                participant in self.volume_levels
                and self.volume_levels[participant] != 1.0
            ):
                # Restore volume to normal when not speaking
                self.volume_levels[participant] = 1.0

                if hasattr(self.client, "set_participant_volume"):
                    try:
                        self.client.set_participant_volume(participant, 1.0)
                        logger.debug(f"Restored volume for participant {participant}")
                    except Exception as e:
                        logger.warning(f"Failed to restore participant volume: {e}")
        except Exception as e:
            logger.warning(f"Error in volume adjustment: {e}")

    def _streaming_processor(self):
        """Background thread for continuous monitoring of speech"""
        while self._running:
            try:
                current_time_ms = time.time() * 1000

                # Check if we have an active buffer
                if (
                    self.audio_buffer.started_at is not None
                    and not self.audio_buffer.is_processing
                    and len(self.audio_buffer.frames) > 0
                ):

                    buffer_duration = current_time_ms - self.audio_buffer.started_at
                    last_process_duration = (
                        current_time_ms - self.audio_buffer.last_updated_at
                        if hasattr(self.audio_buffer, "last_processed_at")
                        else buffer_duration
                    )

                    # Process in chunks during ongoing speech - even if translator is speaking
                    if (
                        buffer_duration > self.PROCESSING_CHUNK_MS
                        and last_process_duration > self.PROCESSING_INTERVAL_MS
                    ):
                        # Special handling for concurrent speech and translation
                        if self.is_translator_speaking:
                            # Process with increased chunk threshold to avoid too many interruptions
                            if buffer_duration > self.PROCESSING_CHUNK_MS * 1.5:
                                logger.debug(
                                    "Processing speech chunk while translator is speaking"
                                )
                                self._process_speech_chunk(is_final=False)
                        else:
                            # Normal processing
                            self._process_speech_chunk(is_final=False)

                    # Process if buffer is too large
                    max_buffer_duration = self.MAX_BUFFER_DURATION_SEC * 1000
                    if buffer_duration > max_buffer_duration:
                        logger.info(
                            f"Processing speech chunk due to large buffer size ({buffer_duration / 1000:.1f}s)"
                        )
                        self._process_speech_chunk(is_final=True)

                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Streaming processor error: {e}")

    async def _generate_and_play_speech(self, text, source_language=None):
        """Generate speech from text and play it back to the target user"""
        self.is_translator_speaking = True
        logger.info(f"Generating speech for text: '{text}'")

        # Store active speaker for volume adjustment
        active_speaker_id = self.audio_buffer.speaker_id

        try:
            # Ensure mic device is ready for playback
            if not self.mic_device:
                self.mic_device = Daily.create_microphone_device(
                    "translator-mic",
                    sample_rate=16000,
                    channels=self.CHANNELS,
                    non_blocking=True,
                )
                logger.info("Created new microphone device")

            # Get voice configuration based on target language
            voice_config = VOICE_CONFIG.get(
                self.target_language, VOICE_CONFIG["default"]
            )
            logger.info(
                f"Using voice for language {self.target_language}: {voice_config['voice_id']}"
            )

            # Set up audio chunks collection
            audio_chunks = []
            start_time = time.time()
            timeout_sec = 10

            # Generate speech with Cartesia
            try:
                for output in ws.send(
                    model_id=voice_config["model_id"],
                    transcript=text,
                    voice_id=voice_config["voice_id"],
                    output_format=OUTPUT_FORMAT,
                    stream=True,
                ):
                    # Check timeout
                    if time.time() - start_time > timeout_sec:
                        logger.warning("TTS timeout reached")
                        break

                    buffer = output["audio"]
                    if not buffer:
                        continue

                    audio_chunks.append(buffer)

                    # If we have the first chunk, start playback to the target user
                    if len(audio_chunks) == 1:
                        logger.info(
                            f"Playing first audio chunk to user {self.user_id}: {len(buffer)} bytes"
                        )

                        # If we're playing to a specific user and not the local participant
                        if self.user_id != self.local_participant_id:
                            try:
                                # Only send audio to the specific user if we have the API
                                if hasattr(self.client, "send_audio_to_participant"):
                                    self.client.send_audio_to_participant(
                                        self.user_id, buffer
                                    )
                                else:
                                    # Fallback: play to everyone through the mic
                                    self.mic_device.write_frames(buffer)
                            except Exception as e:
                                logger.error(
                                    f"Error sending audio to user {self.user_id}: {e}"
                                )
                                # Fallback to the microphone device
                                self.mic_device.write_frames(buffer)
                        else:
                            # Default playback for local participant
                            self.mic_device.write_frames(buffer)
            except Exception as e:
                logger.error(f"Error in TTS service: {str(e)}")
                return

            # Play remaining chunks
            if len(audio_chunks) > 1:
                remaining_buffer = b"".join(audio_chunks[1:])
                if remaining_buffer:
                    audio_duration = (
                        len(remaining_buffer) / OUTPUT_FORMAT["sample_rate"] / 4
                    )
                    logger.info(
                        f"Playing remaining audio: {len(remaining_buffer)} bytes, duration ~{audio_duration:.2f}s"
                    )

                    try:
                        # Play in smaller chunks for more reliable playback
                        chunk_size = 8192  # 8KB chunks
                        for i in range(0, len(remaining_buffer), chunk_size):
                            # If a new speech buffer has started with a different speaker during playback,
                            # ensure we're still processing their frames (don't block new speech)
                            if (
                                self.audio_buffer.speaker_id is not None
                                and active_speaker_id is not None
                                and self.audio_buffer.speaker_id != active_speaker_id
                            ):

                                # Process any accumulated speech from new speaker
                                if (
                                    not self.audio_buffer.is_processing
                                    and len(self.audio_buffer.frames) > 20
                                ):
                                    logger.debug(
                                        "Processing new speech while translation is playing"
                                    )
                                    # Use a separate thread to process speech in parallel
                                    threading.Thread(
                                        target=self._process_speech_chunk,
                                        args=(False, None),
                                        daemon=True,
                                    ).start()

                            # Get the current chunk
                            chunk = remaining_buffer[i : i + chunk_size]

                            # Play to specific user if possible
                            if self.user_id != self.local_participant_id and hasattr(
                                self.client, "send_audio_to_participant"
                            ):
                                try:
                                    self.client.send_audio_to_participant(
                                        self.user_id, chunk
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Error sending chunk to user {self.user_id}: {e}"
                                    )
                                    self.mic_device.write_frames(chunk)
                            else:
                                # Default playback for local participant
                                self.mic_device.write_frames(chunk)

                            await asyncio.sleep(0.05)  # Small delay between chunks

                        # Wait for playback to complete
                        await asyncio.sleep(audio_duration * 0.2)  # Add a safety margin
                        logger.info("Finished playing audio")
                    except Exception as e:
                        logger.error(f"Error playing audio chunks: {e}")

                    logger.info("Audio playback completed")
            elif len(audio_chunks) == 1:
                logger.info("Only first chunk was played")
            else:
                logger.warning("No audio chunks received from TTS service")
        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            logger.info("Speech generation complete")
            self.is_translator_speaking = False

            if (
                not self.audio_buffer.is_processing
                and self.audio_buffer.started_at is not None
                and len(self.audio_buffer.frames) > 10
            ):
                logger.debug("Processing accumulated speech after translation playback")
                self._process_speech_chunk(is_final=False)

    async def _translation_queue_processor(self):
        """Process translation requests from the queue"""
        consecutive_errors = 0
        max_consecutive_errors = 3

        while self._running:
            try:
                # Get next translation task - with timeout to avoid blocking forever
                try:
                    translation_task = await asyncio.wait_for(
                        self.translation_queue.get(), timeout=1.0
                    )
                    # Reset error counter on successful task retrieval
                    consecutive_errors = 0
                except asyncio.TimeoutError:
                    # No items in queue, just continue the loop
                    await asyncio.sleep(0.1)
                    continue

                # Validate task format before processing
                if (
                    not translation_task
                    or not isinstance(translation_task, tuple)
                    or len(translation_task) != 5
                ):
                    logger.warning(
                        f"Invalid translation task format: {type(translation_task)}"
                    )
                    self.translation_queue.task_done()
                    continue

                # Unpack the translation task
                audio_arrays, raw_frames, start_frame, stop_frame, is_final = (
                    translation_task
                )

                # Validate task components
                if not audio_arrays or len(audio_arrays) == 0:
                    logger.warning("Empty audio arrays in translation task")
                    self.translation_queue.task_done()
                    continue

                # Process the translation
                self.active_translation_process = True
                try:
                    await self._process_audio(
                        audio_arrays, raw_frames, start_frame, stop_frame, is_final
                    )
                except Exception as e:
                    logger.error(f"Error processing audio in queue task: {str(e)}")
                    consecutive_errors += 1

                    # If we have too many consecutive errors, clear the queue
                    if consecutive_errors >= max_consecutive_errors:
                        logger.warning(
                            f"Too many consecutive errors ({consecutive_errors}), clearing translation queue"
                        )
                        self._clear_translation_queue()
                        consecutive_errors = 0

            except Exception as e:
                logger.error(f"Translation queue error: {str(e)}")
                consecutive_errors += 1

                # Safety check - if too many errors, reset the queue
                if consecutive_errors >= max_consecutive_errors:
                    logger.warning(
                        "Multiple consecutive queue errors, resetting translation system"
                    )
                    self._clear_translation_queue()
                    consecutive_errors = 0
                    await asyncio.sleep(1.0)  # Brief pause to let system recover
            finally:
                # Ensure we always mark task as done and reset flag
                try:
                    self.translation_queue.task_done()
                except Exception:
                    pass
                self.active_translation_process = False

    def _clear_translation_queue(self):
        """Clear the translation queue in case of errors"""
        try:
            # Empty the queue
            while not self.translation_queue.empty():
                try:
                    self.translation_queue.get_nowait()
                    self.translation_queue.task_done()
                except:
                    pass

            # Reset audio buffer
            self.audio_buffer.is_processing = False
            self.audio_buffer.frames = []
            self.audio_buffer.audio_arrays = []
            self.audio_buffer.started_at = None
            self.audio_buffer.last_updated_at = None
            self.audio_buffer.speech_start_frame = None
            self.audio_buffer.detected_language = None

            logger.info("Translation queue and audio buffer cleared")
        except Exception as e:
            logger.error(f"Error clearing translation queue: {e}")

    def _finalize_speech_buffer(self, speech_stop_frame=None):
        """Process the speech buffer after silence is detected"""
        if not self.audio_buffer.is_processing and len(self.audio_buffer.frames) > 0:
            self._process_speech_chunk(is_final=True, stop_frame=speech_stop_frame)

    def _process_speech_chunk(self, is_final=False, stop_frame=None):
        """Process the current speech buffer"""
        if self.audio_buffer.is_processing or len(self.audio_buffer.frames) == 0:
            return

        self.audio_buffer.is_processing = True

        try:
            # Make a copy of the frames to process
            frames_to_process = self.audio_buffer.frames.copy()
            audio_arrays_to_process = self.audio_buffer.audio_arrays.copy()
            start_frame = self.audio_buffer.speech_start_frame

            # Reset buffer if this is a final chunk
            if is_final:
                self.audio_buffer.frames = []
                self.audio_buffer.audio_arrays = []
                self.audio_buffer.started_at = None
                self.audio_buffer.last_updated_at = None
                self.audio_buffer.speech_start_frame = None
                # Keep detected language for the speaker
            else:
                # Keep a small overlap for context
                overlap_frames = min(5, len(self.audio_buffer.frames))
                self.audio_buffer.frames = self.audio_buffer.frames[-overlap_frames:]
                self.audio_buffer.audio_arrays = self.audio_buffer.audio_arrays[
                    -overlap_frames:
                ]
                if self.audio_buffer.started_at is not None:
                    self.audio_buffer.started_at = time.time() * 1000

            # Use a separate thread to add to the queue to avoid asyncio issues
            def queue_task():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Run the async task in this thread's event loop
                    loop.run_until_complete(
                        self._add_to_translation_queue(
                            audio_arrays_to_process,
                            frames_to_process,
                            start_frame,
                            stop_frame,
                            is_final,
                        )
                    )
                except Exception as e:
                    logger.error(f"Error in queue_task: {str(e)}")
                finally:
                    loop.close()
                    self.audio_buffer.is_processing = False

            # Start the thread
            threading.Thread(target=queue_task, daemon=True).start()

        except Exception as e:
            logger.error(f"Speech chunk processing error: {str(e)}")
            self.audio_buffer.is_processing = False

    async def _add_to_translation_queue(
        self, audio_arrays, frames, start_frame, stop_frame, is_final
    ):
        """Add a translation task to the queue with better queue management"""
        try:
            if self.translation_queue.qsize() >= self.translation_queue.maxsize - 1:
                # If queue is nearly full and this isn't a final chunk, we can skip it
                if not is_final:
                    logger.warning(
                        "Queue almost full, skipping non-final translation task"
                    )
                    return

                # For final chunks, try to make room by removing oldest non-final task
                if not self.translation_queue.empty():
                    try:
                        # Get all items from the queue
                        temp_items = []
                        while not self.translation_queue.empty():
                            item = self.translation_queue.get_nowait()
                            temp_items.append(item)

                        # Filter out one non-final item if possible
                        removed_item = False
                        filtered_items = []
                        for item in temp_items:
                            # Item structure is (audio_arrays, frames, start_frame, stop_frame, is_final)
                            if (
                                not removed_item and not item[4]
                            ):  # if not removed and not final
                                removed_item = True  # Skip this item
                                logger.warning(
                                    "Removed oldest non-final task from queue to make room"
                                )
                            else:
                                filtered_items.append(item)

                        # Put remaining items back in the queue
                        for item in filtered_items:
                            await self.translation_queue.put(item)

                    except Exception as e:
                        logger.error(f"Error managing queue: {e}")

            task = (audio_arrays, frames, start_frame, stop_frame, is_final)

            # Try to add to queue with timeout
            try:
                await asyncio.wait_for(self.translation_queue.put(task), timeout=1.0)
                logger.info(
                    f"Added translation task to queue (final={is_final}, queue size={self.translation_queue.qsize()}/{self.translation_queue.maxsize})"
                )
            except asyncio.TimeoutError:
                if is_final:
                    logger.error(
                        "Failed to add final chunk to queue after timeout - queue might be deadlocked"
                    )
                else:
                    logger.warning(
                        "Queue full after timeout, dropping non-final translation request"
                    )
        except Exception as e:
            logger.error(f"Failed to add to translation queue: {str(e)}")

    async def _process_audio(
        self, audio_arrays, raw_frames, start_frame=None, stop_frame=None, is_final=True
    ):
        """Process audio through the translation pipeline"""
        try:
            # Skip if no frames
            if not audio_arrays or len(audio_arrays) == 0:
                return

            # Combine audio arrays
            audio_np = np.concatenate(audio_arrays)
            duration = len(audio_np) / self.SAMPLE_RATE

            # Skip if too short
            if duration < 0.2:
                return

            # Check for invalid audio data
            if np.isnan(audio_np).any() or np.isinf(audio_np).any():
                return

            # Normalize audio
            max_val = np.max(np.abs(audio_np))
            if max_val > 0:
                audio_np = audio_np / max_val * 0.95
            else:
                return

            # Apply a subtle smoothing filter
            try:
                b, a = signal.butter(2, 0.99)
                audio_np = signal.filtfilt(b, a, audio_np)
            except Exception:
                pass

            # Resample to 16kHz for Whisper
            audio_16k = signal.resample_poly(audio_np, 16000, self.SAMPLE_RATE)
            audio_16k = np.nan_to_num(audio_16k)
            audio_16k = np.clip(audio_16k, -1.0, 1.0)
            audio_int16 = (audio_16k * 32767).astype(np.int16)

            # 1. Speech recognition with WhisperSTTService (with language detection)
            transcription_result = ""
            detected_language = None

            # Start processing
            if start_frame:
                await self.stt_service.start_processing_metrics()
                await self.stt_service.process_frame(
                    start_frame, FrameDirection.UPSTREAM
                )

            # Process audio
            audio_bytes = audio_int16.tobytes()
            try:
                async for result_frame in self.stt_service.run_stt(audio_bytes):
                    logger.info(f"STT result: {result_frame}")
                    if isinstance(result_frame, TranscriptionFrame):
                        transcription_result = result_frame.text
                        # Get detected language from Whisper
                        if hasattr(result_frame, "language") and result_frame.language:
                            detected_language = result_frame.language
                        else:
                            detected_language = "en"
                        break
                    elif isinstance(result_frame, ErrorFrame):
                        logger.error(f"STT error: {result_frame}")
                if stop_frame and is_final:
                    await self.stt_service.process_frame(
                        stop_frame, FrameDirection.DOWNSTREAM
                    )
            except Exception as e:
                logger.error(f"STT processing error: {e}")

            await self.stt_service.stop_processing_metrics()

            # Skip if no transcription
            if not transcription_result or not transcription_result.strip():
                return

            # Store detected language for this speaker if we detected one
            if detected_language and self.audio_buffer.speaker_id:
                self.audio_buffer.detected_language = detected_language
                self.participant_languages[self.audio_buffer.speaker_id] = (
                    detected_language
                )
                logger.info(
                    f"Detected language for participant {self.audio_buffer.speaker_id}: {detected_language}"
                )

            # If we don't have a detected language, use previous detection or default to 'en'
            if not detected_language and self.audio_buffer.speaker_id:
                detected_language = self.participant_languages.get(
                    self.audio_buffer.speaker_id, "en"
                )

            # Skip translation if source language matches target language
            if detected_language == self.target_language:
                logger.info(
                    f"Skipping translation as source language ({detected_language}) matches target language ({self.target_language})"
                )
                return

            # 2. Translate the transcript
            # Add context for continuity
            if self.translation_context and is_final:
                context = " ".join(self.translation_context)
                if len(context) > 0:
                    transcription_result = f"{context} {transcription_result}"

            # Get appropriate translation model
            translation_key = (detected_language, self.target_language)
            if translation_key not in self.translation_models:
                # Check if we need to reverse the model and use it bidirectionally
                reverse_key = (self.target_language, detected_language)
                if reverse_key in self.translation_models:
                    logger.warning(
                        f"Using reverse translation model: {reverse_key} instead of {translation_key}"
                    )
                    translation_key = reverse_key
                else:
                    logger.error(
                        f"No translation model available for {translation_key}"
                    )
                    return

            translator = self.translation_models[translation_key]["pipeline"]
            translation_result = translator(transcription_result, max_length=512)
            translated_text = translation_result[0]["translation_text"]

            # Extract only new content if needed
            if self.translation_context and not is_final:
                last_context = (
                    self.translation_context[-1] if self.translation_context else ""
                )
                if last_context in translated_text:
                    translated_text = translated_text[len(last_context) :].strip()

            # Skip if empty translation
            if not translated_text.strip():
                return

            # Update translation context
            if is_final:
                self.translation_context.append(translated_text)
                if len(self.translation_context) > self.max_context_sentences:
                    self.translation_context = self.translation_context[
                        -self.max_context_sentences :
                    ]

            # 3. Text-to-speech with Cartesia
            await self._generate_and_play_speech(translated_text, detected_language)

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            self.is_translator_speaking = False

    def join(self, url):
        """Join a Daily room"""
        translator_name = f"Translator-{self.target_language}__for__{self.user_id}"

        self.client.join(url)
        self.client.set_user_name(translator_name)

        time.sleep(1)

    def is_running(self):
        """Check if the service is still running"""
        return self._running

    def leave(self):
        """Leave the Daily room"""
        self.client.leave()
