import os
import sys
import json
import time
import wave
import numpy as np
import sounddevice as sd
import argparse
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Generator, Union, Tuple, AsyncGenerator
from dotenv import load_dotenv
import aiohttp
import re
import torch
import psutil
import logging
from cachetools import LRUCache

# --- Configuration & Setup ---

# Basic Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    # Simple check based on common patterns, adjust if needed
    return ("--reload" in sys.argv or 
            os.environ.get("UVICORN_RELOAD", "false").lower() == "true" or
            "uvicorn.workers.UvicornWorker" in str(threading.current_thread()))

# Set a flag to avoid repeat messages ONLY in the main process
_MAIN_PROCESS_STARTED = False
if not is_reloader_process() and not _MAIN_PROCESS_STARTED:
    logger.info("Main process starting...")
    _MAIN_PROCESS_STARTED = True
    IS_MAIN_PROCESS = True
else:
    IS_MAIN_PROCESS = False
    # Suppress logging in reloader processes if desired
    # logging.getLogger().setLevel(logging.WARNING)


# Load environment variables from .env file
load_dotenv()

# --- Hardware Detection ---
HIGH_END_GPU = False
GPU_INFO = "CPU Only"
CPU_INFO = ""
try:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem_gb = props.total_memory / (1024**3)
        compute_capability = f"{props.major}.{props.minor}"
        GPU_INFO = f"CUDA GPU: {gpu_name} ({gpu_mem_gb:.2f} GB, CC {compute_capability})"
        HIGH_END_GPU = (gpu_mem_gb >= 16.0 or 
                        props.major >= 8 or 
                        (gpu_mem_gb >= 12.0 and props.major >= 7))
        if IS_MAIN_PROCESS:
            logger.info(f"🖥️ Hardware: {GPU_INFO}")
            logger.info(f"🚀 Using {'high-performance' if HIGH_END_GPU else 'GPU-optimized'} settings.")
    else:
        raise ImportError("No CUDA GPU found")
except (ImportError, RuntimeError):
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    CPU_INFO = f"CPU: {cpu_cores} cores, {cpu_threads} threads, RAM: {ram_gb:.2f} GB"
    if IS_MAIN_PROCESS:
        logger.info(f"🖥️ Hardware: {CPU_INFO} (No CUDA GPU detected)")
        logger.info("⚙️ Using CPU-optimized settings")

# --- Environment Variable Loading & Validation ---
def get_env_var(name, default=None, required=False, var_type=str):
    value = os.environ.get(name, default)
    if required and value is None:
        logger.error(f"Missing required environment variable: {name}")
        raise ValueError(f"Missing required environment variable: {name}")
    if value is not None:
        try:
            return var_type(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid type for {name}. Expected {var_type.__name__}, got '{value}'. Using default: {default}")
            if required:
                 # If required and type is wrong, use default if available, else raise error
                 if default is not None:
                     return default
                 else:
                     raise ValueError(f"Invalid type for required variable {name} and no default provided.")
            return default
    return value # Return None if not found and not required

# Critical settings
API_URL = get_env_var("ORPHEUS_API_URL", required=True)
MODEL_NAME = get_env_var("ORPHEUS_MODEL_NAME", "default-model-name") # Provide a default or handle if missing

# API connection settings
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"} # Add Accept
REQUEST_TIMEOUT = get_env_var("ORPHEUS_API_TIMEOUT", 120, var_type=int)
CONNECT_TIMEOUT = 10 # Separate connect timeout

# Model generation parameters
MAX_TOKENS = get_env_var("ORPHEUS_MAX_TOKENS", 8192, var_type=int)
TEMPERATURE = get_env_var("ORPHEUS_TEMPERATURE", 0.6, var_type=float)
TOP_P = get_env_var("ORPHEUS_TOP_P", 0.9, var_type=float)
REPETITION_PENALTY = 1.1 # Keep fixed as per original code comment
SAMPLE_RATE = get_env_var("ORPHEUS_SAMPLE_RATE", 24000, var_type=int)

# Parallel processing settings - Adjust based on testing
# More workers might not always be faster due to API limits or GIL
NUM_WORKERS = min(os.cpu_count() or 1, 8) # Cap workers, adjust based on CPU/IO bound nature

# Cache Settings
ENABLE_CACHE = get_env_var("ENABLE_CACHE", "true", var_type=str).lower() == "true"
MAX_CACHE_SIZE = get_env_var("MAX_CACHE_SIZE", 50, var_type=int)

if IS_MAIN_PROCESS:
    logger.info("Configuration loaded:")
    logger.info(f"  API_URL: {API_URL}")
    logger.info(f"  MODEL_NAME: {MODEL_NAME}")
    logger.info(f"  MAX_TOKENS: {MAX_TOKENS}, TEMP: {TEMPERATURE}, TOP_P: {TOP_P}")
    logger.info(f"  SAMPLE_RATE: {SAMPLE_RATE}")
    logger.info(f"  TIMEOUT: {REQUEST_TIMEOUT}s, WORKERS: {NUM_WORKERS}")
    logger.info(f"  CACHE ENABLED: {ENABLE_CACHE}, CACHE SIZE: {MAX_CACHE_SIZE}")

# --- Voices & Language ---
# (Keep the voice definitions as they are)
ENGLISH_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
FRENCH_VOICES = ["pierre", "amelie", "marie"]
GERMAN_VOICES = ["jana", "thomas", "max"]
KOREAN_VOICES = ["유나", "준서"]
HINDI_VOICES = ["ऋतिका"]
MANDARIN_VOICES = ["长乐", "白芷"]
SPANISH_VOICES = ["javi", "sergio", "maria"]
ITALIAN_VOICES = ["pietro", "giulia", "carlo"]
AVAILABLE_VOICES = (ENGLISH_VOICES + FRENCH_VOICES + GERMAN_VOICES + KOREAN_VOICES +
                   HINDI_VOICES + MANDARIN_VOICES + SPANISH_VOICES + ITALIAN_VOICES)
DEFAULT_VOICE = "tara"
VOICE_TO_LANGUAGE = {voice: lang for voices, lang in [
    (ENGLISH_VOICES, "english"), (FRENCH_VOICES, "french"), (GERMAN_VOICES, "german"),
    (KOREAN_VOICES, "korean"), (HINDI_VOICES, "hindi"), (MANDARIN_VOICES, "mandarin"),
    (SPANISH_VOICES, "spanish"), (ITALIAN_VOICES, "italian")] for voice in voices}
AVAILABLE_LANGUAGES = list(set(VOICE_TO_LANGUAGE.values()))


# --- SpeechPipe Integration (Assuming these exist) ---
try:
    # Attempt to import the necessary functions from the local module
    from .speechpipe import turn_token_into_id, CUSTOM_TOKEN_PREFIX, convert_to_audio as speechpipe_convert_to_audio
    SPEECHPIPE_AVAILABLE = True
except ImportError:
    logger.error("Failed to import from .speechpipe. Ensure speechpipe.py is in the same directory.")
    # Define dummy functions to allow the script to run but fail gracefully later
    def turn_token_into_id(token_text, count): return None
    def speechpipe_convert_to_audio(multiframe, count): return None
    CUSTOM_TOKEN_PREFIX = "<?>_" # Dummy value
    SPEECHPIPE_AVAILABLE = False

# --- Performance Monitoring ---
class PerformanceMonitor:
    def __init__(self, description="Operation"):
        self.description = description
        self.start_time = time.monotonic()
        self.token_count = 0
        self.audio_chunks = 0
        self.audio_bytes = 0
        self.last_report_time = self.start_time
        self.report_interval = 5.0 # Report less frequently

    def add_tokens(self, count: int = 1):
        self.token_count += count
        self._check_report()

    def add_audio_chunk(self, audio_bytes: int):
        if audio_bytes > 0:
            self.audio_chunks += 1
            self.audio_bytes += audio_bytes
            self._check_report()

    def _check_report(self):
        current_time = time.monotonic()
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time

    def report(self, final=False):
        elapsed = time.monotonic() - self.start_time
        if elapsed < 0.001: return

        tokens_per_sec = self.token_count / elapsed
        chunks_per_sec = self.audio_chunks / elapsed
        bytes_per_sec = self.audio_bytes / elapsed
        est_duration_sec = self.audio_bytes / (SAMPLE_RATE * 2) # Assuming 16-bit mono

        status = "Final" if final else "Progress"
        logger.info(
            f"{self.description} {status}: "
            f"{tokens_per_sec:.1f} tk/s, "
            f"{chunks_per_sec:.1f} audio chunks/s, "
            f"{bytes_per_sec/1024:.1f} KB/s. "
            f"Total: {self.token_count} tk, {self.audio_chunks} chunks, "
            f"{est_duration_sec:.2f}s audio in {elapsed:.2f}s"
        )

    def __enter__(self):
        self.start_time = time.monotonic()
        logger.info(f"Starting: {self.description}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.report(final=True)
        if exc_type:
             logger.error(f"Error during {self.description}: {exc_val}")
        # Optional: return False to propagate exception, True to suppress


# --- Core Logic ---

def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Formats prompt for the Orpheus API."""
    if voice not in AVAILABLE_VOICES:
        logger.warning(f"Voice '{voice}' not recognized. Using default '{DEFAULT_VOICE}'.")
        voice = DEFAULT_VOICE

    # Assuming the API expects this format based on original code
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    # Note: The format "{voice}: {prompt}" might be specific to older models or internal conventions.
    # Check API documentation for the correct format. Let's assume this is correct for now.
    formatted_prompt = f"{voice}: {prompt}"
    return f"{special_start}{formatted_prompt}{special_end}"

# --- Async API Client ---
# Use a single session for connection pooling
aiohttp_session = None

async def get_aiohttp_session():
    global aiohttp_session
    if aiohttp_session is None or aiohttp_session.closed:
        # Increased connector limits for higher concurrency
        connector = aiohttp.TCPConnector(limit_per_host=50, limit=100, ssl=False) # Adjust limits based on API
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
        aiohttp_session = aiohttp.ClientSession(headers=HEADERS, timeout=timeout, connector=connector)
        logger.debug("Created new aiohttp ClientSession")
    return aiohttp_session

async def close_aiohttp_session():
    global aiohttp_session
    if aiohttp_session and not aiohttp_session.closed:
        await aiohttp_session.close()
        aiohttp_session = None
        logger.debug("Closed aiohttp ClientSession")

async def generate_tokens_from_api_async(
    prompt: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY
) -> AsyncGenerator[str, None]:
    """Generate tokens asynchronously using aiohttp."""
    if not SPEECHPIPE_AVAILABLE:
         logger.error("SpeechPipe components not loaded. Cannot generate tokens.")
         return
         
    formatted_prompt = format_prompt(prompt, voice)
    payload = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True,
        "model": MODEL_NAME,
        "use_cache": True, # Keep KV cache enabled
        # "stop": ["<|eot_id|>", "<|endoftext|>"] # Add common stop tokens if needed
    }

    # Add GPU optimization hints if applicable (ensure API supports these)
    if torch.cuda.is_available():
        # These parameters might be specific to certain backends (like vLLM)
        # Verify if your Orpheus-FASTAPI endpoint supports them
        # payload.update({
        #     "gpu_memory_utilization": 0.9, # Example: Use 90% of GPU memory
        # })
        # if HIGH_END_GPU:
        #     payload.update({
        #          # Example parameters, check API docs
        #         "tensor_parallel_size": torch.cuda.device_count(),
        #     })
        pass # Keep it clean unless parameters are confirmed

    retries = 3
    session = await get_aiohttp_session()
    perf_monitor = PerformanceMonitor(f"Token Generation for '{prompt[:30]}...'")

    for attempt in range(retries):
        try:
            async with session.post(API_URL, json=payload) as response:
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                buffer = ""
                async for line_bytes in response.content.iter_any(): # Read available data efficiently
                    if not line_bytes: continue # Skip empty chunks

                    buffer += line_bytes.decode('utf-8', errors='ignore')
                    
                    # Process lines efficiently
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        
                        if line.startswith('data: '):
                            data_str = line[6:].strip()
                            if not data_str: continue

                            if data_str == '[DONE]':
                                perf_monitor.report(final=True)
                                return # End of stream

                            try:
                                data = json.loads(data_str)
                                choices = data.get('choices', [])
                                if choices:
                                    token_text = choices[0].get('text', '')
                                    if token_text:
                                        # Yield tokens as they arrive
                                        # Splitting by '>' might be specific to the model's tokenization?
                                        # Consider yielding the whole token_text if splitting isn't necessary
                                        # Or use a more robust tokenizer if needed
                                        token_parts = token_text.split('>') # Original logic
                                        for part in token_parts:
                                            if part:
                                                yielded_token = f'{part}>'
                                                perf_monitor.add_tokens()
                                                yield yielded_token
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode JSON line: {data_str}")
                                continue # Skip malformed lines
                        elif line.strip(): # Log unexpected lines
                            logger.debug(f"Received non-data line: {line}")
                
                # Process any remaining buffer content after stream ends
                if buffer.startswith('data: '):
                     # Handle potential last line without newline
                     # (Duplicate code from loop, could be refactored)
                    data_str = buffer[6:].strip()
                    if data_str and data_str != '[DONE]':
                         try:
                             data = json.loads(data_str)
                             choices = data.get('choices', [])
                             if choices:
                                 token_text = choices[0].get('text', '')
                                 if token_text:
                                     token_parts = token_text.split('>')
                                     for part in token_parts:
                                         if part:
                                             yielded_token = f'{part}>'
                                             perf_monitor.add_tokens()
                                             yield yielded_token
                         except json.JSONDecodeError:
                             logger.warning(f"Failed to decode final JSON line: {data_str}")

                return # Successful stream completion

        except aiohttp.ClientResponseError as e:
            logger.error(f"API Error: Status {e.status}, Message: {e.message}. Attempt {attempt + 1}/{retries}")
            if e.status >= 500 and attempt < retries - 1:
                await asyncio.sleep(2 ** attempt) # Exponential backoff
            else:
                raise  # Non-retriable error or max retries reached
        except aiohttp.ClientConnectionError as e:
            logger.error(f"Connection Error: {e}. Attempt {attempt + 1}/{retries}")
            if attempt < retries - 1:
                await close_aiohttp_session() # Force reconnect
                await asyncio.sleep(2 ** attempt)
            else:
                raise
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {REQUEST_TIMEOUT}s. Attempt {attempt + 1}/{retries}")
            if attempt < retries - 1:
                await asyncio.sleep(1) # Shorter delay for timeout
            else:
                raise
        except Exception as e:
            logger.exception(f"Unexpected error during token generation: {e}. Attempt {attempt + 1}/{retries}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise
    
    # If loop finishes without returning/raising, something went wrong
    logger.error("Token generation failed after multiple retries.")
    raise RuntimeError("Failed to generate tokens from API after multiple retries.")


# --- Optimized Audio Decoding ---
# Using ThreadPoolExecutor to run blocking speechpipe calls off the main async loop
audio_executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

def convert_token_id_batch(token_texts: List[str], start_count: int) -> List[Optional[int]]:
    """Converts a batch of token texts to IDs."""
    # This could be parallelized further if turn_token_into_id is CPU bound
    # For now, run it sequentially within the executor thread
    results = []
    for i, text in enumerate(token_texts):
        # Need error handling inside turn_token_into_id or here
        token_id = turn_token_into_id(text, start_count + i)
        results.append(token_id)
    return results

def convert_to_audio_batch(multiframe: List[int], count: int) -> Optional[bytes]:
    """Wrapper for speechpipe audio conversion with performance monitoring."""
    if not SPEECHPIPE_AVAILABLE or not multiframe:
        return None
    # logger.debug(f"Converting {len(multiframe)} frames to audio (count: {count})")
    # Assuming speechpipe_convert_to_audio is blocking (CPU/IO bound)
    return speechpipe_convert_to_audio(multiframe, count)

async def process_tokens_to_audio_optimized(
    token_gen: AsyncGenerator[str, None],
    perf_monitor: PerformanceMonitor
) -> AsyncGenerator[bytes, None]:
    """
    Processes token stream, batches conversions, and yields audio chunks asynchronously.
    """
    if not SPEECHPIPE_AVAILABLE:
         logger.error("SpeechPipe components not loaded. Cannot process audio.")
         return

    loop = asyncio.get_running_loop()
    token_buffer: List[str] = []
    id_buffer: List[int] = []
    count = 0
    
    # Tuning parameters for batching - Adjust based on performance testing
    TOKEN_BATCH_SIZE = 256 # How many tokens to convert to IDs at once
    FRAME_BATCH_SIZE = 28 * 4 # How many frames (IDs) to convert to audio at once (Multiple of original 28)
    MIN_FRAMES_FIRST = 7 * 2 # Generate first chunk faster
    
    first_chunk_yielded = False

    async for token_text in token_gen:
        token_buffer.append(token_text)

        # Convert tokens to IDs in batches using the executor
        if len(token_buffer) >= TOKEN_BATCH_SIZE:
            batch_to_process = token_buffer
            token_buffer = []
            
            try:
                # Run blocking ID conversion in thread pool
                new_ids = await loop.run_in_executor(
                    audio_executor,
                    convert_token_id_batch,
                    batch_to_process,
                    count
                )
                valid_ids = [tid for tid in new_ids if tid is not None and tid > 0] # Filter out invalid IDs
                id_buffer.extend(valid_ids)
                count += len(batch_to_process) # Increment count by processed tokens
            except Exception as e:
                 logger.error(f"Error converting token batch to IDs: {e}")
                 # Decide how to handle: continue, raise, etc.
                 continue # Skip this batch

        # Convert IDs to audio chunks when enough frames are buffered
        process_threshold = MIN_FRAMES_FIRST if not first_chunk_yielded else FRAME_BATCH_SIZE
        
        while len(id_buffer) >= process_threshold:
            frames_to_convert = id_buffer[:process_threshold]
            id_buffer = id_buffer[process_threshold:] # Consume the processed frames

            try:
                 # Run blocking audio conversion in thread pool
                audio_data = await loop.run_in_executor(
                    audio_executor,
                    convert_to_audio_batch,
                    frames_to_convert,
                    count # Pass current count (or adjusted count?) - Check speechpipe needs
                )

                if audio_data:
                    perf_monitor.add_audio_chunk(len(audio_data))
                    yield audio_data
                    if not first_chunk_yielded:
                         first_chunk_yielded = True
                         process_threshold = FRAME_BATCH_SIZE # Switch to larger batches
            except Exception as e:
                 logger.error(f"Error converting frame batch to audio: {e}")
                 continue # Skip this batch

    # Process any remaining tokens
    if token_buffer:
        try:
            new_ids = await loop.run_in_executor(
                audio_executor, convert_token_id_batch, token_buffer, count
            )
            valid_ids = [tid for tid in new_ids if tid is not None and tid > 0]
            id_buffer.extend(valid_ids)
            count += len(token_buffer)
        except Exception as e:
             logger.error(f"Error converting final token batch to IDs: {e}")

    # Process any remaining frames
    while id_buffer:
        frames_to_convert = id_buffer[:FRAME_BATCH_SIZE]
        id_buffer = id_buffer[FRAME_BATCH_SIZE:]
        if not frames_to_convert: break
        try:
            audio_data = await loop.run_in_executor(
                audio_executor, convert_to_audio_batch, frames_to_convert, count
            )
            if audio_data:
                perf_monitor.add_audio_chunk(len(audio_data))
                yield audio_data
        except Exception as e:
            logger.error(f"Error converting final frame batch to audio: {e}")
            # Decide if necessary to break or continue
            break # Stop processing if final conversion fails

# --- Audio Playback ---
# Reusable buffer cache to reduce allocations
class AudioBufferCache:
    def __init__(self):
        self._local = threading.local()

    def _get_buffer(self, size):
        if not hasattr(self._local, 'float_buffer') or self._local.float_buffer.size != size:
            # logger.debug(f"Allocating new float32 buffer of size {size}")
            self._local.float_buffer = np.empty(size, dtype=np.float32)
        return self._local.float_buffer

    def get_float_array(self, audio_bytes: bytes) -> np.ndarray:
        if not audio_bytes:
            return np.array([], dtype=np.float32)
        
        # Assuming int16 input
        required_size = len(audio_bytes) // 2
        float_buffer = self._get_buffer(required_size)
        
        # Use frombuffer with offset=0 and count=-1 (default)
        int16_view = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Vectorized conversion
        np.multiply(int16_view, 1.0 / 32767.0, out=float_buffer)
        return float_buffer

# Global cache instance
audio_buffer_cache = AudioBufferCache()
playback_queue = asyncio.Queue(maxsize=100) # Queue for audio chunks to be played
playback_active = asyncio.Event() # Signal when playback is happening

async def audio_playback_worker():
    """Dedicated worker to play audio chunks from the queue."""
    stream = None
    try:
        # Configure sounddevice stream
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=2048 # Adjust blocksize based on performance
        )
        stream.start()
        logger.info("Audio playback stream started.")
        playback_active.set()

        while True:
            try:
                 # Wait for the next chunk with a timeout
                 audio_chunk_bytes = await asyncio.wait_for(playback_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                 # If queue is empty and the main task might be done, check event
                 if not playback_active.is_set() and playback_queue.empty():
                      logger.debug("Playback worker timeout and queue empty, exiting.")
                      break
                 continue # Continue waiting if playback is supposed to be active


            if audio_chunk_bytes is None:  # Sentinel value indicates end of stream
                logger.debug("Received None sentinel, stopping playback worker.")
                playback_active.clear() # Signal main task we are stopping
                break

            if audio_chunk_bytes:
                try:
                    # Convert bytes to float32 numpy array using cache
                    audio_float = audio_buffer_cache.get_float_array(audio_chunk_bytes)
                    # logger.debug(f"Playing audio chunk of size {len(audio_float)} samples.")
                    stream.write(audio_float)
                except Exception as e:
                    logger.error(f"Error writing audio to stream: {e}")
                    # Optionally break or continue based on error severity
            
            playback_queue.task_done() # Mark item as processed

    except sd.PortAudioError as e:
         logger.error(f"Sounddevice PortAudioError: {e}")
         # Handle specific audio device errors if necessary
    except Exception as e:
        logger.exception(f"Error in audio playback worker: {e}")
    finally:
        if stream:
            try:
                # Wait for buffer to clear before closing? Optional.
                # time.sleep(stream.latency[1])
                if not stream.stopped:
                     stream.stop()
                if not stream.closed:
                     stream.close()
                logger.info("Audio playback stream stopped and closed.")
            except Exception as e:
                 logger.error(f"Error closing audio stream: {e}")
        playback_active.clear() # Ensure event is cleared on exit

async def stream_audio_async(audio_chunk_generator: AsyncGenerator[bytes, None]):
    """Consumes audio chunks and puts them onto the playback queue."""
    playback_task = None
    try:
        # Start the playback worker task
        playback_task = asyncio.create_task(audio_playback_worker())
        await playback_active.wait() # Wait for worker to signal it's ready

        async for audio_chunk in audio_chunk_generator:
            await playback_queue.put(audio_chunk)

        # Signal end of audio stream
        await playback_queue.put(None)
        
        # Wait for queue to be fully processed
        await playback_queue.join()
        
        # Wait for the playback task to finish gracefully
        if playback_task:
             await asyncio.wait_for(playback_task, timeout=10.0) # Wait with timeout

    except asyncio.CancelledError:
         logger.info("Audio streaming cancelled.")
    except Exception as e:
         logger.exception(f"Error during audio streaming: {e}")
    finally:
         playback_active.clear() # Ensure event is cleared
         # Ensure task is cancelled if still running
         if playback_task and not playback_task.done():
             playback_task.cancel()
             try:
                 await playback_task
             except asyncio.CancelledError:
                 pass # Expected cancellation
             except Exception as e:
                 logger.error(f"Error during playback task finalization: {e}")
         logger.info("Audio streaming finished.")

# --- Batch Generation & File Handling ---

def split_text_into_sentences_optimized(text: str) -> List[str]:
    """Split text into sentences using a more robust regex approach."""
    if not text: return []
    # Improved regex handles common abbreviations, direct speech, etc.
    # Breaks after sentence-ending punctuation followed by space/newline,
    # unless preceded by common title abbreviations or part of ellipsis.
    # It's not perfect but better than simple splitting.
    boundaries = re.compile(r"""
        (?<!\w\.\w.)         # Avoid splitting abbreviations like U.S.A.
        (?<![A-Z][a-z]\.)    # Avoid splitting titles like Mr., Ms.
        (?<![A-Z]\.)         # Avoid splitting single letter abbr. like A. Smith
        (?<=[.?!])           # Must end with sentence punctuation
        (?![\'"`’]?\s*$)      # Avoid splitting if followed by end of string
        (?!\.\.)             # Avoid splitting ellipsis
        [\s\n]+              # Match the space/newline after punctuation
        (?=[A-Z"“‘])         # Lookahead for capital letter or quote
        """, re.VERBOSE)
    
    sentences = boundaries.split(text)
    
    # Filter out empty strings and strip whitespace
    filtered_sentences = [s.strip() for s in sentences if s and s.strip()]
    
    # Combine very short sentences to avoid excessive fragmentation
    min_len = 30 # Minimum characters for a standalone sentence fragment
    combined = []
    buffer = ""
    for sentence in filtered_sentences:
        if not buffer:
            buffer = sentence
        elif len(buffer) < min_len or len(sentence) < min_len:
            buffer += " " + sentence
        else:
            combined.append(buffer)
            buffer = sentence
    if buffer:
        combined.append(buffer)
        
    return combined

# Use ProcessPoolExecutor for CPU-bound batch processing
# Ensure objects passed between processes are pickleable
batch_executor = None

def get_batch_executor():
    global batch_executor
    if batch_executor is None:
        # More careful worker count selection
        max_workers = min(NUM_WORKERS, psutil.cpu_count(logical=False) or 2) # Use physical cores
        batch_executor = ProcessPoolExecutor(max_workers=max_workers)
        logger.info(f"Initialized ProcessPoolExecutor with {max_workers} workers.")
    return batch_executor

def shutdown_batch_executor():
     global batch_executor
     if batch_executor:
          logger.info("Shutting down ProcessPoolExecutor...")
          batch_executor.shutdown(wait=True)
          batch_executor = None
          logger.info("ProcessPoolExecutor shut down.")


# This function needs to be defined at the top level to be pickleable
def process_single_batch_sync(batch_text: str, voice: str, temp: float, top_p_val: float, max_tok: int, rep_pen: float, output_dir: str, batch_idx: int) -> Optional[str]:
    """Synchronous function to generate audio for one batch, designed for ProcessPoolExecutor."""
    # This runs in a separate process, needs its own setup if complex dependencies are involved
    # For now, assume basic imports and speechpipe are available via inheritance or path
    
    # Re-import necessary components if they don't survive pickling or are module-level
    from .speechpipe import turn_token_into_id, speechpipe_convert_to_audio
    import wave
    import os
    import time
    import json
    import requests # Using requests sync here as it's simpler in a separate process

    local_logger = logging.getLogger(f"BatchWorker_{batch_idx}")
    local_logger.setLevel(logging.INFO) # Or inherit level
    
    if not batch_text:
        local_logger.warning("Received empty batch text, skipping.")
        return None

    temp_filename = os.path.join(output_dir, f"batch_{batch_idx}_{int(time.time()*1000)}.wav")
    
    formatted_prompt = format_prompt(batch_text, voice) # Reuse formatting logic
    
    payload = {
        "prompt": formatted_prompt, "max_tokens": max_tok, "temperature": temp,
        "top_p": top_p_val, "repeat_penalty": rep_pen, "stream": False, # Get full response for batch
        "model": MODEL_NAME, "use_cache": True,
    }
    
    # Simplified API call within the worker process
    audio_data = b""
    try:
        # Using requests for simplicity within the process
        # Consider a shared async client or queue if performance critical and complex
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # --- THIS PART IS HIGHLY DEPENDENT ON THE *NON-STREAMING* API RESPONSE ---
        # Assuming the non-streaming API returns the full audio directly or token IDs
        # Option A: API returns audio directly (e.g., base64 encoded or raw bytes)
        content_type = response.headers.get('Content-Type', '')
        if 'audio' in content_type:
             audio_data = response.content
             local_logger.info(f"Received {len(audio_data)} bytes of audio directly for batch {batch_idx}.")
        else:
             # Option B: API returns token IDs (less likely for non-streaming TTS)
             # We would need to decode these IDs using speechpipe here
             result = response.json()
             token_ids = result.get("token_ids") # Fictional field, adjust based on actual API
             if token_ids and isinstance(token_ids, list):
                  local_logger.info(f"Received {len(token_ids)} token IDs for batch {batch_idx}. Converting to audio...")
                  # Note: speechpipe_convert_to_audio might expect specific counts/structures
                  # This requires careful adaptation based on speechpipe's API
                  audio_data = speechpipe_convert_to_audio(token_ids, len(token_ids)) # Simplified call
                  if not audio_data:
                       local_logger.warning(f"Speechpipe conversion failed for batch {batch_idx}")
                       
             else:
                 local_logger.error(f"Unexpected non-streaming API response for batch {batch_idx}: {result}")
                 return None # Failed to get audio


    except requests.exceptions.RequestException as e:
        local_logger.error(f"API request failed for batch {batch_idx}: {e}")
        return None
    except Exception as e:
        local_logger.exception(f"Error processing batch {batch_idx}: {e}")
        return None

    if not audio_data:
         local_logger.warning(f"No audio data generated for batch {batch_idx}")
         return None

    # Save the audio chunk to a temporary file
    try:
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # Assuming 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        local_logger.info(f"Saved batch {batch_idx} audio to {temp_filename}")
        return temp_filename
    except Exception as e:
        local_logger.error(f"Failed to save batch {batch_idx} to {temp_filename}: {e}")
        # Clean up failed file?
        if os.path.exists(temp_filename): os.remove(temp_filename)
        return None

def stitch_wav_files_optimized(input_files: List[str], output_file: str):
    """Stitches multiple WAV files into one, optimizing I/O."""
    if not input_files:
        logger.warning("No input files provided for stitching.")
        return
        
    valid_files = [f for f in input_files if f and os.path.exists(f) and os.path.getsize(f) > 44] # Basic WAV header check
    if not valid_files:
        logger.error(f"No valid temporary WAV files found to stitch into {output_file}.")
        return

    logger.info(f"Stitching {len(valid_files)} temporary WAV files into {output_file}...")
    
    try:
        # Get parameters from the first valid file
        with wave.open(valid_files[0], 'rb') as wf_in:
            params = wf_in.getparams()

        # Write to the final output file
        with wave.open(output_file, 'wb') as wf_out:
            wf_out.setparams(params)
            
            # Read and write in larger chunks
            chunk_size = 1024 * 256 # 256 KB chunks
            total_frames = 0
            
            for i, filename in enumerate(valid_files):
                try:
                    with wave.open(filename, 'rb') as wf_in:
                        # Verify compatibility (optional but recommended)
                        if wf_in.getparams()[:3] != params[:3]: # Check channels, sampwidth, framerate
                            logger.warning(f"Skipping incompatible file {filename}: Params {wf_in.getparams()} != {params}")
                            continue
                            
                        while True:
                            frames = wf_in.readframes(chunk_size)
                            if not frames:
                                break
                            wf_out.writeframes(frames)
                            total_frames += len(frames) // wf_in.getsampwidth() // wf_in.getnchannels()
                            
                except wave.Error as e:
                     logger.error(f"Error reading temporary file {filename}: {e}. Skipping.")
                except Exception as e:
                     logger.exception(f"Unexpected error processing {filename}: {e}. Skipping.")
        
        logger.info(f"Successfully stitched {total_frames} frames to {output_file}")

    except wave.Error as e:
        logger.error(f"Error opening or writing WAV file {output_file}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during WAV stitching: {e}")
    finally:
        # Clean up temporary files
        logger.info(f"Cleaning up {len(valid_files)} temporary files...")
        cleaned_count = 0
        for filename in valid_files:
            try:
                os.remove(filename)
                cleaned_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {filename}: {e}")
        logger.info(f"Cleaned up {cleaned_count}/{len(valid_files)} temporary files.")


# --- Main Generation Orchestration ---
# Simple LRU Cache for full audio generation results (if enabled)
audio_cache = LRUCache(maxsize=MAX_CACHE_SIZE)

async def generate_speech_streaming(
    prompt: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY
) -> AsyncGenerator[bytes, None]:
    """Generates speech and yields audio chunks asynchronously for streaming playback."""
    if not SPEECHPIPE_AVAILABLE:
         logger.error("SpeechPipe components not loaded. Cannot generate speech.")
         # Yield empty generator?
         if False: yield
         return

    perf_monitor = PerformanceMonitor(f"Streaming Speech for '{prompt[:30]}...'")
    with perf_monitor:
        token_generator = generate_tokens_from_api_async(
            prompt=prompt, voice=voice, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, repetition_penalty=repetition_penalty
        )
        
        audio_chunk_generator = process_tokens_to_audio_optimized(token_generator, perf_monitor)

        async for audio_chunk in audio_chunk_generator:
            yield audio_chunk


async def generate_speech_to_file(
    prompt: str,
    output_file: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
    use_batching: bool = True,
    max_batch_chars: int = 800 # Smaller batches might be better for parallel processing
):
    """Generates speech and saves it to a file, using batching for long prompts."""
    cache_key = (prompt, voice, temperature, top_p, max_tokens, repetition_penalty, MODEL_NAME)
    if ENABLE_CACHE and cache_key in audio_cache:
        logger.info("Cache hit! Returning cached audio data.")
        # How to handle cached data? Assume it's the final bytes. Needs saving.
        cached_audio_bytes = audio_cache[cache_key]
        try:
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(cached_audio_bytes)
            logger.info(f"Saved cached audio to {output_file}")
            return # Return early
        except Exception as e:
            logger.error(f"Failed to write cached audio to file: {e}")
            # Proceed with generation if saving cache fails

    perf_monitor = PerformanceMonitor(f"File Speech Generation for '{prompt[:30]}...'")
    with perf_monitor:
        # Decide on batching
        should_batch = use_batching and len(prompt) > max_batch_chars * 1.5 # Only batch significantly longer texts

        if not should_batch:
            logger.info("Processing text as a single request (streaming to collect)...")
            all_audio_segments = []
            try:
                async for audio_chunk in generate_speech_streaming(
                        prompt=prompt, voice=voice, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, repetition_penalty=repetition_penalty):
                    all_audio_segments.append(audio_chunk)
                
                if not all_audio_segments:
                     logger.error("No audio segments generated.")
                     return
                
                final_audio = b"".join(all_audio_segments)
                
                # Save to file
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    audio_executor, # Use thread pool for sync file I/O
                    save_audio_sync,
                    final_audio,
                    output_file
                )
                
                # Add to cache if enabled
                if ENABLE_CACHE:
                     logger.info(f"Adding result to cache (key: {str(cache_key)[:100]}...).")
                     audio_cache[cache_key] = final_audio # Store combined bytes
            
            except Exception as e:
                logger.exception("Error during single request generation:")


        else:
            logger.info(f"Text length ({len(prompt)}) exceeds threshold. Using parallel batch processing.")
            sentences = split_text_into_sentences_optimized(prompt)
            if not sentences:
                 logger.error("Text splitting resulted in no sentences.")
                 return

            batches = create_balanced_batches(sentences, max_batch_chars)
            logger.info(f"Split into {len(batches)} batches.")

            temp_dir = os.path.join(os.path.dirname(output_file) or '.', "orpheus_temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            futures = []
            executor = get_batch_executor()
            loop = asyncio.get_running_loop()

            for i, batch_text in enumerate(batches):
                # Submit CPU-bound task to ProcessPoolExecutor from the async loop
                future = loop.run_in_executor(
                    executor,
                    process_single_batch_sync,
                    batch_text, voice, temperature, top_p, max_tokens, repetition_penalty,
                    temp_dir, i
                )
                futures.append(future)

            temp_files = []
            for future in as_completed(futures): # Process as they complete
                try:
                    result_file = await future # Get result from the future
                    if result_file:
                        temp_files.append(result_file)
                    else:
                         logger.warning("A batch process returned no file.")
                except Exception as e:
                    logger.error(f"Error processing a batch future: {e}")

            # Sort temp files (optional, depends if process_single_batch_sync returns filenames reflecting order)
            # temp_files.sort() # Might need numerical sort based on filename pattern

            # Stitch the temporary files together (in a thread pool to not block async loop)
            if temp_files:
                await loop.run_in_executor(
                    audio_executor,
                    stitch_wav_files_optimized,
                    temp_files,
                    output_file
                )
                # Cleanup handled within stitch function now
            else:
                logger.error("No temporary audio files were generated by batch processing.")
                
            # Note: Caching for batched results is more complex.
            # Could cache the final stitched file, but the key needs to be reliable.


def save_audio_sync(audio_bytes: bytes, output_file: str):
     """Synchronous function to save audio bytes to a WAV file."""
     logger.info(f"Saving {len(audio_bytes)} bytes of audio to {output_file}...")
     try:
         with wave.open(output_file, 'wb') as wf:
             wf.setnchannels(1)
             wf.setsampwidth(2)
             wf.setframerate(SAMPLE_RATE)
             wf.writeframes(audio_bytes)
         logger.info(f"Successfully saved audio to {output_file}")
     except Exception as e:
         logger.error(f"Failed to save audio to {output_file}: {e}")


def create_balanced_batches(sentences: List[str], max_chars: int) -> List[str]:
    """Creates batches of sentences trying to stay close to max_chars."""
    batches = []
    current_batch = []
    current_len = 0
    target_min = max_chars * 0.7 # Aim for at least 70% fill

    for sentence in sentences:
        sent_len = len(sentence)
        if sent_len > max_chars:
            # Sentence is too long, split it further (simple split for now)
            # A more sophisticated split preserving words might be needed
            parts = [sentence[i:i+max_chars] for i in range(0, sent_len, max_chars)]
            for part in parts:
                # If current batch is not empty, finish it
                if current_batch:
                    batches.append(" ".join(current_batch))
                    current_batch = []
                    current_len = 0
                # Add the long part as its own batch
                batches.append(part)
            continue # Move to next sentence

        # Check if adding the sentence exceeds the limit
        if current_len + (1 if current_batch else 0) + sent_len > max_chars:
             # If the current batch is reasonably full or adding sentence overflows
             if current_len >= target_min or not current_batch:
                 if current_batch: # Don't add empty batch
                      batches.append(" ".join(current_batch))
                 current_batch = [sentence]
                 current_len = sent_len
             else: # Current batch is small, but adding overflows -> add sentence to it anyway? Or start new?
                  # Starting new is safer to not exceed max_chars significantly
                  batches.append(" ".join(current_batch))
                  current_batch = [sentence]
                  current_len = sent_len
        else:
            # Add sentence to the current batch
            current_batch.append(sentence)
            current_len += (1 if current_batch else 0) + sent_len

    # Add the last batch if it exists
    if current_batch:
        batches.append(" ".join(current_batch))

    return batches

# --- CLI and Main Execution ---

def list_available_voices():
    """Lists available voices."""
    print("Available voices:")
    for lang, voices in [
        ("English", ENGLISH_VOICES), ("French", FRENCH_VOICES), ("German", GERMAN_VOICES),
        ("Korean", KOREAN_VOICES), ("Hindi", HINDI_VOICES), ("Mandarin", MANDARIN_VOICES),
        ("Spanish", SPANISH_VOICES), ("Italian", ITALIAN_VOICES)]:
        print(f"  {lang}: {', '.join(voices)}")
    print(f"\nDefault voice: {DEFAULT_VOICE} (marked with ★ if listed)") # Mark needs adding if needed

async def async_main(args):
    """Asynchronous main function to coordinate generation."""
    
    # Use text from command line or prompt user
    prompt = args.text
    if not prompt:
        # Slightly more robust way to capture positional arguments
        positional_args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
        if positional_args:
            prompt = " ".join(positional_args)
        else:
            try:
                prompt = input("Enter text to synthesize: ")
            except EOFError: # Handle non-interactive environments
                 prompt = "" 
            if not prompt:
                logger.warning("No text provided, using default.")
                prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."

    # Default output file if none provided
    output_file = args.output
    stream_playback = False
    if not output_file:
        if args.stream:
             logger.info("No output file specified, streaming playback...")
             stream_playback = True
        else:
             os.makedirs("outputs", exist_ok=True)
             timestamp = time.strftime("%Y%m%d_%H%M%S")
             sanitized_prompt = re.sub(r'[^\w\-]+', '_', prompt[:30].strip())
             output_file = os.path.abspath(f"outputs/{args.voice}_{sanitized_prompt}_{timestamp}.wav")
             logger.info(f"No output file specified. Saving to {output_file}")
    else:
         output_file = os.path.abspath(args.output)
         # Ensure output directory exists
         os.makedirs(os.path.dirname(output_file), exist_ok=True)


    start_time = time.monotonic()

    try:
        if stream_playback:
            # Generate and stream audio directly to playback
            audio_gen = generate_speech_streaming(
                prompt=prompt, voice=args.voice, temperature=args.temperature,
                top_p=args.top_p, max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty
            )
            await stream_audio_async(audio_gen)
            logger.info("Streaming playback complete.")
        else:
            # Generate and save to file
            await generate_speech_to_file(
                prompt=prompt, output_file=output_file, voice=args.voice,
                temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                repetition_penalty=args.repetition_penalty, use_batching=not args.no_batching
            )
            # No need to print saved message, generate_speech_to_file handles it

    except Exception as e:
        logger.exception(f"Speech generation failed: {e}")
    finally:
        # Clean up resources
        await close_aiohttp_session()
        # Shutdown ProcessPoolExecutor if it was used
        # shutdown_batch_executor() # Do this globally at exit instead

    end_time = time.monotonic()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Orpheus Text-to-Speech using Orpheus-FASTAPI (Async Optimized)")
    parser.add_argument("text", nargs='?', type=str, help="Text to convert to speech (can be positional)")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, choices=AVAILABLE_VOICES, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output", "-o", type=str, help="Output WAV file path. If omitted and --stream is not used, saves to 'outputs/'.")
    parser.add_argument("--stream", action="store_true", help="Stream audio playback directly instead of saving to file (requires --output to be omitted).")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Generation top-p.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Max tokens for generation.")
    # Repetition penalty kept for compatibility but maybe hide if fixed?
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY, help="Repetition penalty (Note: 1.1 recommended).")
    parser.add_argument("--no-batching", action="store_true", help="Disable automatic batch processing for long text (forces streaming API approach even for file output).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")
        
    # Ensure speechpipe is available before proceeding
    if not SPEECHPIPE_AVAILABLE and not args.list_voices:
         logger.critical("SpeechPipe components could not be loaded. Please ensure speechpipe.py is present and functional.")
         sys.exit(1)

    if args.list_voices:
        list_available_voices()
        return

    if args.stream and args.output:
        parser.error("--stream and --output are mutually exclusive.")

    # Run the asynchronous main function
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    finally:
         # Ensure executor cleanup happens on exit
         shutdown_batch_executor()

if __name__ == "__main__":
    # Check if it's the main process before running main logic
    # This helps avoid duplicate execution with some reloaders
    if IS_MAIN_PROCESS:
         main()
    elif not is_reloader_process():
         # If it's not the main process (e.g., ProcessPoolExecutor worker)
         # and not a known reloader process, maybe log it or just exit?
         # print(f"Worker process {os.getpid()} exiting.")
         pass