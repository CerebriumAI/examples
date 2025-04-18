import os
import sys
import requests
import json
import time
import wave
import numpy as np
import sounddevice as sd
import argparse
import threading
import queue as queue_module  # Import as queue_module to avoid naming conflicts
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Generator, Union, Tuple, AsyncGenerator
from dotenv import load_dotenv

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

# Set a flag to avoid repeat messages
IS_RELOADER = is_reloader_process()
if not IS_RELOADER:
    os.environ['UVICORN_STARTED'] = 'true'

# Load environment variables from .env file
load_dotenv()

# Detect hardware capabilities and display information
import torch
import psutil

# Detect if we're on a high-end system based on hardware capabilities
HIGH_END_GPU = False
if torch.cuda.is_available():
    # Get GPU properties
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    gpu_mem_gb = props.total_memory / (1024**3)
    compute_capability = f"{props.major}.{props.minor}"
    
    # Consider high-end if: large VRAM (≥16GB) OR high compute capability (≥8.0) OR large VRAM (≥12GB) with good CC (≥7.0)
    HIGH_END_GPU = (gpu_mem_gb >= 16.0 or 
                    props.major >= 8 or 
                    (gpu_mem_gb >= 12.0 and props.major >= 7))
        
    if HIGH_END_GPU:
        if not IS_RELOADER:
            print(f"🖥️ Hardware: High-end CUDA GPU detected")
            print(f"📊 Device: {gpu_name}")
            print(f"📊 VRAM: {gpu_mem_gb:.2f} GB")
            print(f"📊 Compute Capability: {compute_capability}")
            print("🚀 Using high-performance optimizations")
    else:
        if not IS_RELOADER:
            print(f"🖥️ Hardware: CUDA GPU detected")
            print(f"📊 Device: {gpu_name}")
            print(f"📊 VRAM: {gpu_mem_gb:.2f} GB")
            print(f"📊 Compute Capability: {compute_capability}")
            print("🚀 Using GPU-optimized settings")
else:
    # Get CPU info
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if not IS_RELOADER:
        print(f"🖥️ Hardware: CPU only (No CUDA GPU detected)")
        print(f"📊 CPU: {cpu_cores} cores, {cpu_threads} threads")
        print(f"📊 RAM: {ram_gb:.2f} GB")
        print("⚙️ Using CPU-optimized settings")

# Load configuration from environment variables without hardcoded defaults
# Critical settings - will log errors if missing
required_settings = ["ORPHEUS_API_URL"]
missing_settings = [s for s in required_settings if s not in os.environ]
if missing_settings:
    print(f"ERROR: Missing required environment variable(s): {', '.join(missing_settings)}")
    print("Please set them in .env file or environment. See .env.example for defaults.")

# API connection settings
API_URL = os.environ.get("ORPHEUS_API_URL")
if not API_URL:
    print("WARNING: ORPHEUS_API_URL not set. API calls will fail until configured.")

HEADERS = {
    "Content-Type": "application/json"
}

# Request timeout settings
try:
    REQUEST_TIMEOUT = int(os.environ.get("ORPHEUS_API_TIMEOUT", "120"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_API_TIMEOUT value, using 120 seconds as fallback")
    REQUEST_TIMEOUT = 120

# Model generation parameters from environment variables
try:
    MAX_TOKENS = int(os.environ.get("ORPHEUS_MAX_TOKENS", "8192"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_MAX_TOKENS value, using 8192 as fallback")
    MAX_TOKENS = 8192

try:
    TEMPERATURE = float(os.environ.get("ORPHEUS_TEMPERATURE", "0.6"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_TEMPERATURE value, using 0.6 as fallback")
    TEMPERATURE = 0.6

try:
    TOP_P = float(os.environ.get("ORPHEUS_TOP_P", "0.9"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_TOP_P value, using 0.9 as fallback")
    TOP_P = 0.9

# Repetition penalty is hardcoded to 1.1 which is the only stable value for quality output
REPETITION_PENALTY = 1.1

try:
    SAMPLE_RATE = int(os.environ.get("ORPHEUS_SAMPLE_RATE", "24000"))
except (ValueError, TypeError):
    print("WARNING: Invalid ORPHEUS_SAMPLE_RATE value, using 24000 as fallback")
    SAMPLE_RATE = 24000

# Print loaded configuration only in the main process, not in the reloader
if not IS_RELOADER:
    print(f"Configuration loaded:")
    print(f"  API_URL: {API_URL}")
    print(f"  MAX_TOKENS: {MAX_TOKENS}")
    print(f"  TEMPERATURE: {TEMPERATURE}")
    print(f"  TOP_P: {TOP_P}")
    print(f"  REPETITION_PENALTY: {REPETITION_PENALTY}")

# Parallel processing settings
NUM_WORKERS = 4 if HIGH_END_GPU else 2

# Define voices by language
ENGLISH_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
FRENCH_VOICES = ["pierre", "amelie", "marie"]
GERMAN_VOICES = ["jana", "thomas", "max"]
KOREAN_VOICES = ["유나", "준서"]
HINDI_VOICES = ["ऋतिका"]
MANDARIN_VOICES = ["长乐", "白芷"]
SPANISH_VOICES = ["javi", "sergio", "maria"]
ITALIAN_VOICES = ["pietro", "giulia", "carlo"]

# Combined list for API compatibility
AVAILABLE_VOICES = (
    ENGLISH_VOICES + 
    FRENCH_VOICES + 
    GERMAN_VOICES + 
    KOREAN_VOICES + 
    HINDI_VOICES + 
    MANDARIN_VOICES + 
    SPANISH_VOICES + 
    ITALIAN_VOICES
)
DEFAULT_VOICE = "tara"  # Best voice according to documentation

# Map voices to languages for the UI
VOICE_TO_LANGUAGE = {}
VOICE_TO_LANGUAGE.update({voice: "english" for voice in ENGLISH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "french" for voice in FRENCH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "german" for voice in GERMAN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "korean" for voice in KOREAN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "hindi" for voice in HINDI_VOICES})
VOICE_TO_LANGUAGE.update({voice: "mandarin" for voice in MANDARIN_VOICES})
VOICE_TO_LANGUAGE.update({voice: "spanish" for voice in SPANISH_VOICES})
VOICE_TO_LANGUAGE.update({voice: "italian" for voice in ITALIAN_VOICES})

# Languages list for the UI
AVAILABLE_LANGUAGES = ["english", "french", "german", "korean", "hindi", "mandarin", "spanish", "italian"]

# Import the unified token handling from speechpipe
from .speechpipe import turn_token_into_id, CUSTOM_TOKEN_PREFIX

# Special token IDs for Orpheus model
START_TOKEN_ID = 128259
END_TOKEN_IDS = [128009, 128260, 128261, 128257]

# Performance monitoring
class PerformanceMonitor:
    """Track and report performance metrics"""
    def __init__(self, report_interval=2.0):
        self.start_time = time.time()
        self.token_count = 0
        self.audio_chunks = 0
        self.last_report_time = time.time()
        self.report_interval = report_interval  # seconds
        self.output_bytes = 0  # Track total bytes generated
        self.input_tokens_processed = 0  # Track total input tokens
        
    def add_tokens(self, count: int = 1) -> None:
        self.token_count += count
        self.input_tokens_processed += count
        self.check_report()
        
    def add_audio_chunk(self) -> None:
        self.audio_chunks += 1
        self.check_report()
        
    def add_output_bytes(self, bytes_count: int) -> None:
        """Track total bytes generated for accurate throughput measurement"""
        self.output_bytes += bytes_count
        
    def get_input_count(self) -> int:
        """Get total input tokens processed"""
        return self.input_tokens_processed
        
    def get_output_count(self) -> int:
        """Get total audio chunks generated"""
        return self.audio_chunks
        
    def get_output_bytes(self) -> int:
        """Get total bytes of audio generated"""
        return self.output_bytes
        
    def check_report(self) -> None:
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time
            
    def report(self) -> None:
        elapsed = time.time() - self.start_time
        if elapsed < 0.001:
            return
            
        tokens_per_sec = self.token_count / elapsed
        chunks_per_sec = self.audio_chunks / elapsed
        
        # Estimate audio duration based on audio chunks (each chunk is ~0.085s of audio)
        est_duration = self.audio_chunks * 0.085
        
        # Calculate bandwidth metrics
        kb_generated = self.output_bytes / 1024
        kb_per_sec = kb_generated / elapsed if elapsed > 0 else 0
        
        print(f"Progress: {tokens_per_sec:.1f} tokens/sec, {chunks_per_sec:.1f} chunks/sec, {kb_per_sec:.1f} KB/sec")
        print(f"Generated: est. {est_duration:.1f}s audio, {self.token_count} tokens, {self.audio_chunks} chunks, {kb_generated:.1f}KB in {elapsed:.1f}s")

# Create global performance monitor
perf_monitor = PerformanceMonitor()

def format_prompt(prompt: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    # Validate voice and provide fallback
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE
        
    # Format similar to how engine_class.py does it with special tokens
    formatted_prompt = f"{voice}: {prompt}"
    
    # Add special token markers for the Orpheus-FASTAPI
    special_start = "<|audio|>"  # Using the additional_special_token from config
    special_end = "<|eot_id|>"   # Using the eos_token from config
    
    return f"{special_start}{formatted_prompt}{special_end}"

def generate_tokens_from_api(prompt: str, voice: str = DEFAULT_VOICE, temperature: float = TEMPERATURE, 
                           top_p: float = TOP_P, max_tokens: int = MAX_TOKENS, 
                           repetition_penalty: float = REPETITION_PENALTY) -> Generator[str, None, None]:
    """Generate tokens from text using OpenAI-compatible API with optimized streaming and retry logic."""
    start_time = time.time()
    formatted_prompt = format_prompt(prompt, voice)
    print(f"Generating speech for: {formatted_prompt}")
    
    # Optimize the token generation for GPUs
    if HIGH_END_GPU:
        # Use more aggressive parameters for faster generation on high-end GPUs
        print("Using optimized parameters for high-end GPU")
    elif torch.cuda.is_available():
        print("Using optimized parameters for GPU acceleration")
    
    # Create the request payload (model field may not be required by some endpoints but included for compatibility)
    payload = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True  # Always stream for better performance
    }
    
    # Add model field - this is ignored by many local inference servers for /v1/completions
    # but included for compatibility with OpenAI API and some servers that may use it
    model_name = os.environ.get("ORPHEUS_MODEL_NAME", "Orpheus-3b-FT-Q8_0.gguf")
    payload["model"] = model_name
    
    # Session for connection pooling and retry logic
    session = requests.Session()
    
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            # Make the API request with streaming and timeout
            response = session.post(
                API_URL, 
                headers=HEADERS, 
                json=payload, 
                stream=True,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Error details: {response.text}")
                # Retry on server errors (5xx) but not on client errors (4xx)
                if response.status_code >= 500:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return
            
            # Process the streamed response with better buffering
            buffer = ""
            token_counter = 0
            
            # Iterate through the response to get tokens
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove the 'data: ' prefix
                        
                        if data_str.strip() == '[DONE]':
                            break
                            
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                token_chunk = data['choices'][0].get('text', '')
                                for token_text in token_chunk.split('>'):
                                    token_text = f'{token_text}>'
                                    token_counter += 1
                                    perf_monitor.add_tokens()

                                    if token_text:
                                        yield token_text
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                            continue
            
            # Generation completed successfully
            generation_time = time.time() - start_time
            tokens_per_second = token_counter / generation_time if generation_time > 0 else 0
            print(f"Token generation complete: {token_counter} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
            return
            
        except requests.exceptions.Timeout:
            print(f"Request timed out after {REQUEST_TIMEOUT} seconds")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Token generation failed.")
                return
                
        except requests.exceptions.ConnectionError:
            print(f"Connection error to API at {API_URL}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds... (attempt {retry_count+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Token generation failed.")
                return

# The turn_token_into_id function is now imported from speechpipe.py
# This eliminates duplicate code and ensures consistent behavior

def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """Convert token frames to audio with performance monitoring."""
    # Import here to avoid circular imports
    from .speechpipe import convert_to_audio as orpheus_convert_to_audio
    start_time = time.time()
    result = orpheus_convert_to_audio(multiframe, count)
    
    if result is not None:
        perf_monitor.add_audio_chunk()
        
    return result

async def tokens_decoder(
    tokens_generator,
    performance_monitor: Optional[PerformanceMonitor] = None,
) -> AsyncGenerator[bytes, None]:
    """Optimized asynchronous generator that decodes tokens to audio chunks with minimal latency.
    
    Uses batching and parallel processing to maximize throughput while maintaining
    responsiveness for streaming applications.
    """
    audio_queue = asyncio.Queue(maxsize=512)  # Increased queue size for high throughput
    end_flag = object()
    is_done = False
    
    # Preallocate buffers for better performance
    token_batch = []
    batch_size = 32  # Initial small batch for quick first chunk
    max_batch_size = 128  # Will grow to this for subsequent chunks
    
    # Producer task to fill the queue with batched tokens
    async def producer():
        nonlocal is_done
        try:
            count = 0
            async for token in tokens_generator:
                if token is not None:
                    token_batch.append(token)
                    count += 1
                    
                    # Adaptive batching: small batches at start, larger later
                    current_batch_size = min(count, max_batch_size) if count > 10 else batch_size
                    
                    # Process batch when it reaches target size
                    if len(token_batch) >= current_batch_size:
                        # Avoid blocking on queue.put to prevent backpressure
                        if audio_queue.full():
                            # If queue is full, use smaller batches to reduce latency
                            half_batch = len(token_batch) // 2
                            if half_batch > 0:
                                await audio_queue.put(token_batch[:half_batch])
                                token_batch = token_batch[half_batch:]
                        else:
                            await audio_queue.put(token_batch.copy())
                            token_batch.clear()
                        
                        # Small yield to allow consumer to process
                        await asyncio.sleep(0)
            
            # Put any remaining tokens in the queue
            if token_batch:
                await audio_queue.put(token_batch.copy())
                token_batch.clear()
                
        except Exception as e:
            print(f"Error in tokens producer: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Signal end of token stream
            await audio_queue.put(end_flag)
            is_done = True
    
    # Consumer task to process batched tokens into audio chunks
    async def consumer():
        try:
            # Start with small buffer, grow for subsequent chunks
            audio_buffer_size = 2048  # Starting buffer size
            audio_buffer = bytearray(audio_buffer_size)
            buffer_position = 0
            
            tokens_processed = 0
            current_batch = None
            
            while not is_done or not audio_queue.empty() or current_batch:
                # Get next batch from queue
                if current_batch is None:
                    try:
                        current_batch = await audio_queue.get()
                        # Check for end flag
                        if current_batch is end_flag:
                            break
                    except asyncio.QueueEmpty:
                        # No data available yet, yield control briefly
                        await asyncio.sleep(0.001)
                        continue
                
                if not current_batch:
                    current_batch = None
                    continue
                
                # Process tokens in the current batch efficiently
                batch_tokens = len(current_batch)
                for i in range(batch_tokens):
                    token = current_batch[i]
                    tokens_processed += 1
                    
                    # Build a frame from the token
                    token_frame = []
                    token_id = turn_token_into_id(token, tokens_processed)
                    if token_id is not None and token_id > 0:
                        token_frame.append(token_id)
                    else:
                        continue
                    
                    # Process the token frame
                    audio = convert_to_audio(token_frame, tokens_processed)
                    
                    # Process valid audio data
                    if audio is not None and len(audio) > 0:
                        audio_size = len(audio)
                        
                        # If buffer is too small, resize it
                        if buffer_position + audio_size > len(audio_buffer):
                            # Double buffer size to reduce reallocations
                            new_size = max(len(audio_buffer) * 2, buffer_position + audio_size)
                            new_buffer = bytearray(new_size)
                            new_buffer[:buffer_position] = audio_buffer[:buffer_position]
                            audio_buffer = new_buffer
                        
                        # Copy audio data to buffer
                        audio_buffer[buffer_position:buffer_position + audio_size] = audio
                        buffer_position += audio_size
                        
                        # Yield complete buffer when enough data is accumulated or at end of batch
                        # First chunk should be sent quickly for low latency
                        if tokens_processed < 5 or buffer_position >= 4096 or i == batch_tokens - 1:
                            if buffer_position > 0:
                                if performance_monitor:
                                    performance_monitor.add_output_bytes(buffer_position)
                                
                                yield bytes(audio_buffer[:buffer_position])
                                buffer_position = 0
                
                # Mark batch as processed
                current_batch = None
                
        except Exception as e:
            print(f"Error in tokens consumer: {e}")
            import traceback
            traceback.print_exc()
    
    # Run producer and consumer tasks concurrently
    producer_task = asyncio.create_task(producer())
    
    # Process tokens in the consumer
    async for chunk in consumer():
        yield chunk
    
    # Clean up producer task
    if not producer_task.done():
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass

def tokens_decoder_sync(syn_token_gen, output_file=None):
    """High-throughput synchronous processor with optimized memory and I/O handling."""
    # Use larger queue and batch sizes for high-end systems
    queue_size = 512 if HIGH_END_GPU else 128  # Increased from 400/100 to 512/128
    audio_queue = queue_module.Queue(maxsize=queue_size)
    audio_segments = []
    
    # If output_file is provided, prepare WAV file with efficient I/O
    wav_file = None
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    # Enhanced batch processing for maximum throughput
    batch_size = 96 if HIGH_END_GPU else 48  # Increased from 64/32 to 96/48
    
    # Thread synchronization with optimized events
    producer_done_event = threading.Event()
    producer_started_event = threading.Event()
    
    # Convert synchronous generator to async with efficient batching
    async def async_token_gen():
        batch = []
        try:
            for token in syn_token_gen:
                batch.append(token)
                if len(batch) >= batch_size:
                    for t in batch:
                        yield t
                    batch = []
                    # Minimal cooperative multitasking
                    await asyncio.sleep(0)
        except Exception as e:
            print(f"Error in token generator: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Process any remaining tokens in the final batch
            for t in batch:
                yield t
    
    async def async_producer():
        # Performance tracking with efficient metrics
        start_time = time.time()
        chunk_count = 0
        last_log_time = start_time
        
        try:
            # Signal that producer has started
            producer_started_event.set()
            
            # Process audio with optimized error handling
            async for audio_chunk in tokens_decoder(async_token_gen()):
                # Process each audio chunk
                if audio_chunk:
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
                    
                    # Reduced logging frequency for better performance
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:  # Increased from 3 to 5 seconds
                        elapsed = current_time - last_log_time
                        if elapsed > 0:
                            recent_rate = chunk_count / elapsed
                            print(f"Audio generation rate: {recent_rate:.2f} chunks/second")
                        last_log_time = current_time
                        # Reset counter for next interval
                        chunk_count = 0
        except Exception as e:
            print(f"Error in token processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Always signal completion
            producer_done_event.set()
            # Add sentinel to queue
            audio_queue.put(None)

    def run_async():
        """Run the async producer in its own thread"""
        asyncio.run(async_producer())

    # Use a separate thread with higher priority for producer
    thread = threading.Thread(target=run_async, name="TokenProcessor")
    thread.daemon = True  # Allow thread to be terminated when main thread exits
    thread.start()
    
    # Wait for producer to actually start before proceeding
    # This avoids race conditions where we might try to read from an empty queue
    # before the producer has had a chance to add anything
    producer_started_event.wait(timeout=5.0)
    
    # Optimized I/O approach for all systems
    # This approach is simpler and more reliable than separate code paths
    write_buffer = bytearray()
    buffer_max_size = 1024 * 1024  # 1MB max buffer size (adjustable)
    
    # Keep track of the last time we checked for completion
    last_check_time = time.time()
    check_interval = 1.0  # Check producer status every second
    
    # Process audio chunks until we're done
    while True:
        try:
            # Get the next audio chunk with a short timeout
            # This allows us to periodically check status and handle other events
            audio = audio_queue.get(timeout=0.1)
            
            # None marker indicates end of stream
            if audio is None:
                print("Received end-of-stream marker")
                break
            
            # Store the audio segment for return value
            audio_segments.append(audio)
            
            # Write to file if needed
            if wav_file:
                write_buffer.extend(audio)
                
                # Flush buffer if it's large enough
                if len(write_buffer) >= buffer_max_size:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()  # Reset buffer
        
        except queue_module.Empty:
            # No data available right now
            current_time = time.time()
            
            # Periodically check if producer is done
            if current_time - last_check_time > check_interval:
                last_check_time = current_time
                
                # If producer is done and queue is empty, we're finished
                if producer_done_event.is_set() and audio_queue.empty():
                    print("Producer done and queue empty - finishing consumer")
                    break
                
                # Flush buffer periodically even if not full
                if wav_file and len(write_buffer) > 0:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()  # Reset buffer
    
    # Extra safety check - ensure thread is done
    if thread.is_alive():
        print("Waiting for token processor thread to complete...")
        thread.join(timeout=10.0)
        if thread.is_alive():
            print("WARNING: Token processor thread did not complete within timeout")
    
    # Final flush of any remaining data
    if wav_file and len(write_buffer) > 0:
        print(f"Final buffer flush: {len(write_buffer)} bytes")
        wav_file.writeframes(write_buffer)
    
    # Close WAV file if opened
    if wav_file:
        wav_file.close()
        if output_file:
            print(f"Audio saved to {output_file}")
    
    # Calculate and print detailed performance metrics
    if audio_segments:
        total_bytes = sum(len(segment) for segment in audio_segments)
        duration = total_bytes / (2 * SAMPLE_RATE)  # 2 bytes per sample at 24kHz
        total_time = time.time() - perf_monitor.start_time
        realtime_factor = duration / total_time if total_time > 0 else 0
        
        print(f"Generated {len(audio_segments)} audio segments")
        print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        print(f"Realtime factor: {realtime_factor:.2f}x")
        
        if realtime_factor < 1.0:
            print("⚠️ Warning: Generation is slower than realtime")
        else:
            print(f"✓ Generation is {realtime_factor:.1f}x faster than realtime")
    
    return audio_segments

def stream_audio(audio_buffer):
    """Stream audio buffer to output device with error handling."""
    if audio_buffer is None or len(audio_buffer) == 0:
        return
    
    try:
        # Convert bytes to NumPy array (16-bit PCM)
        audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
        
        # Normalize to float in range [-1, 1] for playback
        audio_float = audio_data.astype(np.float32) / 32767.0
        
        # Play the audio with proper device selection and error handling
        sd.play(audio_float, SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"Audio playback error: {e}")

import re
import numpy as np
from io import BytesIO
import wave

def split_text_into_sentences(text):
    """Split text into sentences with a more reliable approach."""
    # We'll use a simple approach that doesn't rely on variable-width lookbehinds
    # which aren't supported in Python's regex engine
    
    # First, split on common sentence ending punctuation
    # This isn't perfect but works for most cases and avoids the regex error
    parts = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        
        # If we hit a sentence ending followed by a space, consider this a potential sentence end
        if char in (' ', '\n', '\t') and len(current_sentence) > 1:
            prev_char = current_sentence[-2]
            if prev_char in ('.', '!', '?'):
                # Check if this is likely a real sentence end and not an abbreviation
                # (Simple heuristic: if there's a space before the period, it's likely a real sentence end)
                if len(current_sentence) > 3 and current_sentence[-3] not in ('.', ' '):
                    parts.append(current_sentence.strip())
                    current_sentence = ""
    
    # Add any remaining text
    if current_sentence.strip():
        parts.append(current_sentence.strip())
    
    # Combine very short segments to avoid tiny audio files
    min_chars = 20  # Minimum reasonable sentence length
    combined_sentences = []
    i = 0
    
    while i < len(parts):
        current = parts[i]
        
        # If this is a short sentence and not the last one, combine with next
        while i < len(parts) - 1 and len(current) < min_chars:
            i += 1
            current += " " + parts[i]
            
        combined_sentences.append(current)
        i += 1
    
    return combined_sentences

async def stream_speech_from_api(
    prompt: str,
    voice: str = "tara"
) -> AsyncGenerator[bytes, None]:
    """Generate speech from text and stream audio chunks with maximum throughput.
    
    Optimized for streaming API with minimal latency for first chunk and high
    throughput for subsequent chunks.
    
    Args:
        prompt: Text to convert to speech
        voice: Voice to use (default: Orpheus)
        
    Yields:
        Audio chunks as bytes
    """
    print(f"Streaming: {len(prompt)} chars, voice: {voice}")
    
    # Initialize performance monitoring
    perf_monitor = PerformanceMonitor(report_interval=0.5)  # More frequent updates
    start_time = time.time()
    
    # Track processed tokens to prevent repeats or drops
    processed_token_ids = set()
    last_token_position = 0
    
    try:
        # Generate tokens from API (synchronous generator)
        token_gen = generate_tokens_from_api(prompt, voice)
        
        # Create inner async generator for better token tracking
        async def tracked_token_generator():
            nonlocal last_token_position
            token_buffer = []
            
            for token in token_gen:
                # Buffer tokens to detect boundaries properly
                token_buffer.append(token)
                
                # Process tokens in small groups to maintain context
                if len(token_buffer) >= 7:  # Standard Orpheus model frame size
                    for t in token_buffer:
                        token_id = hash(t)  # Use hash as unique identifier
                        if token_id not in processed_token_ids:
                            processed_token_ids.add(token_id)
                            yield t
                    token_buffer = []
            
            # Process any remaining tokens in the buffer
            for t in token_buffer:
                token_id = hash(t)
                if token_id not in processed_token_ids:
                    processed_token_ids.add(token_id)
                    yield t
        
        # Process tokens with improved overlap handling
        async for chunk in tokens_decoder(tracked_token_generator()):
            if chunk:
                # Track performance stats for monitoring
                chunk_size = len(chunk)
                perf_monitor.add_audio_chunk()
                perf_monitor.add_output_bytes(chunk_size)
                
                # Yield valid audio chunks
                yield chunk
                
    except Exception as e:
        print(f"Error in stream_speech_from_api: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Report performance
        elapsed = time.time() - start_time
        tokens_count = perf_monitor.get_input_count()
        chunks_count = perf_monitor.get_output_count()
        bytes_count = perf_monitor.get_output_bytes()
        
        if elapsed > 0:
            print(f"Stream complete: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            print(f"Performance: {tokens_count/elapsed:.2f} tokens/sec, {chunks_count/elapsed:.2f} chunks/sec")
            print(f"Generated {chunks_count} chunks, {bytes_count/1024:.1f}KB in {elapsed:.2f}s")

def generate_speech_from_api(prompt, voice=DEFAULT_VOICE, output_file=None, temperature=TEMPERATURE, 
                     top_p=TOP_P, max_tokens=MAX_TOKENS, repetition_penalty=None, 
                     use_batching=True, max_batch_chars=1000):
    """Generate speech from text using Orpheus model with performance optimizations."""
    print(f"Starting speech generation for '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
    print(f"Using voice: {voice}, GPU acceleration: {'Yes (High-end)' if HIGH_END_GPU else 'Yes' if torch.cuda.is_available() else 'No'}")
    
    # Reset performance monitor
    global perf_monitor
    perf_monitor = PerformanceMonitor()
    
    start_time = time.time()
    
    # For shorter text, use the standard non-batched approach
    if not use_batching or len(prompt) < max_batch_chars:
        # Note: we ignore any provided repetition_penalty and always use the hardcoded value
        # This ensures consistent quality regardless of what might be passed in
        result = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=prompt, 
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY  # Always use hardcoded value
            ),
            output_file=output_file
        )
        
        # Report final performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total speech generation completed in {total_time:.2f} seconds")
        
        return result
    
    # For longer text, use sentence-based batching
    print(f"Using sentence-based batching for text with {len(prompt)} characters")
    
    # Split the text into sentences
    sentences = split_text_into_sentences(prompt)
    print(f"Split text into {len(sentences)} segments")
    
    # Create batches by combining sentences up to max_batch_chars
    batches = []
    current_batch = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the batch size, start a new batch
        if len(current_batch) + len(sentence) > max_batch_chars and current_batch:
            batches.append(current_batch)
            current_batch = sentence
        else:
            # Add separator space if needed
            if current_batch:
                current_batch += " "
            current_batch += sentence
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(current_batch)
    
    print(f"Created {len(batches)} batches for processing")
    
    # Process each batch and collect audio segments
    all_audio_segments = []
    batch_temp_files = []
    
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} characters)")
        
        # Create a temporary file for this batch if an output file is requested
        temp_output_file = None
        if output_file:
            temp_output_file = f"outputs/temp_batch_{i}_{int(time.time())}.wav"
            batch_temp_files.append(temp_output_file)
        
        # Generate speech for this batch
        batch_segments = tokens_decoder_sync(
            generate_tokens_from_api(
                prompt=batch,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=REPETITION_PENALTY
            ),
            output_file=temp_output_file
        )
        
        # Add to our collection
        all_audio_segments.extend(batch_segments)
    
    # If an output file was requested, stitch together the temporary files
    if output_file and batch_temp_files:
        # Stitch together WAV files
        stitch_wav_files(batch_temp_files, output_file)
        
        # Clean up temporary files
        for temp_file in batch_temp_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {temp_file}: {e}")
    
    # Report final performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate combined duration
    if all_audio_segments:
        total_bytes = sum(len(segment) for segment in all_audio_segments)
        duration = total_bytes / (2 * SAMPLE_RATE)  # 2 bytes per sample at 24kHz
        print(f"Generated {len(all_audio_segments)} audio segments")
        print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        print(f"Realtime factor: {duration/total_time:.2f}x")
        
    print(f"Total speech generation completed in {total_time:.2f} seconds")
    
    return all_audio_segments

def stitch_wav_files(input_files, output_file, crossfade_ms=50):
    """Stitch multiple WAV files together with crossfading for smooth transitions."""
    if not input_files:
        return
        
    print(f"Stitching {len(input_files)} WAV files together with {crossfade_ms}ms crossfade")
    
    # If only one file, just copy it
    if len(input_files) == 1:
        import shutil
        shutil.copy(input_files[0], output_file)
        return
    
    # Convert crossfade_ms to samples
    crossfade_samples = int(SAMPLE_RATE * crossfade_ms / 1000)
    print(f"Using {crossfade_samples} samples for crossfade at {SAMPLE_RATE}Hz")
    
    # Build the final audio in memory with crossfades
    final_audio = np.array([], dtype=np.int16)
    first_params = None
    
    for i, input_file in enumerate(input_files):
        try:
            with wave.open(input_file, 'rb') as wav:
                if first_params is None:
                    first_params = wav.getparams()
                elif wav.getparams() != first_params:
                    print(f"Warning: WAV file {input_file} has different parameters")
                    
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                
                if i == 0:
                    # First segment - use as is
                    final_audio = audio
                else:
                    # Apply crossfade with previous segment
                    if len(final_audio) >= crossfade_samples and len(audio) >= crossfade_samples:
                        # Create crossfade weights
                        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
                        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
                        
                        # Apply crossfade
                        crossfade_region = (final_audio[-crossfade_samples:] * fade_out + 
                                           audio[:crossfade_samples] * fade_in).astype(np.int16)
                        
                        # Combine: original without last crossfade_samples + crossfade + new without first crossfade_samples
                        final_audio = np.concatenate([final_audio[:-crossfade_samples], 
                                                    crossfade_region, 
                                                    audio[crossfade_samples:]])
                    else:
                        # One segment too short for crossfade, just append
                        print(f"Segment {i} too short for crossfade, concatenating directly")
                        final_audio = np.concatenate([final_audio, audio])
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            if i == 0:
                raise  # Critical failure if first file fails
    
    # Write the final audio data to the output file
    try:
        with wave.open(output_file, 'wb') as output_wav:
            if first_params is None:
                raise ValueError("No valid WAV files were processed")
                
            output_wav.setparams(first_params)
            output_wav.writeframes(final_audio.tobytes())
        
        print(f"Successfully stitched audio to {output_file} with crossfading")
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}")
        raise

def list_available_voices():
    """List all available voices with the recommended one marked."""
    print("Available voices (in order of conversational realism):")
    for i, voice in enumerate(AVAILABLE_VOICES):
        marker = "★" if voice == DEFAULT_VOICE else " "
        print(f"{marker} {voice}")
    print(f"\nDefault voice: {DEFAULT_VOICE}")
    
    print("\nAvailable emotion tags:")
    print("<laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Orpheus Text-to-Speech using Orpheus-FASTAPI")
    parser.add_argument("--text", type=str, help="Text to convert to speech")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help=f"Voice to use (default: {DEFAULT_VOICE})")
    parser.add_argument("--output", type=str, help="Output WAV file path")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P, help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY, 
                       help="Repetition penalty (fixed at 1.1 for stable generation - parameter kept for compatibility)")
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_available_voices()
        return
    
    # Use text from command line or prompt user
    prompt = args.text
    if not prompt:
        if len(sys.argv) > 1 and sys.argv[1] not in ("--voice", "--output", "--temperature", "--top_p", "--repetition_penalty"):
            prompt = " ".join([arg for arg in sys.argv[1:] if not arg.startswith("--")])
        else:
            prompt = input("Enter text to synthesize: ")
            if not prompt:
                prompt = "Hello, I am Orpheus, an AI assistant with emotional speech capabilities."
    
    # Default output file if none provided
    output_file = args.output
    if not output_file:
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        # Generate a filename based on the voice and a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/{args.voice}_{timestamp}.wav"
        print(f"No output file specified. Saving to {output_file}")
    
    # Generate speech
    start_time = time.time()
    audio_segments = generate_speech_from_api(
        prompt=prompt,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        output_file=output_file
    )
    end_time = time.time()
    
    print(f"Speech generation completed in {end_time - start_time:.2f} seconds")
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    main()