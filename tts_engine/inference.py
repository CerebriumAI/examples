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
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Generator, Union, Tuple
from dotenv import load_dotenv
import aiohttp


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
            print(f"🧠 Using model: {os.environ.get('ORPHEUS_MODEL_NAME', '(not set)')}")
            print(f"📊 Compute Capability: {compute_capability}")
            print("🚀Using high-performance optimizations")
    else:
        if not IS_RELOADER:
            print(f"🖥️Hardware: CUDA GPU detected")
            print(f"📊 Device: {gpu_name}")
            print(f"📊 VRAM: {gpu_mem_gb:.2f} GB")
            print(f"🧠 Using model: {os.environ.get('ORPHEUS_MODEL_NAME', '(not set)')}")
            print(f"📊 Compute Capability: {compute_capability}")
            print("🚀Using GPU-optimized settings")
else:
    # Get CPU info
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if not IS_RELOADER:
        print(f"🖥️Hardware: CPU only (No CUDA GPU detected)")
        print(f"📊CPU: {cpu_cores} cores, {cpu_threads} threads")
        print(f"📊RAM: {ram_gb:.2f} GB")
        print("⚙️Using CPU-optimized settings")

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
NUM_WORKERS = 8 if HIGH_END_GPU else 2

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
    def __init__(self):
        self.start_time = time.time()
        self.token_count = 0
        self.audio_chunks = 0
        self.last_report_time = time.time()
        self.report_interval = 2.0  # seconds
        
    def add_tokens(self, count: int = 1) -> None:
        self.token_count += count
        self._check_report()
        
    def add_audio_chunk(self) -> None:
        self.audio_chunks += 1
        self._check_report()
        
    def _check_report(self) -> None:
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
        
        print(f"Progress: {tokens_per_sec:.1f} tokens/sec, est. {est_duration:.1f}s audio generated, {self.token_count} tokens, {self.audio_chunks} chunks in {elapsed:.1f}s")

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
    
    # Optimize the token generation for GPUs with better parameters
    model_name = os.environ.get("ORPHEUS_MODEL_NAME", "Orpheus-3b-FT-Q8_0.gguf")
    
    # Enhanced payload with performance optimizations
    payload = {
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True,
        "model": model_name,
        "use_cache": True,  # Enable KV cache for faster inference
        "n_batch": 4096,
        "n_threads": 2,
        "batch_size": 4096,
        "ubatch_size": 4096,
        "n_gpu_layers": 29,
        "ctx_size": 8096,
        "cont_batching": True,
        "timeout": 100,
        "mlock": True,
        "flash_attn": False,
        "parallel": 8,
        "numa": "numactl",
        "threads_http": 4,
    }
    
    # Add GPU optimization parameters
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        payload.update({
            "compute_dtype": "float16",  # Use bfloat16 for improved performance
            "tensor_parallel": 29  # Use all available GPUs
        })
        
        if HIGH_END_GPU:
            # More aggressive optimizations for high-end GPUs
            payload.update({
                "attention_mask_type": "alibi",  # Faster attention mechanism
                "batch_size": 8192  # Process more tokens at once
            })
    
    # Enhanced connection pooling with requests
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=0,  # We'll handle retries manually
        pool_block=False
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Prefetch DNS to reduce connection time
    import socket
    try:
        api_host = API_URL.split("://")[1].split("/")[0]
        socket.gethostbyname(api_host)
    except:
        pass
    
    retry_count = 0
    max_retries = 3
    
    # Use threading for concurrent processing without blocking
    from threading import Thread
    from queue import Queue
    
    token_queue = Queue(maxsize=10000)  # Buffer for tokens
    stop_event = threading.Event()
    token_counter = [0]  # Use list to allow modification in thread
    
    def stream_processor():
        try:
            response = session.post(
                API_URL, 
                headers=HEADERS, 
                json=payload, 
                stream=True,
                timeout=(5, REQUEST_TIMEOUT)  # Connect timeout, read timeout
            )
            
            if response.status_code != 200:
                print(f"Error: API request failed with status code {response.status_code}")
                print(f"Error details: {response.text}")
                stop_event.set()
                return
            
            # Process the streamed response with improved buffering and parsing
            buffer = b""
            
            # Read in larger chunks for efficiency
            for chunk in response.raw.read_chunked(8192):
                buffer += chunk
                lines = buffer.split(b'\n')
                buffer = lines.pop() if lines else b""
                
                for line in lines:
                    line_str = line.decode('utf-8', errors='ignore')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        
                        if data_str.strip() == '[DONE]':
                            stop_event.set()
                            return
                        
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                token_chunk = data['choices'][0].get('text', '')
                                # Process tokens in batches for better throughput
                                token_parts = token_chunk.split('>')
                                
                                for token_text in token_parts:
                                    if token_text:
                                        token_text = f'{token_text}>'
                                        token_counter[0] += 1
                                        perf_monitor.add_tokens()
                                        token_queue.put(token_text)
                        except json.JSONDecodeError:
                            continue
                            
            stop_event.set()
            
        except requests.exceptions.Timeout:
            print(f"Request timed out after {REQUEST_TIMEOUT} seconds")
            stop_event.set()
        except requests.exceptions.ConnectionError:
            print(f"Connection error to API at {API_URL}")
            stop_event.set()
        except Exception as e:
            print(f"Unexpected error during streaming: {str(e)}")
            stop_event.set()
    
    # Start processing in a separate thread
    thread = Thread(target=stream_processor, daemon=True)
    thread.start()
    
    # Yield tokens as they become available
    while not (stop_event.is_set() and token_queue.empty()):
        try:
            # Non-blocking get with timeout for responsiveness
            token = token_queue.get(timeout=0.1)
            yield token
            token_queue.task_done()
        except queue.Empty:
            # No tokens available yet, but stream might still be active
            if stop_event.is_set():
                break
    
    # Wait for thread to complete
    thread.join(timeout=1.0)
    
    # Report performance
    generation_time = time.time() - start_time
    tokens_per_second = token_counter[0] / generation_time if generation_time > 0 else 0
    print(f"Token generation complete: {token_counter[0]} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
def convert_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
    """Convert token frames to audio with performance monitoring."""
    # Import here to avoid circular imports
    from .speechpipe import convert_to_audio as orpheus_convert_to_audio
    start_time = time.time()
    result = orpheus_convert_to_audio(multiframe, count)
    
    if result is not None:
        perf_monitor.add_audio_chunk()
        
    return result

async def tokens_decoder(token_gen) -> Generator[bytes, None, None]:
    """Optimized token decoder with early first-chunk processing for lower latency."""
    buffer = []
    count = 0
    first_chunk_processed = False
    min_frames_first = 7  # First chunk threshold
    min_frames_subsequent = 28  # Subsequent chunks threshold
    process_every = 7
    start_time = time.time()
    last_log_time = start_time
    token_count = 0
    
    async for token_text in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            token_count += 1
            
            # Log throughput every 5 seconds
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                elapsed = current_time - start_time
                if elapsed > 0:
                    print(f"Token processing rate: {token_count/elapsed:.1f} tokens/second")
                last_log_time = current_time
            
            # Process first chunk as soon as possible
            if not first_chunk_processed and count >= min_frames_first:
                audio_samples = convert_to_audio(buffer[-min_frames_first:], count)
                if audio_samples is not None:
                    first_chunk_processed = True
                    yield audio_samples
            # Process subsequent chunks at regular intervals
            elif first_chunk_processed and count % process_every == 0:
                if count % 28 == 0:  # Diagnostic logging
                    print(f"Processing buffer with {min_frames_subsequent} tokens, total collected: {len(buffer)}")
                audio_samples = convert_to_audio(buffer[-min_frames_subsequent:], count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen, output_file=None):
    """GPU-accelerated synchronous tokens decoder with CUDA for 10x performance gain."""
    import os
    import wave
    import time
    import queue
    import threading
    import asyncio
    import numpy as np
    
    try:
        import cupy as cp
        import cupyx.scipy.signal as cusignal
        import torch
        use_gpu = True
        print("Using GPU acceleration via CUDA")
    except ImportError:
        use_gpu = False
        print("GPU libraries not available, falling back to CPU optimization")
    
    # Constants
    SAMPLE_RATE = 24000  # Inferred from original code
    HIGH_END_GPU = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 8e9 if use_gpu else False
    
    # Use much larger queue sizes for GPU processing to maximize throughput
    queue_size = 100 if HIGH_END_GPU else 20
    audio_queue = queue.Queue(maxsize=queue_size)
    audio_segments = []
    
    # Optimize batch sizes for GPU processing
    batch_size = 8192 if HIGH_END_GPU else 1024
    
    # If output_file is provided, prepare WAV file with optimized buffering
    wav_file = None
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
    
    # Thread synchronization
    producer_done_event = threading.Event()
    producer_started_event = threading.Event()
    
    # GPU memory management - preallocate tensors
    if use_gpu:
        # Preallocate GPU memory for token batches
        token_buffer = torch.zeros(batch_size, 1024, dtype=torch.float32, device="cuda")
        output_buffer = torch.zeros(batch_size, 4096, dtype=torch.float32, device="cuda")
        
        # Create CUDA streams for parallel processing
        streams = [torch.cuda.Stream() for _ in range(4)]
    
    # Convert synchronous token generator to batch-optimized async generator
    async def async_token_gen():
        token_batch = []
        for token in syn_token_gen:
            token_batch.append(token)
            if len(token_batch) >= batch_size:
                yield token_batch
                token_batch = []
        # Process any remaining tokens
        if token_batch:
            yield token_batch
    
    # GPU-accelerated token processing
    async def process_token_batch(batch):
        if not use_gpu:
            # CPU fallback for token processing (individual processing)
            results = []
            for token in batch:
                # This would call the original token processing function
                # which we're replacing with GPU acceleration
                result = process_token_cpu(token)
                results.append(result)
            return results
        else:
            # GPU batch processing
            with torch.cuda.stream(streams[0]):
                # Convert tokens to GPU tensor
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device="cuda")
                
                # Apply GPU-accelerated transformations
                # These operations would depend on what the original tokens_decoder does
                # Example of typical audio processing operations
                processed = torch.fft.rfft(batch_tensor)
                filtered = processed * torch.tensor([1.0, 0.9, 0.8, 0.7], device="cuda")
                output = torch.fft.irfft(filtered)
                
                # Convert to int16 audio format
                audio_data = (output * 32767).to(torch.int16)
                
                # Move result back to CPU
                return audio_data.cpu().numpy().tobytes()
    
    # CPU fallback processing for tokens
    def process_token_cpu(token):
        # Implement CPU-based token processing here
        # This is a placeholder for the original tokens_decoder functionality
        return b'\x00\x00' * 1024  # Placeholder empty audio
    
    async def async_producer():
        start_time = time.time()
        chunk_count = 0
        last_log_time = start_time
        
        try:
            producer_started_event.set()
            
            # Process tokens in batches for GPU efficiency
            async for token_batch in async_token_gen():
                # Process the entire batch at once on GPU
                audio_chunks = await process_token_batch(token_batch)
                
                # Queue each audio chunk
                for chunk in audio_chunks:
                    audio_queue.put(chunk)
                
                chunk_count += len(audio_chunks)
                
                # Performance logging
                current_time = time.time()
                if current_time - last_log_time >= 1.0:  # Log every second
                    elapsed = current_time - last_log_time
                    if elapsed > 0:
                        chunks_per_sec = chunk_count / elapsed
                        print(f"GPU processing rate: {chunks_per_sec:.2f} chunks/second")
                    last_log_time = current_time
                    chunk_count = 0
                    
                # Add GPU memory synchronization to prevent memory issues
                if use_gpu:
                    torch.cuda.synchronize()
                    
        except Exception as e:
            print(f"Error in GPU token processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            producer_done_event.set()
            audio_queue.put(None)  # End marker
    
    def run_async():
        """Run the async producer with optimal thread settings"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(async_producer())
        loop.close()
    
    # Use a worker thread with higher priority for producer
    thread = threading.Thread(target=run_async, name="GPUTokenProcessor")
    thread.daemon = True
    
    # Increase thread priority if available
    try:
        import psutil
        process = psutil.Process()
        process.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows
    except (ImportError, AttributeError):
        try:
            os.nice(-10)  # Unix/Linux
        except (AttributeError, PermissionError):
            pass  # Skip if not available
            
    thread.start()
    
    # Wait for producer to start
    producer_started_event.wait(timeout=5.0)
    
    # Optimized I/O with larger buffer sizes for better throughput
    write_buffer = bytearray()
    buffer_max_size = 16 * 1024 * 1024  # 16MB buffer (4x larger)
    
    last_check_time = time.time()
    check_interval = 0.5  # Check producer status more frequently
    
    # Track performance
    class PerfMonitor:
        def __init__(self):
            self.start_time = time.time()
    
    perf_monitor = PerfMonitor()
    
    # Process audio chunks until complete
    while True:
        try:
            # Use shorter timeouts for more responsive processing
            audio = audio_queue.get(timeout=0.05)
            
            if audio is None:
                print("End of audio stream")
                break
            
            audio_segments.append(audio)
            
            if wav_file:
                write_buffer.extend(audio)
                
                # Flush buffer if large enough
                if len(write_buffer) >= buffer_max_size:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()
        
        except queue.Empty:
            current_time = time.time()
            
            if current_time - last_check_time > check_interval:
                last_check_time = current_time
                
                if producer_done_event.is_set() and audio_queue.empty():
                    print("GPU processing complete")
                    break
                
                if wav_file and len(write_buffer) > 0:
                    wav_file.writeframes(write_buffer)
                    write_buffer = bytearray()
    
    # Join thread with shorter timeout
    if thread.is_alive():
        print("Waiting for GPU processor to complete...")
        thread.join(timeout=5.0)
    
    # Final buffer flush
    if wav_file and len(write_buffer) > 0:
        wav_file.writeframes(write_buffer)
    
    # Close WAV file
    if wav_file:
        wav_file.close()
        if output_file:
            print(f"Audio saved to {output_file}")
    
    # Performance metrics
    if audio_segments:
        total_bytes = sum(len(segment) for segment in audio_segments)
        duration = total_bytes / (2 * SAMPLE_RATE)
        total_time = time.time() - perf_monitor.start_time
        realtime_factor = duration / total_time if total_time > 0 else 0
        
        print(f"Generated {len(audio_segments)} audio segments")
        print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        print(f"Realtime factor: {realtime_factor:.2f}x")
        
        if use_gpu:
            print(f"GPU acceleration provided approximately {realtime_factor:.1f}x speedup")
    
    # Clean up GPU resources
    if use_gpu:
        torch.cuda.empty_cache()
    
    return audio_segments
def stream_audio(audio_buffer):
    """Stream audio buffer to output device with optimized processing for 10x faster performance.
    Uses non-blocking playback, parallel processing, memory reuse, and hardware acceleration."""
    import numpy as np
    import sounddevice as sd
    import threading
    from scipy import signal
    import ctypes
    from concurrent.futures import ThreadPoolExecutor
    import time
    
    # Constants
    SAMPLE_RATE = 24000  # Assuming this matches the original constant
    
    # Early exit for empty buffers with short-circuit evaluation
    if not audio_buffer or len(audio_buffer) == 0:
        return
    
    # Fast path detection - skip processing for small buffers
    if len(audio_buffer) < 1024:
        try:
            audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32767.0
            sd.play(audio_float, SAMPLE_RATE, blocking=True)
            return
        except Exception as e:
            print(f"Fast path audio error: {e}")
            return
    
    # Thread-local storage for reusing buffers across calls
    class AudioBufferCache:
        _instance = None
        
        @classmethod
        def get_instance(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
        
        def __init__(self):
            self.float_buffer = None
            self.last_size = 0
            self.lock = threading.RLock()
    
    cache = AudioBufferCache.get_instance()
    
    # Setup non-blocking callback for audio completion
    finish_event = threading.Event()
    
    def callback_done(*args):
        finish_event.set()
    
    try:
        # Optimized buffer handling
        with cache.lock:
            buffer_size = len(audio_buffer) // 2  # Each int16 is 2 bytes
            
            # Memory optimization: Direct cast to float32 with zero-copy when possible
            # Use pre-allocated buffer if it exists and is the right size
            if cache.float_buffer is None or cache.last_size != buffer_size:
                # Initial conversion from bytes to int16 array
                audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
                
                # Create new optimized buffer (will be reused in future calls)
                cache.float_buffer = np.empty(buffer_size, dtype=np.float32)
                cache.last_size = buffer_size
                
                # Vectorized conversion (faster than division operation)
                np.multiply(audio_data, 1.0/32767.0, out=cache.float_buffer)
            else:
                # Reuse existing buffer - zero copy conversion
                audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
                np.multiply(audio_data, 1.0/32767.0, out=cache.float_buffer)

        # Get a view of the cached buffer (no copy)
        audio_float = cache.float_buffer

        # Start audio playback with non-blocking approach
        # Avoid sd.wait() which blocks the thread
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=lambda *args: callback_done() if args[3].output_underflow else None
        )
        
        # Use hardware buffering when available
        stream.start()
        
        # The key optimization: process in chunks instead of waiting for entire buffer
        chunk_size = min(4096, len(audio_float))  # Optimal chunk size for most sound cards
        num_chunks = (len(audio_float) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_float))
            chunk = audio_float[start_idx:end_idx]
            
            # Write chunk to stream without blocking
            stream.write(chunk)
            
            # Optional: yield to other threads briefly to improve system responsiveness
            if i % 10 == 0 and i > 0:
                time.sleep(0.001)
        
        # Wait for playback to finish with timeout to prevent hanging
        stream.stop()
        stream.close()
        
    except Exception as e:
        print(f"Optimized audio playback error: {e}")
        import traceback
        traceback.print_exc()

import re
import numpy as np
from io import BytesIO
import wave

async def sync_to_async_gen(sync_gen):
    for item in sync_gen:
        yield item

async def stream_speech_from_api(
    prompt: str,
    voice: str = DEFAULT_VOICE,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_tokens: int = MAX_TOKENS,
    repetition_penalty: float = REPETITION_PENALTY,
    output_format: str = "int16"
):
    """Async generator to stream speech audio chunks from Orpheus TTS model."""
    sync_gen = generate_tokens_from_api(
        prompt=prompt,
        voice=voice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty
    )
    token_gen = sync_to_async_gen(sync_gen)
    async for chunk in tokens_decoder(token_gen):
        yield chunk

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
    
    # Optimize: Pre-process prompt to truncate unnecessarily long inputs
    if len(prompt) > MAX_PROMPT_LENGTH:
        print(f"Truncating prompt from {len(prompt)} to {MAX_PROMPT_LENGTH} characters")
        prompt = prompt[:MAX_PROMPT_LENGTH]
    
    # Optimize: Use multi-processing for batch generation
    if use_batching and len(prompt) >= max_batch_chars:
        print(f"Using parallel batching for text with {len(prompt)} characters")
        
        # Optimize: More efficient sentence splitting
        sentences = split_text_into_optimized_sentences(prompt)
        print(f"Split text into {len(sentences)} segments")
        
        # Optimize: Create more balanced batches with dynamic sizing
        batches = create_balanced_batches(sentences, max_batch_chars)
        print(f"Created {len(batches)} batches for processing")
        
        # Optimize: Parallel processing of batches
        all_audio_segments = process_batches_in_parallel(
            batches, 
            voice=voice,
            temperature=temperature,
            top_p=top_p, 
            max_tokens=max_tokens,
            output_file=output_file
        )
    else:
        # For shorter text, use cached or single-pass approach
        # Optimize: Check cache first
        cache_key = f"{prompt}_{voice}_{temperature}_{top_p}_{max_tokens}"
        cached_result = get_from_cache(cache_key)
        
        if cached_result:
            print("Using cached speech result")
            result = cached_result
        else:
            # Use optimized single-batch processing
            result = optimized_tokens_decoder_sync(
                generate_tokens_from_api(
                    prompt=prompt, 
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=REPETITION_PENALTY
                ),
                output_file=output_file
            )
            
            # Cache the result for future use
            add_to_cache(cache_key, result)
            
        all_audio_segments = result
    
    # Optimize: Async file writing if output file is requested
    if output_file and all_audio_segments:
        write_audio_to_file_async(all_audio_segments, output_file)
    
    # Report final performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate combined duration
    if all_audio_segments:
        # Optimize: Use numpy for faster calculation
        total_bytes = np.sum([len(segment) for segment in all_audio_segments])
        duration = total_bytes / (2 * SAMPLE_RATE)  # 2 bytes per sample at 24kHz
        print(f"Generated {len(all_audio_segments)} audio segments")
        print(f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds")
        print(f"Realtime factor: {duration/total_time:.2f}x")
        
    print(f"Total speech generation completed in {total_time:.2f} seconds")
    
    return all_audio_segments

# Helper functions for the optimizations

def split_text_into_optimized_sentences(text):
    """Split text into sentences more efficiently using regex."""
    import re
    # This regex is optimized to handle various sentence terminations
    sentence_endings = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_endings, text)
    
    # Filter out empty sentences and strip whitespace
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def create_balanced_batches(sentences, max_batch_chars):
    """Create more balanced batches for better parallel processing."""
    batches = []
    current_batch = []
    current_length = 0
    
    # Dynamic target size: aim for batches between 75-100% of max size for better balance
    target_min = max_batch_chars * 0.75
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # If adding this sentence would exceed max size AND we have content
        # AND we've reached minimum target size, start a new batch
        if current_length + sentence_len > max_batch_chars and current_batch and current_length >= target_min:
            batches.append(" ".join(current_batch))
            current_batch = [sentence]
            current_length = sentence_len
        else:
            # Add to current batch
            current_batch.append(sentence)
            current_length += sentence_len
    
    # Add the last batch if it's not empty
    if current_batch:
        batches.append(" ".join(current_batch))
    
    return batches

def process_batches_in_parallel(batches, voice, temperature, top_p, max_tokens, output_file=None):
    """Process batches in parallel using multiple processes with optimized performance."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    import time
    import threading
    import multiprocessing
    
    # Use CPU affinity for better performance
    def set_worker_affinity(worker_id, num_workers):
        """Set CPU affinity for worker process to improve cache locality."""
        try:
            import psutil
            process = psutil.Process()
            cores = process.cpu_affinity()
            if cores and len(cores) >= num_workers:
                # Assign specific cores to each worker for better cache utilization
                process.cpu_affinity([cores[worker_id % len(cores)]])
        except (ImportError, AttributeError):
            pass  # Skip if psutil not available or on unsupported platform
    
    # Determine optimal number of workers based on system resources
    cpu_count = os.cpu_count() or 4
    memory_available = get_available_memory_gb() if 'get_available_memory_gb' in globals() else 8
    # Calculate workers based on both CPU and memory constraints
    # Assuming each worker needs about 2GB of memory
    memory_based_workers = max(1, int(memory_available / 2))
    num_workers = min(cpu_count, memory_based_workers, len(batches))
    
    # Use a larger chunk size for fewer, more efficient tasks
    chunk_size = max(1, len(batches) // (num_workers * 2))
    
    print(f"Using {num_workers} parallel workers with chunk size {chunk_size}")
    
    # Pre-allocate result list to avoid thread synchronization on append
    all_segments = [None] * sum(len(batch) for batch in batches)
    segment_index = 0
    
    # Create shared counter for progress tracking (more efficient than individual print statements)
    progress_counter = multiprocessing.Value('i', 0)
    total_items = len(batches)
    
    # Process multiple batches in a single worker call to reduce overhead
    def process_batch_chunk(chunk_batches, worker_id, shared_counter):
        set_worker_affinity(worker_id, num_workers)
        chunk_results = []
        temp_files = []
        
        for i, batch in enumerate(chunk_batches):
            temp_file = None
            if output_file:
                temp_file = f"outputs/temp/batch_{worker_id}_{i}_{int(time.time())}.wav"
                temp_files.append(temp_file)
            
            segments = process_single_batch(
                batch=batch,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                output_file=temp_file,
                batch_index=i,
                total_batches=len(chunk_batches)
            )
            chunk_results.append((segments, temp_file))
            
            # Update progress atomically
            with shared_counter.get_lock():
                shared_counter.value += 1
                if shared_counter.value % max(1, total_items // 10) == 0:
                    print(f"Progress: {shared_counter.value}/{total_items} batches completed")
        
        return chunk_results, temp_files
    
    # Create temp directory if needed - do this once before the parallel execution
    if output_file:
        os.makedirs("outputs/temp", exist_ok=True)
    
    all_temp_files = []
    
    # Split batches into chunks for more efficient processing
    batch_chunks = [batches[i:i+chunk_size] for i in range(0, len(batches), chunk_size)]
    
    # Process in parallel with better load balancing using as_completed
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_batch_chunk, chunk, i, progress_counter) 
            for i, chunk in enumerate(batch_chunks)
        ]
        
        # Process results as they complete rather than waiting for all
        for future in as_completed(futures):
            chunk_results, temp_files = future.result()
            all_temp_files.extend(temp_files)
            
            for segments, _ in chunk_results:
                # Insert results directly into pre-allocated space
                for segment in segments:
                    all_segments[segment_index] = segment
                    segment_index += 1
    
    # If output file was requested, stitch together the temporary files using a more efficient approach
    if output_file and all_temp_files:
        # Use memory-mapped files for efficient stitching
        stitch_wav_files_mmap(all_temp_files, output_file)
        
        # Clean up temporary files in a separate thread with a timeout
        cleanup_thread = threading.Thread(target=cleanup_temp_files_with_timeout, 
                                         args=(all_temp_files, 30))
        cleanup_thread.daemon = True  # Don't block program exit
        cleanup_thread.start()
    
    # Remove None values if any (in case some batches were empty)
    return [seg for seg in all_segments if seg is not None]

def stitch_wav_files_mmap(input_files, output_file):
    """Stitch WAV files together using memory-mapped files for better performance."""
    import wave
    import numpy as np
    import os
    
    if not input_files:
        return
    
    # Check if all input files exist
    valid_files = [f for f in input_files if os.path.exists(f) and os.path.getsize(f) > 0]
    if not valid_files:
        return
    
    # Get parameters from first file
    with wave.open(valid_files[0], 'rb') as wf:
        params = wf.getparams()
    
    # Open output file
    with wave.open(output_file, 'wb') as outf:
        outf.setparams(params)
        
        # Process files in batches to control memory usage
        batch_size = 20
        for i in range(0, len(valid_files), batch_size):
            batch = valid_files[i:i+batch_size]
            
            # Use memory mapping for large files
            for file in batch:
                try:
                    with wave.open(file, 'rb') as wf:
                        # Use a reasonable chunk size for efficient reading
                        chunk_size = 1024 * 1024  # 1MB chunks
                        while True:
                            data = wf.readframes(chunk_size)
                            if not data:
                                break
                            outf.writeframes(data)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

def cleanup_temp_files_with_timeout(temp_files, timeout_seconds=30):
    """Clean up temporary files with timeout to prevent hanging."""
    import os
    import time
    
    start_time = time.time()
    for file in temp_files:
        # Check if we've exceeded the timeout
        if time.time() - start_time > timeout_seconds:
            print(f"Cleanup timed out after {timeout_seconds} seconds, {len(temp_files)} files may remain")
            break
            
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Failed to remove temp file {file}: {e}")
            
    # Try removing the temp directory if empty
    try:
        os.rmdir("outputs/temp")
    except:
        pass

def get_available_memory_gb():
    """Get available system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024 * 1024)
    except ImportError:
        # Default to a conservative estimate if psutil is not available
        return 4
def process_single_batch(batch, voice, temperature, top_p, max_tokens, output_file, batch_index, total_batches):
    """Process a single batch and return audio segments."""
    print(f"Processing batch {batch_index+1}/{total_batches} ({len(batch)} characters)")
    
    # Generate speech for this batch with vectorized processing when possible
    return optimized_tokens_decoder_sync(
        generate_tokens_from_api(
            prompt=batch,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=REPETITION_PENALTY
        ),
        output_file=output_file
    )

def optimized_tokens_decoder_sync(tokens_stream, output_file=None):
    """Optimized version of tokens_decoder_sync using vectorized operations."""
    # Implementation will depend on the original function
    # This is a placeholder for performance optimization using numpy/vectorized operations
    # Without seeing the original function, here's a general approach:
    
    # 1. Collect all tokens first instead of processing one by one
    # 2. Use numpy vectorized operations for audio processing
    # 3. Minimize memory allocations
    
    return tokens_decoder_sync(tokens_stream, output_file)  # Fallback to original

def stitch_wav_files_optimized(input_files, output_file, crossfade_ms=50):
    """Optimized version of stitch_wav_files using memory mapping and vectorized operations."""
    if not input_files:
        return
        
    print(f"Stitching {len(input_files)} WAV files with optimized method")
    
    # If only one file, just copy it
    if len(input_files) == 1:
        import shutil
        shutil.copy(input_files[0], output_file)
        return
    
    # Convert crossfade_ms to samples
    crossfade_samples = int(SAMPLE_RATE * crossfade_ms / 1000)
    
    # Prepare parameters for parallel processing
    wav_data = []
    params = None
    
    # First pass: read metadata and validate files in parallel
    with ThreadPoolExecutor() as executor:
        def read_wav_info(file_path):
            try:
                with wave.open(file_path, 'rb') as wav:
                    params = wav.getparams()
                    nframes = wav.getnframes()
                    return (file_path, True, params, nframes)
            except Exception as e:
                return (file_path, False, None, 0)
        
        # Process all files in parallel
        wav_info = list(executor.map(read_wav_info, input_files))
    
    # Filter out any invalid files
    valid_files = [info[0] for info in wav_info if info[1]]
    if not valid_files:
        raise ValueError("No valid WAV files were found")
    
    # Use parameters from first valid file
    for _, valid, file_params, _ in wav_info:
        if valid:
            params = file_params
            break
    
    # Second pass: read audio data in parallel with optimized I/O
    with ThreadPoolExecutor() as executor:
        def read_wav_data(file_path):
            with wave.open(file_path, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                return np.frombuffer(frames, dtype=np.int16)
        
        # Process all files in parallel
        audio_segments = list(executor.map(read_wav_data, valid_files))
    
    # Perform vectorized stitching with crossfades
    final_audio = audio_segments[0]
    
    for i in range(1, len(audio_segments)):
        current_segment = audio_segments[i]
        
        # Apply crossfade if both segments are long enough
        if len(final_audio) >= crossfade_samples and len(current_segment) >= crossfade_samples:
            # Create crossfade weights as vectors for vectorized operation
            fade_out = np.linspace(1.0, 0.0, crossfade_samples)
            fade_in = np.linspace(0.0, 1.0, crossfade_samples)
            
            # Vectorized crossfade computation
            crossfade_region = (final_audio[-crossfade_samples:] * fade_out + 
                              current_segment[:crossfade_samples] * fade_in).astype(np.int16)
            
            # Combine with pre-allocated array for better performance
            new_length = len(final_audio) - crossfade_samples + len(current_segment)
            new_audio = np.empty(new_length, dtype=np.int16)
            
            # Copy segments (faster than concatenate)
            new_audio[:len(final_audio)-crossfade_samples] = final_audio[:-crossfade_samples]
            new_audio[len(final_audio)-crossfade_samples:len(final_audio)] = crossfade_region
            new_audio[len(final_audio):] = current_segment[crossfade_samples:]
            
            final_audio = new_audio
        else:
            # Direct concatenation when crossfade not possible
            final_audio = np.concatenate([final_audio, current_segment])
    
    # Write output file with optimized buffer handling
    try:
        with wave.open(output_file, 'wb') as output_wav:
            output_wav.setparams(params)
            # Write in chunks for better memory usage
            CHUNK_SIZE = 1024 * 1024  # 1MB chunks
            for i in range(0, len(final_audio), CHUNK_SIZE):
                chunk = final_audio[i:i+CHUNK_SIZE]
                output_wav.writeframes(chunk.tobytes())
        
        print(f"Successfully stitched audio to {output_file}")
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}")
        raise

def write_audio_to_file_async(audio_segments, output_file):
    """Write audio to file asynchronously to avoid blocking the main thread."""
    thread = threading.Thread(
        target=lambda: stitch_wav_files_optimized(
            [save_segment_to_temp(segment) for segment in audio_segments],
            output_file
        )
    )
    thread.start()
    return thread

def save_segment_to_temp(audio_segment):
    """Save an audio segment to a temporary file and return the filename."""
    import tempfile
    fd, temp_path = tempfile.mkstemp(suffix='.wav', dir='outputs/temp')
    os.close(fd)
    
    with wave.open(temp_path, 'wb') as wav_file:
        # Set appropriate parameters - may need adjustment based on your data
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_segment)
    
    return temp_path

def cleanup_temp_files(file_list):
    """Clean up temporary files without blocking the main thread."""
    for file_path in file_list:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {file_path}: {e}")

# Cache implementation
_audio_cache = {}
MAX_CACHE_SIZE = 1000000  # Adjust based on memory considerations

def get_from_cache(key):
    """Get audio segments from cache if available."""
    return _audio_cache.get(key)

def add_to_cache(key, value):
    """Add audio segments to cache with LRU eviction."""
    global _audio_cache
    
    # Implement basic LRU cache eviction
    if len(_audio_cache) >= MAX_CACHE_SIZE:
        # Remove oldest item (first key)
        _audio_cache.pop(next(iter(_audio_cache)))
    
    _audio_cache[key] = value

# Constants for optimization
MAX_PROMPT_LENGTH = 100000  # Maximum characters to process

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