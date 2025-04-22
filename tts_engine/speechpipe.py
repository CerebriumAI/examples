from snac import SNAC
import numpy as np
import torch
# Enable TF32 and cuDNN benchmarking for faster GPU inference
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
import asyncio
import threading
import queue
import time
import os
import sys
import contextlib

# Helper to detect if running in Uvicorn's reloader (same as in inference.py)
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

# Set a flag to avoid repeat messages
IS_RELOADER = is_reloader_process()

# Try to enable torch.compile if PyTorch 2.0+ is available
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, 'compile'):
        TORCH_COMPILE_AVAILABLE = True
        if not IS_RELOADER:
            print("PyTorch 2.0+ detected, torch.compile is available")
except:
    pass

# Try to enable CUDA graphs if available
CUDA_GRAPHS_AVAILABLE = False
try:
    if torch.cuda.is_available() and hasattr(torch.cuda, 'make_graphed_callables'):
        CUDA_GRAPHS_AVAILABLE = True
        if not IS_RELOADER:
            print("CUDA graphs support is available")
except:
    pass

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if not IS_RELOADER:
    print(f"Using device: {snac_device}")
model = model.to(snac_device)
# Cast model to half precision if using CUDA for faster inference
if snac_device == "cuda":
    model = model.half()
    if not IS_RELOADER:
        print("SNAC model cast to FP16 for faster GPU inference")

# Disable torch.compile as it requires Triton which isn't installed
# We'll use regular PyTorch optimization techniques instead
if not IS_RELOADER:
    print("Using standard PyTorch optimizations (torch.compile disabled)")

# Prepare CUDA streams for parallel processing if available
cuda_stream = None
if snac_device == "cuda":
    cuda_stream = torch.cuda.Stream()
    if not IS_RELOADER:
        print("Using CUDA stream for parallel processing")

if TORCH_COMPILE_AVAILABLE and snac_device == "cuda":
    try:
        model = torch.compile(model)
        if not IS_RELOADER:
            print(" SNAC model compiled for optimized GPU inference")
    except Exception as e:
        print(f" torch.compile failed: {e}")

def convert_to_audio(multiframe, count):
    """
    Optimized version of convert_to_audio that eliminates inefficient tensor operations
    and reduces CPU-GPU transfers for much faster inference on high-end GPUs.
    """
    if len(multiframe) < 7:
        return None
  
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    # Vectorize frame slicing: reshape and slice
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
    frame_mat = frame_tensor.view(num_frames, 7)
    codes_0 = frame_mat[:, 0]
    # Two values per frame: columns 1 and 4
    codes_1 = frame_mat[:, [1, 4]].reshape(-1)
    # Four values per frame: columns 2,3,5,6
    codes_2 = frame_mat[:, [2, 3, 5, 6]].reshape(-1)
    
    # Reshape codes into expected format
    codes = [
        codes_0.unsqueeze(0), 
        codes_1.unsqueeze(0), 
        codes_2.unsqueeze(0)
    ]
    
    # Check tokens are in valid range
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    # Use CUDA stream for parallel processing if available
    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()
    autocast_ctx = torch.cuda.amp.autocast() if snac_device == "cuda" else contextlib.nullcontext()
    
    with stream_ctx, torch.inference_mode(), autocast_ctx:
        # Decode the audio
        audio_hat = model.decode(codes)
        
        # Extract the relevant slice and efficiently convert to bytes
        # Keep data on GPU as long as possible
        audio_slice = audio_hat[:, :, 2048:4096]
        
        # Process on GPU if possible, with minimal data transfer
        if snac_device == "cuda":
            # Scale directly on GPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Only transfer the final result to CPU
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            # For non-CUDA devices, fall back to the original approach
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
    return audio_bytes

# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a single global cache for token processing
token_id_cache = {}
MAX_CACHE_SIZE = 10000  # Increased cache size for better performance

def turn_token_into_id(token_string, index):
    """
    Optimized token-to-ID conversion with caching.
    This is the definitive implementation used by both inference.py and speechpipe.py.
    
    Args:
        token_string: The token string to convert
        index: Position index used for token offset calculation
        
    Returns:
        int: Token ID if valid, None otherwise
    """
    # Check cache first (significant speedup for repeated tokens)
    cache_key = (token_string, index % 7)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
        
    # Early rejection for obvious non-matches
    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None
        
    # Process token
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    last_token = token_string[last_token_start:]
    
    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None
        
    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)
        
        # Cache the result if it's valid
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id
            
        return token_id
    except (ValueError, IndexError):
        return None

async def tokens_decoder(token_gen):
    """Optimized token decoder with early first-chunk processing for lower latency"""
    buffer = []
    count = 0
    
    # Track if first chunk has been processed
    first_chunk_processed = False
    
    # Use larger batch sizes for higher GPU throughput
    min_frames_first = 16   # First chunk: larger for better GPU utilization
    min_frames_subsequent = 64  # Subsequent chunks: larger for batch processing
    ideal_frames = 128  # Ideal frame size for high-throughput streaming
    process_every_n = 32  # Process every 32 tokens
    
    start_time = time.time()
    token_count = 0
    last_log_time = start_time
    
    async for token_sim in token_gen:
        token_count += 1
        
        # Use the unified turn_token_into_id which already handles caching
        token = turn_token_into_id(token_sim, count)
        
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Log throughput periodically
            current_time = time.time()
            if current_time - last_log_time > 5.0:  # Every 5 seconds
                elapsed = current_time - last_log_time
                if elapsed > 0:
                    recent_tokens = token_count
                    tokens_per_sec = recent_tokens / elapsed
                    print(f"Token processing rate: {tokens_per_sec:.1f} tokens/second")
                last_log_time = current_time
                token_count = 0
            
            # Different processing logic based on whether first chunk has been processed
            if not first_chunk_processed:
                # Process first chunk as soon as possible for minimal latency
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]
                    
                    # Process the first chunk of audio for immediate feedback
                    print(f"Processing first audio chunk with {len(buffer_to_proc)} tokens for low latency")
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True  # Mark first chunk as processed
                        yield audio_samples
            else:
                # For subsequent chunks, process every 56 tokens for larger batches
                if count % process_every_n == 0:
                    if len(buffer) >= ideal_frames:
                        buffer_to_proc = buffer[-ideal_frames:]
                    elif len(buffer) >= min_frames_subsequent:
                        buffer_to_proc = buffer[-min_frames_subsequent:]
                    else:
                        continue
                    # Minimal debug output for speed
                    # Process the tokens
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples
    
    # CRITICAL: End-of-generation handling - process all remaining frames
    # Process remaining complete frames (ideal size)
    if len(buffer) >= ideal_frames:
        buffer_to_proc = buffer[-ideal_frames:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
    # Process any additional complete frames (minimum size)
    elif len(buffer) >= min_frames_subsequent:
        buffer_to_proc = buffer[-min_frames_subsequent:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
    # Final: pad to minimum if enough for one process_every_n
    elif len(buffer) >= process_every_n:
        last_token = buffer[-1]
        padding_needed = min_frames_subsequent - len(buffer)
        padding = [last_token] * padding_needed
        padded_buffer = buffer + padding
        audio_samples = convert_to_audio(padded_buffer, count)
        if audio_samples is not None:
            yield audio_samples
# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    """Optimized synchronous decoder with larger queue and parallel processing"""
    # Use a larger queue for RTX 4090 to maximize GPU utilization
    max_queue_size = 128 if snac_device == "cuda" else 8
    audio_queue = queue.Queue(maxsize=max_queue_size)
    
    # Collect tokens in batches for higher throughput
    batch_size = 64 if snac_device == "cuda" else 4
    
    # Convert the synchronous token generator into an async generator with batching
    async def async_token_gen():
        token_batch = []
        for token in syn_token_gen:
            token_batch.append(token)
            # Process in batches for efficiency
            if len(token_batch) >= batch_size:
                for t in token_batch:
                    yield t
                token_batch = []
        # Process any remaining tokens
        for t in token_batch:
            yield t

    async def async_producer():
        # Start timer for performance logging
        start_time = time.time()
        chunk_count = 0
        
        try:
            # Process audio chunks from the token decoder
            async for audio_chunk in tokens_decoder(async_token_gen()):
                if audio_chunk:  # Validate audio chunk before adding to queue
                    audio_queue.put(audio_chunk)
                    chunk_count += 1
                    
                    # Log performance stats periodically
                    if chunk_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"Generated {chunk_count} chunks in {elapsed:.2f}s ({chunk_count/elapsed:.2f} chunks/sec)")
        except Exception as e:
            print(f"Error in audio producer: {e}")
            import traceback
            traceback.print_exc()
        finally:    
            # Signal completion
            print("Audio producer completed - finalizing all chunks")
            audio_queue.put(None)  # Sentinel

    def run_async():
        asyncio.run(async_producer())

    # Use a higher priority thread for RTX 4090 to ensure it stays fed with work
    thread = threading.Thread(target=run_async)
    thread.daemon = True  # Allow the thread to be terminated when the main thread exits
    thread.start()

    # Use larger buffer for final audio assembly
    buffer_size = 5
    audio_buffer = []
    
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        audio_buffer.append(audio)
        # Yield buffered audio chunks for smoother playback
        if len(audio_buffer) >= buffer_size:
            for chunk in audio_buffer:
                yield chunk
            audio_buffer = []
    
    # Yield any remaining audio in the buffer
    for chunk in audio_buffer:
        yield chunk

    thread.join()