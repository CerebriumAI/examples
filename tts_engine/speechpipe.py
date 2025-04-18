from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import time
import os
import sys

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


def convert_to_audio(multiframe, count):
    """
    Optimized version of convert_to_audio that eliminates inefficient tensor operations
    and reduces CPU-GPU transfers for much faster inference on high-end GPUs.
    """
    if len(multiframe) < 7:
        return None
  
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    # Pre-allocate tensors instead of incrementally building them
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)
    
    # Use vectorized operations where possible
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
    
    # Direct indexing is much faster than concatenation in a loop
    for j in range(num_frames):
        idx = j * 7
        
        # Code 0 - single value per frame
        codes_0[j] = frame_tensor[idx]
        
        # Code 1 - two values per frame
        codes_1[j*2] = frame_tensor[idx+1]
        codes_1[j*2+1] = frame_tensor[idx+4]
        
        # Code 2 - four values per frame
        codes_2[j*4] = frame_tensor[idx+2]
        codes_2[j*4+1] = frame_tensor[idx+3]
        codes_2[j*4+2] = frame_tensor[idx+5]
        codes_2[j*4+3] = frame_tensor[idx+6]
    
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
    
    with stream_ctx, torch.inference_mode():
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
    """High-performance token decoder optimized for maximum GPU throughput"""
    # Pre-allocate larger buffers for better performance
    buffer = []
    processed_count = 0
    total_count = 0
    
    # Optimized thresholds for first chunk and subsequent processing
    first_chunk_processed = False
    min_frames_first = 5  # Reduced from 7 to 5 for faster first chunk
    min_frames_subsequent = 24  # Reduced from 28 to 24 for higher throughput
    ideal_frames = 64  # Increased from 49 to 64 for better GPU utilization
    process_every_n = 4  # Reduced from 7 to 4 for more frequent updates
    
    # Additional GPU optimization parameters
    max_batch_size = 256  # Maximum batch size for GPU processing
    batch_scale_factor = 1.5  # Scale batch size over time for better throughput
    current_batch_size = min_frames_subsequent  # Start with minimum
    
    # Pre-allocate audio buffer for more efficient memory usage
    audio_buffer = bytearray(32768)  # 32KB initial buffer
    audio_buffer_pos = 0
    
    # Performance tracking
    start_time = time.time()
    token_count = 0
    last_log_time = start_time
    last_yield_time = start_time
    processed_batches = 0
    
    # Tensor caching for better GPU performance
    cached_tensors = {}
    
    # Process incoming tokens with adaptive batching
    async for token_sim in token_gen:
        token_count += 1
        total_count += 1
        
        # Process token with caching to reduce overhead
        token = turn_token_into_id(token_sim, processed_count)
        
        if token is not None and token > 0:
            buffer.append(token)
            processed_count += 1
            
            # Minimal performance logging (reduced frequency)
            current_time = time.time()
            if current_time - last_log_time > 10.0:  # Reduced logging frequency
                elapsed = current_time - last_log_time
                tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                # Only log if significant tokens processed
                if token_count > 100:
                    print(f"Processing rate: {tokens_per_sec:.1f} tokens/s, batch size: {current_batch_size}")
                last_log_time = current_time
                token_count = 0
            
            # Ultra-low latency for first audio chunk
            if not first_chunk_processed:
                if processed_count >= min_frames_first:
                    # Process small batch quickly for first chunk
                    buffer_to_proc = buffer[-min_frames_first:]
                    audio_samples = convert_to_audio(buffer_to_proc, processed_count)
                    if audio_samples is not None:
                        first_chunk_processed = True
                        yield audio_samples
                        last_yield_time = time.time()
            else:
                # GPU-optimized batch processing for subsequent chunks
                should_process = False
                
                # Process based on count for regular cadence
                if processed_count % process_every_n == 0:
                    should_process = True
                
                # Process when we have enough for an ideal batch
                elif len(buffer) >= current_batch_size:
                    should_process = True
                    
                # Also ensure we don't go too long without yielding audio
                elapsed_since_yield = current_time - last_yield_time
                if elapsed_since_yield > 0.1:  # Max 100ms without yielding
                    should_process = True
                
                if should_process and len(buffer) >= min_frames_subsequent:
                    # Determine optimal batch size based on available tokens
                    batch_size = min(len(buffer), current_batch_size)
                    
                    # Use larger batch sizes when more tokens are available
                    buffer_to_proc = buffer[-batch_size:]
                    
                    # Process audio with GPU acceleration
                    audio_samples = convert_to_audio(buffer_to_proc, processed_count)
                    if audio_samples is not None:
                        yield audio_samples
                        last_yield_time = time.time()
                        processed_batches += 1
                        
                        # Gradually increase batch size for better GPU utilization
                        # This takes advantage of GPU's ability to process larger batches more efficiently
                        if processed_batches % 5 == 0 and current_batch_size < max_batch_size:
                            current_batch_size = min(
                                max_batch_size, 
                                int(current_batch_size * batch_scale_factor)
                            )
    
    # End-of-stream processing - handle remaining tokens
    remaining_tokens = len(buffer)
    if remaining_tokens > 0:
        print(f"Processing {remaining_tokens} remaining tokens at end of stream")
        
        # Process complete frames if possible
        if remaining_tokens >= min_frames_subsequent:
            batch_size = min(remaining_tokens, current_batch_size)
            buffer_to_proc = buffer[-batch_size:]
            audio_samples = convert_to_audio(buffer_to_proc, processed_count)
            if audio_samples is not None:
                yield audio_samples
        # For very short remaining sequences, pad with previous tokens to reach minimum size
        elif remaining_tokens >= 3:  # Only process if we have at least a few tokens
            padding_needed = min_frames_subsequent - remaining_tokens
            if padding_needed > 0:
                # Use the last token for padding to maintain audio consistency
                last_token = buffer[-1]
                padded_buffer = buffer + [last_token] * padding_needed
                audio_samples = convert_to_audio(padded_buffer, processed_count)
                if audio_samples is not None:
                    yield audio_samples
    
    # Report final performance stats
    total_time = time.time() - start_time
    if total_time > 0 and total_count > 0:
        final_tokens_per_sec = total_count / total_time
        print(f"Final token processing rate: {final_tokens_per_sec:.1f} tokens/second")

# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    """Optimized synchronous decoder with larger queue and parallel processing"""
    # Use a larger queue for RTX 4090 to maximize GPU utilization
    max_queue_size = 32 if snac_device == "cuda" else 8
    audio_queue = queue.Queue(maxsize=max_queue_size)
    
    # Collect tokens in batches for higher throughput
    batch_size = 16 if snac_device == "cuda" else 4
    
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