from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import time
import os
import sys
import gc
# Place this at the very top of your script, before any torch imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

# Check for CUDA availability *before* trying to use it
if torch.cuda.is_available():
    # Define snac_device early if CUDA is available
    snac_device = "cuda"
    # Reserve ~1 GB at startup (adjust as needed)
    reserve_mb = 2048
    try:
        # Use the now-defined snac_device
        dummy = torch.empty((reserve_mb * 1024 * 1024 // 4,), dtype=torch.float32, device=snac_device)
        del dummy
        torch.cuda.empty_cache()
        print(f"Reservoir of {reserve_mb} MiB reserved and released")
    except Exception as e:
        print(f"Warning: Failed to reserve GPU memory: {e}")
else:
    # Define snac_device if CUDA is not available
    snac_device = "mps" if torch.backends.mps.is_available() else "cpu"

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
# This definition is now redundant if CUDA is available, but harmless
# It ensures snac_device is defined if CUDA check somehow failed earlier
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if not IS_RELOADER:
    print(f"Using device: {snac_device}")
model = model.to(snac_device)

if torch.cuda.is_available():
    # Allow using the full memory of the selected device
    # Use device index 0 instead of the string "cuda"
    torch.cuda.set_per_process_memory_fraction(1.0, device=0)
    print("Set per-process memory fraction to 100% of GPU memory")

# Disable torch.compile as it requires Triton which isn't installed
# We'll use regular PyTorch optimization techniques instead
if not IS_RELOADER:
    print("Using standard PyTorch optimizations (torch.compile disabled)")

# Configure optimal CUDA settings for inference
if snac_device == "cuda":
    # Set benchmark mode for faster runtime on fixed input sizes
    torch.backends.cudnn.benchmark = True
    # Disable gradient operations entirely
    torch.set_grad_enabled(False)
    # Set memory allocation settings for lower fragmentation
    torch.cuda.empty_cache()
    if not IS_RELOADER:
        print(f"CUDA optimizations enabled: cudnn benchmark, gradients disabled")
        
    # Create primary and secondary CUDA streams for overlapping operations
    cuda_stream = torch.cuda.Stream()
    cuda_stream_secondary = torch.cuda.Stream()
    if not IS_RELOADER:
        print("Using multiple CUDA streams for concurrent processing")
else:
    cuda_stream = None
    cuda_stream_secondary = None

# Increase token cache size for better performance 
MAX_CACHE_SIZE = 25000  # Increased from 10000

# Try to use TorchScript for model optimization
try:
    if snac_device == "cuda":
        # Try to apply TorchScript to the model for better performance
        model = torch.jit.script(model)
        if not IS_RELOADER:
            print("TorchScript optimization applied to model")
except Exception as e:
    if not IS_RELOADER:
        print(f"TorchScript optimization skipped: {e}")

# Setup CUDA graphs for the model's decode operation if available
decode_graphed = None  # Disabled due to tensor size mismatch issues with dynamic inputs
# Keeping the code commented below for future reference
"""
if CUDA_GRAPHS_AVAILABLE and snac_device == "cuda":
    try:
        # We'll create a graphed version of the decode method
        sample_codes_0 = torch.zeros(1, 7, dtype=torch.int32, device=snac_device)
        sample_codes_1 = torch.zeros(1, 14, dtype=torch.int32, device=snac_device)
        sample_codes_2 = torch.zeros(1, 28, dtype=torch.int32, device=snac_device)
        sample_codes = [sample_codes_0, sample_codes_1, sample_codes_2]
        
        # Create graphed callable for the decode operation
        def decode_wrapper(codes):
            return model.decode(codes)
        
        decode_graphed = torch.cuda.make_graphed_callables(
            decode_wrapper, (sample_codes,)
        )
        
        if not IS_RELOADER:
            print("CUDA graphs enabled for decode operation")
    except Exception as e:
        if not IS_RELOADER:
            print(f"CUDA graphs setup failed: {e}")
        decode_graphed = None
"""

def convert_to_audio(multiframe, count):
    """
    Heavily optimized version of convert_to_audio that maximizes GPU utilization
    with vectorized operations and optimized memory transfers.
    """
    if len(multiframe) < 7:
        return None
  
    num_frames = len(multiframe) // 7
    
    # Periodic GPU memory management to reduce fragmentation
    if count % 1000 == 0 and snac_device == "cuda":
        torch.cuda.empty_cache()
    
    # Run in secondary stream if available to overlap with other operations
    stream_to_use = cuda_stream_secondary if cuda_stream_secondary is not None else cuda_stream
    stream_ctx = torch.cuda.stream(stream_to_use) if stream_to_use is not None else torch.no_grad()
    
    with stream_ctx, torch.inference_mode():
        # Transfer data to GPU in one batch
        frame_tensor = torch.tensor(multiframe[:num_frames*7], dtype=torch.int32, device=snac_device)
        
        # Pre-allocate output tensors
        codes_0 = torch.zeros(1, num_frames, dtype=torch.int32, device=snac_device)
        codes_1 = torch.zeros(1, num_frames * 2, dtype=torch.int32, device=snac_device)
        codes_2 = torch.zeros(1, num_frames * 4, dtype=torch.int32, device=snac_device)
        
        # Vectorized operations for far better GPU utilization
        # Reshape the input to make indexing more efficient
        frame_reshaped = frame_tensor.view(-1, 7)
        
        # Direct vectorized assignment
        codes_0[0, :] = frame_reshaped[:, 0]
        
        # For codes_1, we need indices 1 and 4 from each group of 7
        codes_1_indices = torch.tensor([1, 4], device=snac_device)
        codes_1_data = torch.index_select(frame_reshaped, 1, codes_1_indices)
        codes_1[0, ::2] = codes_1_data[:, 0]  # Even indices get index 1
        codes_1[0, 1::2] = codes_1_data[:, 1]  # Odd indices get index 4
        
        # For codes_2, we need indices 2, 3, 5 and 6
        codes_2_indices = torch.tensor([2, 3, 5, 6], device=snac_device)
        codes_2_data = torch.index_select(frame_reshaped, 1, codes_2_indices)
        codes_2[0, ::4] = codes_2_data[:, 0]     # Indices 0, 4, 8...
        codes_2[0, 1::4] = codes_2_data[:, 1]    # Indices 1, 5, 9...
        codes_2[0, 2::4] = codes_2_data[:, 2]    # Indices 2, 6, 10...
        codes_2[0, 3::4] = codes_2_data[:, 3]    # Indices 3, 7, 11...
        
        # Combine codes
        codes = [codes_0, codes_1, codes_2]
        
        # Fix: Validate each tensor separately to avoid shape mismatch errors
        valid_range = (
            torch.all(codes[0] >= 0) and torch.all(codes[0] < 4096) and
            torch.all(codes[1] >= 0) and torch.all(codes[1] < 4096) and
            torch.all(codes[2] >= 0) and torch.all(codes[2] < 4096)
        )
        
        if not valid_range:
            return None
            
        # Use CUDA graphs if available for the decode operation
        if decode_graphed is not None:
            audio_hat = decode_graphed(codes)
        else:
            audio_hat = model.decode(codes)
        
        # Wait for CUDA stream to complete before continuing
        if stream_to_use is not None:
            stream_to_use.synchronize()
        
        # Extract the relevant slice
        audio_slice = audio_hat[:, :, 2048:4096]
        
        # Process on GPU if possible
        if snac_device == "cuda":
            # Scale and convert to int16 directly on GPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Minimize CPU-GPU transfer by only moving the final result
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            # For non-CUDA devices, use the original approach
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
    
    return audio_bytes

# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a single global cache for token processing
token_id_cache = {}

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
    
    # Use different thresholds for first chunk vs. subsequent chunks
    min_frames_first = 7  # Just one chunk (7 tokens) for first audio - ultra-low latency
    min_frames_subsequent = 28  # Standard minimum (4 chunks of 7 tokens) after first audio
    # Increased ideal frame size for larger GPU batches
    ideal_frames = 98  # Ideal standard frame size (14x7 window), increased from 49
    process_every_n = 7  # Process every 7 tokens (standard for Orpheus model) - unchanged
    
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
                # For subsequent chunks, use original processing with proper batching
                if count % process_every_n == 0:
                    # Use same prioritization logic as before
                    if len(buffer) >= ideal_frames:
                        buffer_to_proc = buffer[-ideal_frames:]
                    elif len(buffer) >= min_frames_subsequent:
                        buffer_to_proc = buffer[-min_frames_subsequent:]
                    else:
                        continue
                    
                    # Debug output to help diagnose issues
                    if count % 28 == 0:
                        print(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}")
                    
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
            
    # Final special case: even if we don't have minimum frames, try to process
    # what we have by padding with silence tokens that won't affect the audio
    elif len(buffer) >= process_every_n:
        # Pad to minimum frame requirement with copies of the final token
        # This is more continuous than using unrelated tokens from the beginning
        last_token = buffer[-1]
        padding_needed = min_frames_subsequent - len(buffer)
        
        # Create a padding array of copies of the last token
        # This maintains continuity much better than circular buffering
        padding = [last_token] * padding_needed
        padded_buffer = buffer + padding
        
        print(f"Processing final partial frame: {len(buffer)} tokens + {padding_needed} repeated-token padding")
        audio_samples = convert_to_audio(padded_buffer, count)
        if audio_samples is not None:
            yield audio_samples
# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    """Optimized synchronous decoder with larger queue and parallel processing"""
    # Use a larger queue for CUDA devices to maximize GPU utilization
    # Increased queue size for CUDA
    max_queue_size = 128 if snac_device == "cuda" else 24  # Increased from 64/16
    audio_queue = queue.Queue(maxsize=max_queue_size)
    
    # Collect tokens in larger batches for higher throughput on CUDA
    # Increased batch size for CUDA
    batch_size = 64 if snac_device == "cuda" else 12       # Increased from 32/8
    
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

    # Run the producer in a separate thread
    thread = threading.Thread(target=run_async)
    thread.daemon = True  # Allow the thread to be terminated when the main thread exits
    thread.start()

    # Use larger buffer for final audio assembly
    buffer_size = 10 if snac_device == "cuda" else 5       # Increased from 5
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
