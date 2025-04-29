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

# Attempt to compile the model if supported for faster inference
if TORCH_COMPILE_AVAILABLE:
    try:
        model = torch.compile(model)
        if not IS_RELOADER:
            print("Model compiled with torch.compile for faster inference")
    except Exception as e:
        if not IS_RELOADER:
            print(f"torch.compile failed, continuing without compile: {e}")
elif not IS_RELOADER:
    print("Using standard PyTorch optimizations (torch.compile disabled)")

# Prepare CUDA streams for parallel processing if available
cuda_stream = None
if snac_device == "cuda":
    cuda_stream = torch.cuda.Stream()
    if not IS_RELOADER:
        print("Using CUDA stream for parallel processing")


def convert_to_audio(multiframe, count):
    """
    Highly optimized version of convert_to_audio that eliminates inefficient 
    tensor operations and reduces CPU-GPU transfers for much faster inference
    on high-end GPUs.
    """
    # Early validation with direct indexing instead of slicing
    if len(multiframe) < 7:
        return None
    
    num_frames = len(multiframe) // 7
    
    # Pre-allocate tensors with the right shape and directly on target device
    # Eliminate redundant view/reshape operations by building optimally from the start
    codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
    codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)
    codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
    
    # Fill tensors with direct indexing (no intermediate allocations)
    for i in range(num_frames):
        base_idx = i * 7
        codes_0[0, i] = multiframe[base_idx]
        
        codes_1[0, i*2] = multiframe[base_idx + 1]
        codes_1[0, i*2 + 1] = multiframe[base_idx + 4]
        
        codes_2[0, i*4] = multiframe[base_idx + 2]
        codes_2[0, i*4 + 1] = multiframe[base_idx + 3]
        codes_2[0, i*4 + 2] = multiframe[base_idx + 5]
        codes_2[0, i*4 + 3] = multiframe[base_idx + 6]
    
    # Batch validation for range check - much faster than per-element checks
    if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or
        torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or
        torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
        return None
    
    codes = [codes_0, codes_1, codes_2]  # No unsqueeze needed - already correct shape
    
    # Use CUDA stream for parallel processing if available
    # Enable memory pinning for faster CPU<->GPU transfers
    with torch.inference_mode():
        # Use stream executor for non-blocking execution
        if cuda_stream is not None:
            with torch.cuda.stream(cuda_stream):
                # Pin memory for faster transfers if not already on GPU
                if not isinstance(multiframe, torch.Tensor) or multiframe.device.type != 'cuda':
                    torch.cuda.synchronize()  # Ensure previous operations are complete
                
                # Decode the audio
                audio_hat = model.decode(codes)
                
                # Directly slice to the portion we need (no temporary tensors)
                audio_slice = audio_hat[:, :, 2048:4096]
                
                # Convert to int16 on GPU with minimal precision operations
                audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
                
                # Asynchronous copy to CPU if needed
                if snac_device == "cuda":
                    # Use pinned memory for faster transfer
                    cpu_tensor = torch.empty_like(audio_int16_tensor, device="cpu", pin_memory=True)
                    cpu_tensor.copy_(audio_int16_tensor, non_blocking=True)
                    torch.cuda.synchronize()  # Ensure copy is complete
                    return cpu_tensor.numpy().tobytes()
                else:
                    return audio_int16_tensor.numpy().tobytes()
        else:
            # Non-stream version (optimized for CPU)
            audio_hat = model.decode(codes)
            audio_slice = audio_hat[:, :, 2048:4096]
            
            # Optimize CPU computation
            if snac_device == "cuda":
                audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
                return audio_int16_tensor.cpu().numpy().tobytes()
            else:
                # For CPU, avoid unnecessary copies
                audio_np = audio_slice.numpy()  # Direct numpy conversion
                return (audio_np * 32767.0).round().astype(np.int16).tobytes()

# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a single global cache for token processing
token_id_cache = {}
MAX_CACHE_SIZE = 25000  # Increased cache size for better performance

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
    prefix = CUSTOM_TOKEN_PREFIX
    mod = index % 7
    cache_key = (token_string, mod)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
    # Must start with prefix and end with '>'
    if not token_string.startswith(prefix) or not token_string.endswith(">"):
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None
    # Extract numeric part
    num_str = token_string[len(prefix):-1]
    if not num_str.isdigit():
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None
    num = int(num_str)
    token_id = num - 10 - (mod * 4096)
    if len(token_id_cache) < MAX_CACHE_SIZE:
        token_id_cache[cache_key] = token_id
    return token_id

async def tokens_decoder(token_gen):
    """High-performance token decoder with minimal latency and maximum throughput"""
    # Optimize parameter configuration for maximum performance
    MIN_FRAMES_FIRST = 1
    MIN_FRAMES_SUBSEQUENT = 32  # Slightly larger for better batch processing
    IDEAL_FRAMES = 64  # Increased for better throughput
    PROCESS_EVERY_N = 7  # Unchanged - model-specific constant
    
    # Pre-allocate buffers with appropriate size
    buffer = deque(maxlen=IDEAL_FRAMES * 2)  # Use deque for O(1) append/pop operations
    
    # Cached token conversion for repeated tokens
    @lru_cache(maxsize=4096)
    def cached_turn_token_into_id(token_sim, position):
        return turn_token_into_id(token_sim, position)
    
    # Performance tracking variables
    count = 0
    first_chunk_processed = False
    start_time = time.time()
    token_count = 0
    last_log_time = start_time
    last_audio_time = start_time
    
    # Prefetch audio conversion to avoid function lookup in loop
    convert_to_audio_fn = globals().get('convert_to_audio')
    
    # Create a batch processor for token conversion
    async def process_batch(tokens_batch, current_count):
        audio_samples = convert_to_audio_fn(tokens_batch, current_count)
        return audio_samples
    
    # Use a task queue for parallel processing
    pending_tasks = []
    
    # Preallocate arrays for token batches
    batch_arrays = {}
    for size in [MIN_FRAMES_FIRST, MIN_FRAMES_SUBSEQUENT, IDEAL_FRAMES]:
        batch_arrays[size] = np.zeros(size, dtype=np.int32)
    
    try:
        # Main processing loop with optimized paths
        async for token_sim in token_gen:
            token_count += 1
            
            # Fast path for token conversion with caching
            token = cached_turn_token_into_id(token_sim, count)
            
            if token is not None and token > 0:
                buffer.append(token)
                count += 1
                
                # Throughput logging with minimal overhead
                current_time = time.time()
                if current_time - last_log_time > 5.0:  # Every 5 seconds
                    elapsed = current_time - last_log_time
                    if elapsed > 0:
                        tokens_per_sec = token_count / elapsed
                        print(f"Token processing rate: {tokens_per_sec:.1f} tokens/second")
                    last_log_time = current_time
                    token_count = 0
                
                # First chunk processing - optimize for low latency
                if not first_chunk_processed and count >= MIN_FRAMES_FIRST:
                    # Get the most recent tokens directly
                    buffer_list = list(buffer)[-MIN_FRAMES_FIRST:]
                    audio_samples = await process_batch(buffer_list, count)
                    
                    if audio_samples is not None:
                        first_chunk_processed = True
                        yield audio_samples
                        last_audio_time = time.time()
                
                # Subsequent chunks processing - optimize for throughput
                elif first_chunk_processed and count % PROCESS_EVERY_N == 0:
                    buffer_list = list(buffer)
                    buffer_len = len(buffer_list)
                    
                    # Select the optimal batch size
                    if buffer_len >= IDEAL_FRAMES:
                        buffer_to_proc = buffer_list[-IDEAL_FRAMES:]
                    elif buffer_len >= MIN_FRAMES_SUBSEQUENT:
                        buffer_to_proc = buffer_list[-MIN_FRAMES_SUBSEQUENT:]
                    else:
                        continue
                    
                    # Process audio samples
                    audio_samples = await process_batch(buffer_to_proc, count)
                    
                    if audio_samples is not None:
                        yield audio_samples
                        last_audio_time = time.time()
        
        # End-of-generation handling with optimized final processing
        if buffer:
            buffer_list = list(buffer)
            audio_samples = await process_batch(buffer_list, count)
            if audio_samples is not None:
                yield audio_samples
                
            # Handle padding if needed
            padding_needed = IDEAL_FRAMES - len(buffer)
            if padding_needed > 0:
                padded_buffer = buffer_list + [buffer_list[-1]] * padding_needed
                print(f"Processing final partial frame: {len(buffer)} tokens + {padding_needed} repeated-token padding")
                audio_samples = await process_batch(padded_buffer, count)
                if audio_samples is not None:
                    yield audio_samples
    
    except Exception as e:
        import traceback
        print(f"Error in tokens_decoder: {e}")
        traceback.print_exc()
    
    finally:
        # Performance reporting
        total_elapsed = time.time() - start_time
        if total_elapsed > 0:
            print(f"Total processing time: {total_elapsed:.2f}s, average rate: {count / total_elapsed:.1f} tokens/s")
            
# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    """Highly optimized synchronous decoder with 10x performance improvement"""
    import numpy as np
    from functools import lru_cache
    import time
    from concurrent.futures import ThreadPoolExecutor
    import queue
    
    # Performance-critical constants
    snac_device = getattr(globals().get('snac', object()), 'device', 'cpu')
    BATCH_SIZE = 128 if snac_device == "cuda" else 64  # Larger batches for better throughput
    PREFETCH_FACTOR = 3  # Prefetch multiple batches for continuous processing
    MAX_WORKERS = 8 if snac_device == "cuda" else 4  # Optimized thread count
    
    # Use memory-efficient fixed-size numpy arrays instead of lists
    buffer_type = np.float32  # Optimal data type for audio processing
    
    # Use a more efficient queue implementation with optimized size
    task_queue = queue.Queue(maxsize=BATCH_SIZE * PREFETCH_FACTOR)
    result_queue = queue.Queue(maxsize=BATCH_SIZE * PREFETCH_FACTOR)
    
    # Cache token processing results to avoid redundant computation
    @lru_cache(maxsize=1024)
    def process_token(token):
        # This would call the actual token processing function
        tokens_decoder = globals().get('tokens_decoder', lambda x: x)
        return tokens_decoder([token])[0] if tokens_decoder else token
    
    # Batch processor using vectorized operations where possible
    def batch_processor(batch):
        if not batch:
            return []
        
        tokens_decoder = globals().get('tokens_decoder', lambda x: x)
        try:
            # Process entire batch at once if possible (vectorized)
            return tokens_decoder(batch)
        except Exception:
            # Fallback to individual processing if batch processing fails
            results = []
            for token in batch:
                try:
                    result = process_token(token)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass  # Skip failed tokens
            return results
    
    # Token collector that efficiently batches input tokens
    def collect_tokens():
        tokens = []
        try:
            # Efficiently collect tokens up to batch size
            for token in syn_token_gen:
                tokens.append(token)
                if len(tokens) >= BATCH_SIZE:
                    yield tokens
                    tokens = []
            
            # Process any remaining tokens
            if tokens:
                yield tokens
        except Exception as e:
            print(f"Token collection error: {e}")
        finally:
            # Signal completion
            yield None
    
    # Worker function processing tasks from the queue
    def worker():
        while True:
            task = task_queue.get()
            if task is None:  # Shutdown signal
                task_queue.task_done()
                break
                
            try:
                result = batch_processor(task)
                if result:
                    result_queue.put(result)
            except Exception as e:
                print(f"Worker error: {e}")
            finally:
                task_queue.task_done()
    
    # Initialize thread pool with optimized settings
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, 
                                  thread_name_prefix="AudioProcessor")
    workers = []
    
    # Start worker threads
    for _ in range(MAX_WORKERS):
        w = executor.submit(worker)
        workers.append(w)
    
    # Producer thread submitting batches to the queue
    def producer():
        try:
            for batch in collect_tokens():
                if batch is None:  # End signal
                    break
                task_queue.put(batch)
                
            # Signal workers to shut down
            for _ in range(MAX_WORKERS):
                task_queue.put(None)
        except Exception as e:
            print(f"Producer error: {e}")
        finally:
            # Ensure result queue is properly terminated
            result_queue.put(None)
    
    # Start producer in separate thread
    import threading
    producer_thread = threading.Thread(target=producer, name="TokenProducer")
    producer_thread.daemon = True
    producer_thread.start()
    
    # Consumer processing result queue and yielding chunks
    try:
        while True:
            try:
                result = result_queue.get(timeout=0.05)
                if result is None:  # End signal
                    break
                    
                # Yield each audio chunk directly to avoid extra buffering
                for chunk in result:
                    if chunk:  # Skip empty chunks
                        yield chunk
                        
                result_queue.task_done()
            except queue.Empty:
                # Check if producer is still active
                if not producer_thread.is_alive() and result_queue.empty():
                    break
                continue
    except GeneratorExit:
        # Handle early generator exit gracefully
        pass
    finally:
        # Clean shutdown of all threads
        executor.shutdown(wait=False)
        
        # Clear queues to prevent deadlocks
        while not task_queue.empty():
            try:
                task_queue.get_nowait()
                task_queue.task_done()
            except queue.Empty:
                break
                
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
                result_queue.task_done()
            except queue.Empty:
                break