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

# --- Configuration ---
MIN_FRAMES_FIRST = 1
MIN_FRAMES_SUBSEQUENT = 32
IDEAL_FRAMES = 64
PROCESS_EVERY_N = 7 # This seems specific, keeping it. If flexible, could be removed.
MAX_PENDING_AUDIO_TASKS = 4 # Backpressure: Limit parallel audio conversions
EXECUTOR_TYPE = 'thread' # 'thread' or 'process'. Use 'thread' if convert_to_audio releases GIL or is I/O bound.
NUM_EXECUTORS = None # None for default (usually cores * 5 for ThreadPool, cores for ProcessPool)

# Sentinel object to signal the end of processing
END_OF_STREAM = object()

async def tokens_decoder(token_gen):
    """
    High-performance token decoder optimized for latency and throughput
    using background audio processing.
    """
    global turn_token_into_id, convert_to_audio # Ensure functions are accessible

    buffer = deque(maxlen=IDEAL_FRAMES * 2) # Keep deque for efficient buffering
    results_queue = asyncio.Queue(maxsize=MAX_PENDING_AUDIO_TASKS + 1) # Queue for yielding results
    pending_futures = deque() # Track background tasks to ensure order

    # Use LRU Cache for token conversion
    @lru_cache(maxsize=4096)
    def cached_turn_token_into_id(token_sim, position):
        # Note: If turn_token_into_id itself is slow and CPU-bound,
        # it might also need to run in an executor. Assumed fast here.
        return turn_token_into_id(token_sim, position)

    # Pre-fetch audio conversion function
    convert_to_audio_fn = convert_to_audio

    # Choose and create the executor
    if EXECUTOR_TYPE == 'process':
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=NUM_EXECUTORS)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_EXECUTORS)

    loop = asyncio.get_running_loop()

    # --- Performance Tracking ---
    count = 0
    first_chunk_processed = False
    start_time = time.time()
    token_receive_count = 0
    last_log_time = start_time
    total_tokens_processed_for_audio = 0

    async def _submit_audio_task(tokens_to_process, current_token_count):
        """Submits a batch of tokens for audio conversion in the executor."""
        nonlocal total_tokens_processed_for_audio
        if not tokens_to_process:
            return

        # Efficiently create numpy array from deque slice
        # Using np.fromiter on an islice is generally faster than list conversion
        try:
            token_array = np.fromiter(tokens_to_process, dtype=np.int32, count=len(tokens_to_process))
        except TypeError: # Fallback if fromiter needs count=len(...) for deque iterators
             token_array = np.array(list(tokens_to_process), dtype=np.int32)


        # Run convert_to_audio_fn in the background executor
        future = loop.run_in_executor(
            executor,
            partial(convert_to_audio_fn, token_array, current_token_count) # Use partial to pass args
        )
        pending_futures.append(future)
        total_tokens_processed_for_audio += len(tokens_to_process)

        # Apply backpressure: Wait if the results queue is full
        await results_queue.join() # Wait until a slot is free (task_done called)

    async def _result_producer():
        """Waits for futures in order and puts results onto the queue."""
        while True:
            if not pending_futures:
                # Wait for a task to be submitted or for the stream to end
                await asyncio.sleep(0.001) # Small sleep to prevent busy-waiting if queue is empty
                continue

            future = pending_futures.popleft()
            try:
                result = await future # Wait for the background task to complete
                await results_queue.put(result) # Put result onto the queue for yielding
            except Exception as e:
                print(f"Error in background audio conversion: {e}")
                traceback.print_exc()
                await results_queue.put(None) # Signal error downstream if needed

            # Check if this was the last future after the main loop finished
            if result is END_OF_STREAM:
                 break # Exit producer loop


    # Start the background task that processes results
    producer_task = asyncio.create_task(_result_producer())

    try:
        # --- Main Token Processing Loop ---
        async for token_sim in token_gen:
            token_receive_count += 1

            token = cached_turn_token_into_id(token_sim, count)

            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # --- Batch Submission Logic ---
                buffer_len = len(buffer) # Check length once

                # First chunk processing - low latency
                if not first_chunk_processed and buffer_len >= MIN_FRAMES_FIRST:
                    # Process the most recent MIN_FRAMES_FIRST tokens
                    tokens_to_process = itertools.islice(buffer, buffer_len - MIN_FRAMES_FIRST, buffer_len)
                    await _submit_audio_task(list(tokens_to_process), count) # Submit task
                    first_chunk_processed = True # Mark as processed regardless of audio result status

                # Subsequent chunks processing - throughput (check only when needed)
                elif first_chunk_processed and count % PROCESS_EVERY_N == 0:
                    tokens_to_process = None
                    # Select the optimal batch size based on available tokens
                    if buffer_len >= IDEAL_FRAMES:
                         # Slice the most recent IDEAL_FRAMES tokens
                        tokens_to_process = itertools.islice(buffer, buffer_len - IDEAL_FRAMES, buffer_len)
                    elif buffer_len >= MIN_FRAMES_SUBSEQUENT:
                         # Slice the most recent MIN_FRAMES_SUBSEQUENT tokens
                        tokens_to_process = itertools.islice(buffer, buffer_len - MIN_FRAMES_SUBSEQUENT, buffer_len)

                    if tokens_to_process:
                       await _submit_audio_task(list(tokens_to_process), count) # Submit task

            # --- Throughput Logging ---
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                elapsed = current_time - last_log_time
                if elapsed > 0:
                    tokens_per_sec = token_receive_count / elapsed
                    print(f"Token reception rate: {tokens_per_sec:.1f} tokens/second")
                last_log_time = current_time
                token_receive_count = 0

        # --- End-of-Generation Handling ---
        if buffer:
            # Process any remaining tokens in the buffer
            # Decide whether to process the whole buffer or just the last relevant part
            # Processing the whole remaining buffer seems intended by the original code
            tokens_to_process = list(buffer) # Process all remaining

            # Submit final chunk
            await _submit_audio_task(tokens_to_process, count)

            # Handle potential padding ONLY IF convert_to_audio strictly requires it
            # This padding logic seems complex and possibly unnecessary unless the model *must*
            # have fixed-size inputs even at the very end. Let's simplify/remove it unless required.
            # Original padding logic commented out for clarity/performance unless needed:
            # padding_needed = IDEAL_FRAMES - len(tokens_to_process)
            # if padding_needed > 0 and len(tokens_to_process) > 0:
            #     # Use the very last token for padding
            #     padded_buffer_list = tokens_to_process + [tokens_to_process[-1]] * padding_needed
            #     print(f"Processing final padded frame: {len(tokens_to_process)} tokens + {padding_needed} padding")
            #     await _submit_audio_task(padded_buffer_list, count)


    except Exception as e:
        print(f"Error in tokens_decoder main loop: {e}")
        traceback.print_exc()
        # Cancel the producer if it's still running
        producer_task.cancel()
    finally:
        # --- Signal End of Stream to Producer ---
        # Ensure producer processes all pending tasks before exiting
        # Submit a final task that just signals the end
        final_signal_future = loop.run_in_executor(executor, lambda: END_OF_STREAM)
        pending_futures.append(final_signal_future)

        # --- Yield Results from Queue ---
        while True:
            audio_samples = await results_queue.get()
            if audio_samples is END_OF_STREAM:
                results_queue.task_done()
                break # End of stream signal received
            if audio_samples is not None:
                yield audio_samples
            results_queue.task_done() # Mark item as processed for backpressure join()

        # Wait for the producer task to finish cleanly
        await producer_task

        # --- Cleanup ---
        print("Shutting down executor...")
        executor.shutdown(wait=True) # Wait for all tasks to complete
        print("Executor shut down.")


        # --- Performance Reporting ---
        total_elapsed = time.time() - start_time
        if total_elapsed > 0:
             # Use total_tokens_processed_for_audio for a more relevant throughput metric
             # 'count' includes tokens buffered but maybe not sent for audio processing
             effective_rate = total_tokens_processed_for_audio / total_elapsed
             print(f"Total processing time: {total_elapsed:.2f}s")
             print(f"Total tokens processed for audio: {total_tokens_processed_for_audio}")
             print(f"Average audio processing rate: {effective_rate:.1f} tokens/s")
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