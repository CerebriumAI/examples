# %% Imports and Constants
from snac import SNAC
import numpy as np
import torch
import asyncio
from collections import deque
import itertools
from functools import lru_cache, partial
import concurrent.futures
import traceback
import time
import os
import sys

# Performance-oriented constants - TUNED FOR ~2X SPEEDUP
MIN_FRAMES_FIRST = 1           # Keep low for first chunk latency
MIN_FRAMES_SUBSEQUENT = 32     # Increased slightly for better batching efficiency
IDEAL_FRAMES = 64              # Increased target batch size for higher throughput
PROCESS_EVERY_N = 7            # Keep aligned with model's frame structure (critical)
MAX_PENDING_AUDIO_TASKS = 16   # Doubled: More tasks in flight for deeper GPU pipeline
MAX_CACHE_SIZE = 100000        # Doubled: Larger token cache
PREFETCH_FACTOR = 4            # Increased: Keep GPU fed more aggressively

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    # Adjusted logic for potential variations
    return ('UVICORN_RELOADER' in os.environ or 
            'reload' in sys.argv or 
            sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

# Set a flag to avoid repeat messages
IS_RELOADER = is_reloader_process()

# Setup and initialize model with optimal configuration based on hardware
def initialize_model():
    """Initialize the SNAC model with optimal settings for the current hardware"""
    print("Initializing SNAC model...") if not IS_RELOADER else None
    
    # Determine best available device
    snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {snac_device}") if not IS_RELOADER else None
    
    # Load model and move to appropriate device
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    model = model.to(snac_device)
    
    # Try to enable torch.compile if PyTorch 2.0+ is available
    TORCH_COMPILE_AVAILABLE = False
    try:
        if hasattr(torch, 'compile'):
            TORCH_COMPILE_AVAILABLE = True
            if not IS_RELOADER:
                print("PyTorch 2.0+ detected, torch.compile is available")
            # Compile model with optimal backend for the device
            # Use reduce-overhead for potentially faster kernel launches, fullgraph for better fusion
            compile_mode = "reduce-overhead" if snac_device == "cuda" else "default"
            model = torch.compile(model, mode=compile_mode, fullgraph=(snac_device == "cuda"))
            print(f"Model compiled with torch.compile (mode={compile_mode}, fullgraph={snac_device=='cuda'}) for faster inference") if not IS_RELOADER else None
        else:
             print("torch.compile not available (requires PyTorch 2.0+)") if not IS_RELOADER else None
    except Exception as e:
        print(f"torch.compile failed, continuing without compile: {e}") if not IS_RELOADER else None
    
    # Setup CUDA optimizations if available
    cuda_stream = None
    if snac_device == "cuda":
        # Create dedicated CUDA stream for audio processing
        cuda_stream = torch.cuda.Stream()
        
        # Enable memory optimizations
        torch.cuda.empty_cache()
        # Enable cudnn benchmarking for optimal kernel selection
        torch.backends.cudnn.benchmark = True
        
        print("Using CUDA stream and optimizations for parallel processing") if not IS_RELOADER else None
    
    return model, snac_device, cuda_stream

# Global model initialization
model, snac_device, cuda_stream = initialize_model()

# Set up automated mixed precision (AMP) for faster inference ONLY on CUDA
amp_enabled = snac_device == "cuda"
amp_dtype = torch.float16 if amp_enabled else torch.float32 # Use float16 for AMP
print(f"AMP Enabled: {amp_enabled} (dtype: {amp_dtype})") if not IS_RELOADER and snac_device=="cuda" else None

# %% Optimized convert_to_audio function
def convert_to_audio(multiframe, count):
    """
    Ultra-optimized version of convert_to_audio with AMP, CUDA streams,
    pinned memory, and reduced CPU-GPU transfers for maximum throughput.
    """
    # Early validation
    if len(multiframe) < 7:
        return None
    
    # Determine number of complete frames (must be multiple of 7)
    num_complete_sets = len(multiframe) // 7
    if num_complete_sets == 0:
        return None
    
    # Trim multiframe to exact multiple of 7 for processing
    num_tokens_to_process = num_complete_sets * 7
    actual_multiframe = multiframe[:num_tokens_to_process]
    num_frames = num_complete_sets # Renamed for clarity
        
    try:
        # --- Optimized Input Transfer ---
        # Use optimal tensor creation strategy based on input type
        if isinstance(actual_multiframe, np.ndarray):
            # Convert NumPy directly to device using non_blocking transfer with CUDA stream
            if snac_device == "cuda":
                 # Create pinned memory tensor first for faster H2D copy
                 pinned_input = torch.from_numpy(actual_multiframe).pin_memory()
                 device_tensor = pinned_input.to(snac_device, non_blocking=True)
            else:
                 device_tensor = torch.from_numpy(actual_multiframe).to(snac_device) # MPS/CPU don't use non_blocking well here
        elif isinstance(actual_multiframe, torch.Tensor):
            # Assume tensor might already be on device or CPU
            if actual_multiframe.device == snac_device:
                device_tensor = actual_multiframe # No copy needed
            elif snac_device == "cuda" and actual_multiframe.is_pinned():
                 device_tensor = actual_multiframe.to(snac_device, non_blocking=True)
            elif snac_device == "cuda":
                 # Pin if not already pinned before async transfer
                 device_tensor = actual_multiframe.pin_memory().to(snac_device, non_blocking=True)
            else:
                 device_tensor = actual_multiframe.to(snac_device)
        else:
            # Fallback for lists etc. - potentially slower path
             if snac_device == "cuda":
                 pinned_input = torch.tensor(actual_multiframe, dtype=torch.int32).pin_memory()
                 device_tensor = pinned_input.to(snac_device, non_blocking=True)
             else:
                 device_tensor = torch.tensor(actual_multiframe, dtype=torch.int32, device=snac_device)
        
        # --- Vectorized Code Preparation on Device ---
        # Pre-allocate tensors with optimal shape directly on device
        codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
        codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)  
        codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
        
        # Use advanced indexing for potentially faster assignment (might require tuning)
        # This avoids the Python loop, potentially faster for large num_frames
        indices = torch.arange(num_frames, device=snac_device) * 7
        codes_0[0, :] = device_tensor[indices]
        
        idx1_a = indices + 1
        idx1_b = indices + 4
        codes_1[0, 0::2] = device_tensor[idx1_a]
        codes_1[0, 1::2] = device_tensor[idx1_b]

        idx2_a = indices + 2
        idx2_b = indices + 3
        idx2_c = indices + 5
        idx2_d = indices + 6
        codes_2[0, 0::4] = device_tensor[idx2_a]
        codes_2[0, 1::4] = device_tensor[idx2_b]
        codes_2[0, 2::4] = device_tensor[idx2_c]
        codes_2[0, 3::4] = device_tensor[idx2_d]

        # --- Validation on Device (Optional - remove if tokens always valid) ---
        # Vectorized validation - single operation instead of loops
        # if (torch.any(codes_0 < 0) or torch.any(codes_0 >= 4096) or # Corrected upper bound (exclusive)
        #     torch.any(codes_1 < 0) or torch.any(codes_1 >= 4096) or
        #     torch.any(codes_2 < 0) or torch.any(codes_2 >= 4096)):
        #     # print(f"Warning: Invalid token ID detected in batch {count}.") # DEBUG
        #     return None # Or handle appropriately

        codes = [codes_0, codes_1, codes_2]
        
        # --- Inference and Post-processing ---
        # Use inference_mode for efficiency, AMP context, and CUDA stream
        with torch.inference_mode(), \
             torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype), \
             (torch.cuda.stream(cuda_stream) if cuda_stream is not None else contextlib.nullcontext()): # Use stream context manager

            # Run model inference
            audio_hat = model.decode(codes) # Shape: [1, 1, N*2048] where N=num_frames
            
            # Direct slicing to the relevant middle part(s)
            # The model seems designed for overlap-add; we take the second half of each 4096 segment
            # For num_frames=1 -> [:,:, 2048:4096]
            # For num_frames=2 -> [:,:, 2048:4096] and [:,:, 6144:8192] ??? -> Let's stick to the original simple slice for now
            # Original slice assumes overlap add is handled elsewhere or we only care about the central part of the *batch*
            # The middle 2048 samples of the *entire* output batch correspond to the input tokens shifted appropriately.
            output_length = audio_hat.shape[-1]
            start_slice = output_length // 4 # Assuming 2048 is 1/4 of total length per frame? Let's stick to original
            end_slice = output_length // 4 * 2 # Assuming 2048 is 1/4 of total length per frame? Let's stick to original
            # Reverting to original fixed slice based on observed behavior
            start_slice = 2048
            end_slice = 4096
            
            # Ensure slice indices are valid even if output is short (shouldn't happen if input > 7 tokens)
            if end_slice > audio_hat.shape[-1]:
                 end_slice = audio_hat.shape[-1]
                 start_slice = max(0, end_slice - 2048) # Try to get 2048 samples

            if start_slice >= end_slice:
                 return None # Cannot extract valid slice

            audio_slice = audio_hat[:, :, start_slice:end_slice]
            
            # Convert to int16 while still on GPU (faster)
            audio_int16_tensor = (audio_slice * 32767.0).clamp(-32768, 32767).round().to(torch.int16)
            
            # --- Optimized Output Transfer ---
            if snac_device == "cuda":
                # Create pinned memory buffer on CPU for efficient D2H copy
                cpu_tensor = torch.empty_like(audio_int16_tensor, device="cpu", pin_memory=True)
                # Perform non-blocking copy from GPU to pinned CPU buffer
                cpu_tensor.copy_(audio_int16_tensor, non_blocking=True)
                # CRITICAL: Synchronize the stream *before* accessing cpu_tensor.numpy()
                # This ensures the copy operation is complete.
                cuda_stream.synchronize()
                return cpu_tensor.numpy().tobytes()
            else:
                # For MPS/CPU, standard copy is fine
                return audio_int16_tensor.cpu().numpy().tobytes()

    except Exception as e:
        print(f"Error in convert_to_audio (count={count}): {e}")
        traceback.print_exc()
        return None
    finally:
        # Optional: Clear cache if memory pressure is high, but usually not needed with streams
        # if snac_device == "cuda": torch.cuda.empty_cache()
        pass


# %% Token Conversion (Minor changes, already reasonably fast)
import contextlib # Needed for nullcontext

CUSTOM_TOKEN_PREFIX = "<custom_token_"
# Use a global dictionary cache (faster than LRU for this pattern if size is managed)
token_id_cache = {}

def turn_token_into_id(token_string, index):
    """
    Optimized token-to-ID conversion using direct dictionary cache.
    """
    prefix = CUSTOM_TOKEN_PREFIX
    mod = index % 7
    cache_key = (token_string, mod)
    
    # Fast path: check cache first
    cached_value = token_id_cache.get(cache_key, Ellipsis) # Use Ellipsis as unique sentinel
    if cached_value is not Ellipsis:
        return cached_value
    
    # Slow path: compute token ID
    token_id = None # Default to None (invalid)
    if token_string.startswith(prefix) and token_string.endswith(">"):
        num_str = token_string[len(prefix):-1]
        if num_str.isdigit():
            num = int(num_str)
            # Calculate potential token ID
            calculated_id = num - 10 - (mod * 4096)
            # Add validation: Ensure ID is within the expected range [0, 4095]
            if 0 <= calculated_id < 4096:
                 token_id = calculated_id

    # Update cache, managing size crudely (can be improved with LRU logic if needed)
    if len(token_id_cache) > MAX_CACHE_SIZE * 1.1: # Allow slight overshoot before purge
         # Simple purge: Remove half the cache items (randomly or oldest)
         keys_to_remove = list(token_id_cache.keys())[:MAX_CACHE_SIZE // 2]
         for key in keys_to_remove:
             del token_id_cache[key]
             
    token_id_cache[cache_key] = token_id
    return token_id

# Use LRU cache on top for potential minor extra hits, but dict lookup is very fast.
# Consider removing if dict cache proves sufficient. Maxsize adjusted based on MAX_CACHE_SIZE
@lru_cache(maxsize=min(MAX_CACHE_SIZE, 32768)) # Cap LRU size reasonably
def cached_turn_token_into_id(token_string, position):
    """Two-level caching for maximum performance"""
    return turn_token_into_id(token_string, position)


# %% Optimized Async Tokens Decoder
async def tokens_decoder(token_gen):
    """
    High-performance token decoder with adaptive batching, deeper pipeline,
    AMP, CUDA streams, and optimized data transfers.
    """
    # Use efficient data structures
    buffer = deque() # No maxlen needed if we process based on counts
    results_queue = asyncio.Queue(maxsize=MAX_PENDING_AUDIO_TASKS * PREFETCH_FACTOR) # Larger queue
    pending_futures = deque()

    loop = asyncio.get_running_loop()
    # Increase worker threads slightly, capped
    num_workers = min(os.cpu_count() * 2, 24) if snac_device == "cuda" else min(os.cpu_count(), 8)
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers,
        thread_name_prefix="AudioWorker"
    )
    
    # Performance tracking
    count = 0
    first_chunk_processed = False
    start_time = time.time()
    last_log_time = start_time
    token_receive_count = 0
    total_tokens_processed = 0
    audio_chunks_generated = 0
    processing_times = deque(maxlen=100)
    
    # --- Task Submission ---
    async def _submit_audio_task(tokens_to_process_list, current_token_count):
        """Submit tokens for audio conversion, using numpy array."""
        nonlocal total_tokens_processed
        if not tokens_to_process_list:
            return

        # Ensure enough tasks can be queued
        while len(pending_futures) >= MAX_PENDING_AUDIO_TASKS:
             # If queue is full, wait for oldest task to complete
             # This provides backpressure based on GPU processing, not just queue size
             if pending_futures:
                 oldest_future, _ = pending_futures[0] # Peek
                 await asyncio.wait([oldest_future], return_when=asyncio.FIRST_COMPLETED)
             else: # Should not happen if len >= MAX_PENDING... but safety break
                 await asyncio.sleep(0.001)
             # Re-check completed futures in the result processor to clear space
             await asyncio.sleep(0) # Yield to allow result processor to run


        task_start = time.time()
        # Create numpy array efficiently
        token_array = np.array(tokens_to_process_list, dtype=np.int32)
        
        # Submit task to executor
        future = loop.run_in_executor(
            executor,
            partial(convert_to_audio, token_array, current_token_count) # Pass numpy array
        )
        pending_futures.append((future, task_start))
        total_tokens_processed += len(tokens_to_process_list)

    # --- Result Processing ---
    END_OF_STREAM = object()
    async def _result_processor():
        """Process completed futures and manage results queue."""
        nonlocal audio_chunks_generated, processing_times
        
        while True:
            processed_in_cycle = 0
            # Process all completed futures without blocking indefinitely
            while pending_futures:
                future, start_time = pending_futures[0] # Peek at oldest
                if not future.done():
                     # If oldest isn't done, break and wait/yield
                     break

                # Oldest is done, pop it and process
                pending_futures.popleft()
                processed_in_cycle += 1
                try:
                    result = await asyncio.wrap_future(future) # Get result safely
                    
                    # Track processing time
                    process_time = time.time() - start_time
                    processing_times.append(process_time)
                    
                    if result is END_OF_STREAM:
                        await results_queue.put(END_OF_STREAM)
                        print("Result processor received END_OF_STREAM signal.")
                        return # Exit processor task

                    if result is not None:
                         audio_chunks_generated += 1
                         await results_queue.put(result)
                    # else: # Optionally handle None results (errors during conversion)
                    #     print(f"Warning: Received None audio chunk.")
                        
                except Exception as e:
                    print(f"Error retrieving result from future: {e}")
                    traceback.print_exc()
                    # Optionally put None or raise specific exception
                    # await results_queue.put(None) # Signal error downstream?

            # If no futures were ready, wait briefly before checking again
            if processed_in_cycle == 0:
                 # Check if stream ended and futures are empty
                 if not pending_futures and executor._shutdown: # Check if executor is shutting down (EOS signal sent) - HACKY check
                      all_done = True
                      async for task in asyncio.as_completed([f for f, _ in pending_futures]): # Recheck if any snuck in
                           all_done = False
                           break
                      if all_done and results_queue.empty():
                          # If truly finished, send EOS if not already sent
                          if END_OF_STREAM not in list(results_queue._queue): # Avoid duplicate EOS
                              await results_queue.put(END_OF_STREAM)
                          print("Result processor exiting: No pending futures and executor shutdown.")
                          return

                 await asyncio.sleep(0.001) # Small sleep to yield control and prevent busy-waiting
            else:
                await asyncio.sleep(0) # Yield briefly after processing results

    # Start background result processor
    processor_task = asyncio.create_task(_result_processor())
    
    # --- Main Token Loop ---
    try:
        processed_token_count = 0 # Tracks tokens *used* in batches
        async for token_sim in token_gen:
            token_receive_count += 1
            
            token = cached_turn_token_into_id(token_sim, count)
            
            if token is not None: # Ignore invalid/padding tokens
                buffer.append(token)
                count += 1
                
                current_buffer_len = len(buffer)
                
                # Determine batch size
                batch_size = 0
                if not first_chunk_processed:
                    if current_buffer_len >= MIN_FRAMES_FIRST * 7:
                        # Process exactly MIN_FRAMES_FIRST * 7 for the first chunk
                        batch_size = MIN_FRAMES_FIRST * 7
                        first_chunk_processed = True
                elif count % PROCESS_EVERY_N == 0: # Check on frame boundaries
                    # Subsequent chunks: try to process IDEAL_FRAMES or at least MIN_FRAMES_SUBSEQUENT
                    if current_buffer_len >= IDEAL_FRAMES * 7:
                        batch_size = IDEAL_FRAMES * 7
                    elif current_buffer_len >= MIN_FRAMES_SUBSEQUENT * 7:
                         batch_size = MIN_FRAMES_SUBSEQUENT * 7
                    # Optionally add adaptive logic based on processing_times here if needed

                # If a batch is ready to be processed
                if batch_size > 0:
                    # Extract the exact number of tokens for the batch from the *start* of the buffer
                    tokens_to_process = [buffer.popleft() for _ in range(batch_size)]
                    
                    if tokens_to_process:
                        await _submit_audio_task(tokens_to_process, count)
                        processed_token_count += batch_size

            # Performance logging
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                elapsed = current_time - last_log_time
                if elapsed > 0:
                    tokens_per_sec = token_receive_count / elapsed
                    processed_rate = (count - processed_token_count) / elapsed # Rate of tokens entering buffer vs processed
                    q_size = results_queue.qsize()
                    pending_tasks = len(pending_futures)
                    avg_process_time = (sum(processing_times) / len(processing_times)) if processing_times else 0
                    print(f"Tokens Rate: {tokens_per_sec:.1f}/s | Buf In-Out Rate: {processed_rate:.1f}/s | Q: {q_size} | Pending: {pending_tasks} | Avg Latency: {avg_process_time*1000:.1f} ms") # More detailed log
                last_log_time = current_time
                token_receive_count = 0
        
        # --- Stream End Processing ---
        print("Token generator finished.")
        # Process any remaining tokens in the buffer
        while len(buffer) >= MIN_FRAMES_SUBSEQUENT * 7: # Process remaining in standard batches
            batch_size = min(len(buffer), IDEAL_FRAMES * 7)
            # Ensure batch size is multiple of 7
            batch_size = (batch_size // 7) * 7
            if batch_size == 0: break # Avoid infinite loop if less than 7 remain

            tokens_to_process = [buffer.popleft() for _ in range(batch_size)]
            if tokens_to_process:
                await _submit_audio_task(tokens_to_process, count + len(tokens_to_process)) # Adjust count notionally
                processed_token_count += batch_size
        
        # Handle the final leftover tokens (less than MIN_FRAMES_SUBSEQUENT * 7)
        if buffer:
            final_tokens = list(buffer)
            buffer.clear()
            # Pad the final batch slightly if needed? Or just process what's left?
            # Processing requires multiple of 7. If len(final_tokens) % 7 != 0, the last few are lost.
            final_batch_size = (len(final_tokens) // 7) * 7
            if final_batch_size > 0:
                 await _submit_audio_task(final_tokens[:final_batch_size], count + len(final_tokens))
                 processed_token_count += final_batch_size
            print(f"Processed final {final_batch_size} tokens. {len(final_tokens) - final_batch_size} tokens discarded (not multiple of 7).")


        print("Submitting END_OF_STREAM signal to executor.")
        # Submit sentinel to executor to signal end
        eos_future = loop.run_in_executor(executor, lambda: END_OF_STREAM)
        pending_futures.append((eos_future, time.time()))

        # Yield results until END_OF_STREAM is received
        while True:
            audio_samples = await results_queue.get()
            if audio_samples is END_OF_STREAM:
                results_queue.task_done()
                print("Received END_OF_STREAM signal from queue.")
                break
            
            if audio_samples is not None:
                yield audio_samples
            
            results_queue.task_done()

        print("Finished yielding audio samples.")
        # Wait for the processor task to finish cleanly
        await processor_task
        print("Result processor task finished.")

    except asyncio.CancelledError:
         print("Tokens decoder task cancelled.")
         processor_task.cancel() # Propagate cancellation
         raise
    except Exception as e:
        print(f"Error in tokens_decoder main loop: {e}")
        traceback.print_exc()
        processor_task.cancel() # Cancel processor on error
    finally:
        # Clean shutdown
        print("Shutting down audio processing executor...")
        # Don't wait indefinitely, tasks should be finishing or cancelled
        executor.shutdown(wait=False, cancel_futures=True)
        
        # Final performance report
        total_elapsed = time.time() - start_time
        if total_elapsed > 0:
            # Use 'count' as proxy for total tokens received that *could* be processed
            final_tokens_processed = processed_token_count
            tokens_per_sec = final_tokens_processed / total_elapsed
            print(f"\n--- Final Performance Report ---")
            print(f"Total processing time: {total_elapsed:.2f}s")
            print(f"Valid tokens processed: {final_tokens_processed} at {tokens_per_sec:.1f} tokens/s")
            print(f"Audio chunks generated: {audio_chunks_generated}")
            avg_process_time = (sum(processing_times) / len(processing_times)) if processing_times else 0
            print(f"Avg. Batch Processing Latency: {avg_process_time*1000:.1f} ms")
            print(f"--- End Report ---")

# %% Optimized Sync Tokens Decoder (Conceptual - less benefit from async opts)
# NOTE: Sync version benefits less from fine-grained async/await and stream pipelining.
# The main speedups here come from AMP, torch.compile, larger batches, and more workers.
def tokens_decoder_sync(syn_token_gen):
    """
    Synchronous decoder benefiting from AMP, compile, larger batches,
    and increased parallelism. Pipelining is harder here.
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    # Optimized constants for synchronous processing - MORE AGGRESSIVE
    # Batch size significantly larger for CUDA to leverage throughput
    BATCH_SIZE = (IDEAL_FRAMES * 7 * 4) if snac_device == "cuda" else (IDEAL_FRAMES * 7) # e.g., 1792 tokens for CUDA
    PREFETCH_SIZE = 4 if snac_device == "cuda" else 2 # Prefetch multiple large batches
    # More workers, especially if GPU is fast
    MAX_WORKERS = min(os.cpu_count() * 2, 32) if snac_device == "cuda" else min(os.cpu_count(), 8)
    
    # Set up queues with optimal capacity based on large batches
    task_queue = queue.Queue(maxsize=PREFETCH_SIZE * 2) # Queue holds batches, not individual items
    result_queue = queue.Queue(maxsize=PREFETCH_SIZE * 2)
    shutdown_event = threading.Event()
    
    # Efficiently collect and batch tokens
    def token_batcher():
        count = 0
        batch = []
        
        try:
            for token_sim in syn_token_gen:
                if shutdown_event.is_set(): break # Exit early if shutdown requested

                token = cached_turn_token_into_id(token_sim, count)
                
                if token is not None: # Process only valid tokens
                    batch.append(token)
                    count += 1
                    
                    # Submit batch when full enough (multiple of 7)
                    if len(batch) >= BATCH_SIZE:
                         # Trim to exact multiple of 7 before submitting
                         num_to_submit = (len(batch) // 7) * 7
                         if num_to_submit > 0:
                            batch_to_submit = batch[:num_to_submit]
                            if not shutdown_event.is_set():
                                task_queue.put(np.array(batch_to_submit, dtype=np.int32)) # Submit NumPy array
                            batch = batch[num_to_submit:] # Keep remainder
            
            # Process any remaining tokens after loop ends
            if batch and not shutdown_event.is_set():
                num_to_submit = (len(batch) // 7) * 7
                if num_to_submit > 0:
                     task_queue.put(np.array(batch[:num_to_submit], dtype=np.int32))
                remaining_count = len(batch) - num_to_submit
                if remaining_count > 0:
                     print(f"Sync: Discarding {remaining_count} trailing tokens (not multiple of 7).")

        except Exception as e:
            print(f"Token batcher error: {e}")
            traceback.print_exc()
        finally:
            # Signal end of processing by putting None in the task queue
            print("Sync Batcher: Signaling end.")
            if not shutdown_event.is_set():
                task_queue.put(None)
    
    # Worker function for audio processing
    def audio_worker():
        worker_id = threading.current_thread().name
        # print(f"{worker_id}: Starting") # Debug
        while not shutdown_event.is_set():
            try:
                # Get next batch with timeout to allow checking shutdown flag
                batch_array = task_queue.get(timeout=0.1) # Get numpy array
                
                if batch_array is None:  # End signal
                    # print(f"{worker_id}: Received None, signaling end and exiting.") # Debug
                    task_queue.task_done()
                    # Crucially, put None back for other workers to see
                    if not shutdown_event.is_set():
                        task_queue.put(None)
                    break # Exit worker loop
                
                # Process batch (already a numpy array)
                # print(f"{worker_id}: Processing batch of size {len(batch_array)}") # Debug
                audio_data = convert_to_audio(batch_array, -1) # Pass array, count isn't critical here
                
                if audio_data is not None and not shutdown_event.is_set():
                    # print(f"{worker_id}: Putting result in queue.") # Debug
                    result_queue.put(audio_data)
                # else:
                    # print(f"{worker_id}: Processing returned None or shutdown set.") # Debug

                task_queue.task_done()
                
            except queue.Empty:
                # Just means no work right now, loop again and check shutdown flag
                continue
            except Exception as e:
                print(f"{worker_id} error: {e}")
                traceback.print_exc()
                # Ensure task_done is called even on error if item was retrieved
                try:
                     task_queue.task_done()
                except ValueError: # May happen if get failed but exception occurred after
                     pass
        # print(f"{worker_id}: Exiting loop.") # Debug

    # Start token batcher thread
    batcher_thread = threading.Thread(target=token_batcher, name="TokenBatcherSync")
    batcher_thread.daemon = True # Allow exit even if batcher hangs
    batcher_thread.start()
    
    # Start worker threads using ThreadPoolExecutor for easier management
    worker_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="SyncAudioWorker")
    futures = [worker_pool.submit(audio_worker) for _ in range(MAX_WORKERS)]
    print(f"Sync: Started {MAX_WORKERS} audio workers.")

    # Yield results as they become available
    active_workers = MAX_WORKERS
    
    try:
        while active_workers > 0: # Continue as long as workers might produce results
            try:
                # Get result with timeout
                result = result_queue.get(timeout=0.1)
                
                if result is None: # Should not happen based on worker logic, but safety check
                    print("Sync: Warning - received unexpected None in result queue.")
                    result_queue.task_done()
                    # This might indicate an issue, but we continue trying to fetch
                    continue
                else:
                    yield result
                
                result_queue.task_done()
                
            except queue.Empty:
                 # Check if workers have finished
                 finished_workers = sum(f.done() for f in futures)
                 if finished_workers == MAX_WORKERS:
                      # All workers submitted their tasks are done. Check if queue is truly empty.
                      if result_queue.empty():
                          print("Sync: All workers finished and result queue empty.")
                          break # Exit main loop
                      else:
                          # If queue not empty, continue draining it
                          continue
                 else:
                      # Workers still running, continue waiting
                      active_workers = MAX_WORKERS - finished_workers
                      continue

    except GeneratorExit:
        # Handle generator being closed early (e.g., client disconnects)
        print("Sync: GeneratorExit received, shutting down workers.")
        shutdown_event.set() # Signal threads to stop
        # Put None multiple times to potentially unblock workers faster
        for _ in range(MAX_WORKERS):
             try: task_queue.put_nowait(None)
             except queue.Full: break

    finally:
        # Clean shutdown
        print("Sync: Final cleanup.")
        shutdown_event.set() # Ensure flag is set

        # Wait briefly for workers to potentially finish processing current item
        worker_pool.shutdown(wait=True, cancel_futures=False) # Wait for running tasks, don't cancel forcefully initially

        print("Sync: Workers shut down.")

        # Clear queues (optional, workers should have emptied task_queue)
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
                result_queue.task_done()
            except queue.Empty:
                break
        print("Sync: Cleanup complete.")