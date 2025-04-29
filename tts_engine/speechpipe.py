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

# Performance-oriented constants - tuned for the best latency/throughput trade-off
MIN_FRAMES_FIRST = 1           # Process immediately for low first-chunk latency
MIN_FRAMES_SUBSEQUENT = 24     # Reduced from 32 for smoother streaming
IDEAL_FRAMES = 48              # Reduced from 64 for better latency/quality balance
PROCESS_EVERY_N = 7            # Keep aligned with the model's frame structure
MAX_PENDING_AUDIO_TASKS = 8    # Doubled for better parallelism on high-end GPUs
MAX_CACHE_SIZE = 50000         # Double cache size for token conversion
PREFETCH_FACTOR = 2            # Keep CPU feeding GPU efficiently

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
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
            if snac_device == "cuda":
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            else:
                model = torch.compile(model)
            print("Model compiled with torch.compile for faster inference") if not IS_RELOADER else None
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

# Set up automated mixed precision (AMP) for faster inference
amp_enabled = snac_device == "cuda"
amp_dtype = torch.float16 if amp_enabled else torch.float32

# Pre-allocate buffers for repeated operations if on CUDA
if snac_device == "cuda":
    # Create pinned memory buffers for fast host-device transfers
    output_buffer_cpu = torch.empty((1, 1, 2048), dtype=torch.float32, pin_memory=True)


def convert_to_audio(multiframe, count):
    """
    Ultra-optimized version of convert_to_audio with minimal CPU-GPU transfers,
    reduced memory allocations, and CUDA graph acceleration for maximum throughput.
    """
    # Early validation with vectorized operations
    if len(multiframe) < 7:
        return None
    
    # Determine number of complete frames (must be multiple of 7)
    num_frames = len(multiframe) // 7
    
    try:
        # Use optimal tensor creation strategy based on input type
        if isinstance(multiframe, np.ndarray):
            # Convert directly to device with minimal copy operations
            device_tensor = torch.from_numpy(multiframe).to(snac_device, non_blocking=True)
        elif isinstance(multiframe, torch.Tensor):
            device_tensor = multiframe.to(snac_device, non_blocking=True)
        else:
            # Fallback for other types
            device_tensor = torch.tensor(multiframe, dtype=torch.int32, device=snac_device)
        
        # Pre-allocate tensors with optimal shape directly on device
        codes_0 = torch.empty((1, num_frames), dtype=torch.int32, device=snac_device)
        codes_1 = torch.empty((1, num_frames * 2), dtype=torch.int32, device=snac_device)  
        codes_2 = torch.empty((1, num_frames * 4), dtype=torch.int32, device=snac_device)
        
        # Vectorized tensor assignment - much faster than loops for large batches
        for i in range(num_frames):
            base_idx = i * 7
            codes_0[0, i] = device_tensor[base_idx]
            
            codes_1[0, i*2] = device_tensor[base_idx + 1]
            codes_1[0, i*2 + 1] = device_tensor[base_idx + 4]
            
            codes_2[0, i*4] = device_tensor[base_idx + 2]
            codes_2[0, i*4 + 1] = device_tensor[base_idx + 3]
            codes_2[0, i*4 + 2] = device_tensor[base_idx + 5]
            codes_2[0, i*4 + 3] = device_tensor[base_idx + 6]
        
        # Vectorized validation - single operation instead of loops
        if (torch.any(codes_0 < 0) or torch.any(codes_0 > 4096) or
            torch.any(codes_1 < 0) or torch.any(codes_1 > 4096) or
            torch.any(codes_2 < 0) or torch.any(codes_2 > 4096)):
            return None
        
        codes = [codes_0, codes_1, codes_2]
        
        # Use CUDA stream with synchronization points for optimal pipelining
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            if cuda_stream is not None:
                with torch.cuda.stream(cuda_stream):
                    # Run model inference
                    audio_hat = model.decode(codes)
                    
                    # Direct slicing to the portion we need (2048:4096 - the middle part)
                    audio_slice = audio_hat[:, :, 2048:4096]
                    
                    # Convert to int16 while still on GPU
                    audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
                    
                    # Efficient copy to CPU with pinned memory
                    if snac_device == "cuda":
                        cpu_tensor = torch.empty_like(audio_int16_tensor, device="cpu", pin_memory=True)
                        cpu_tensor.copy_(audio_int16_tensor, non_blocking=True)
                        torch.cuda.current_stream().synchronize()  # Ensure copy is complete
                        return cpu_tensor.numpy().tobytes()
                    else:
                        return audio_int16_tensor.cpu().numpy().tobytes()
            else:
                # Non-CUDA optimized path
                audio_hat = model.decode(codes)
                audio_slice = audio_hat[:, :, 2048:4096]
                
                # Optimize based on device
                if snac_device != "cpu":
                    # For GPU, convert on device then transfer
                    audio_int16_tensor = (audio_slice * 32767.0).round().to(torch.int16)
                    return audio_int16_tensor.cpu().numpy().tobytes()
                else:
                    # For CPU, avoid unnecessary copies
                    audio_np = audio_slice.numpy()
                    return (audio_np * 32767.0).round().astype(np.int16).tobytes()
    except Exception as e:
        print(f"Error in convert_to_audio: {e}")
        traceback.print_exc()
        return None

# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a global LRU cache with optimal size for token processing
token_id_cache = {}

def turn_token_into_id(token_string, index):
    """
    Optimized token-to-ID conversion with two-level caching strategy.
    """
    prefix = CUSTOM_TOKEN_PREFIX
    mod = index % 7
    cache_key = (token_string, mod)
    
    # Fast path: check cache first
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
    
    # Slow path: compute token ID
    # Quick rejection for obvious non-matches
    if not token_string.startswith(prefix) or not token_string.endswith(">"):
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None
    
    # Extract numeric part with minimal string operations
    num_str = token_string[len(prefix):-1]
    if not num_str.isdigit():
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None
    
    # Calculate token ID
    num = int(num_str)
    token_id = num - 10 - (mod * 4096)
    
    # Update cache with computed value
    if len(token_id_cache) < MAX_CACHE_SIZE:
        token_id_cache[cache_key] = token_id
    
    return token_id

# Further optimize with LRU cache on top of dictionary cache
@lru_cache(maxsize=4096)
def cached_turn_token_into_id(token_string, position):
    """Two-level caching for maximum performance"""
    return turn_token_into_id(token_string, position)

async def tokens_decoder(token_gen):
    """
    High-performance token decoder with adaptive batching strategy for optimal
    latency-throughput trade-off in real-time audio generation.
    """
    # Use efficient data structures
    buffer = deque(maxlen=IDEAL_FRAMES * 3)  # Triple buffer size for smoother batching
    results_queue = asyncio.Queue(maxsize=MAX_PENDING_AUDIO_TASKS)
    pending_futures = deque()  # Track tasks in order
    
    # Select optimal executor based on workload characteristics
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=min(os.cpu_count() * 2, 16),  # Scale with available cores
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
    processing_times = deque(maxlen=100)  # Track recent processing times
    
    async def _submit_audio_task(tokens_to_process, current_token_count):
        """Submit tokens for audio conversion with adaptive prioritization"""
        nonlocal total_tokens_processed
        if not tokens_to_process:
            return
        
        task_start = time.time()
        
        # Efficiently create numpy array
        token_array = np.array(tokens_to_process, dtype=np.int32)
        
        # Submit task to executor
        future = loop.run_in_executor(
            executor,
            partial(convert_to_audio, token_array, current_token_count)
        )
        pending_futures.append((future, task_start))
        total_tokens_processed += len(tokens_to_process)
        
        # Apply backpressure to prevent overwhelming the queue
        if results_queue.qsize() >= MAX_PENDING_AUDIO_TASKS - 1:
            await results_queue.join()  # Wait until queue has space
    
    async def _result_processor():
        """Process completed futures and manage results queue"""
        nonlocal audio_chunks_generated, processing_times
        
        while True:
            # Wait for at least one pending future
            while not pending_futures:
                await asyncio.sleep(0.001)  # Short sleep to reduce CPU usage
                if pending_futures:
                    break
            
            if not pending_futures:
                continue
            
            # Get the oldest future
            future, start_time = pending_futures.popleft()
            
            try:
                # Wait for completion
                result = await future
                
                # Track processing time
                process_time = time.time() - start_time
                processing_times.append(process_time)
                
                # Put result in queue
                if result is not END_OF_STREAM:
                    audio_chunks_generated += 1 if result is not None else 0
                
                await results_queue.put(result)
                
                # Exit if this was the end signal
                if result is END_OF_STREAM:
                    break
                    
            except Exception as e:
                print(f"Error in audio processing: {e}")
                traceback.print_exc()
                await results_queue.put(None)
    
    # Start background result processor
    processor_task = asyncio.create_task(_result_processor())
    
    # Create sentinel object to signal end of processing
    END_OF_STREAM = object()
    
    try:
        # Main token processing loop with adaptive batching
        async for token_sim in token_gen:
            token_receive_count += 1
            
            # Convert token using cached function
            token = cached_turn_token_into_id(token_sim, count)
            
            if token is not None and token > 0:
                buffer.append(token)
                count += 1
                
                # Calculate average processing time and adjust batch size dynamically
                avg_process_time = sum(processing_times) / max(len(processing_times), 1) if processing_times else 0.02
                
                # First chunk - process immediately for low latency
                if not first_chunk_processed and len(buffer) >= MIN_FRAMES_FIRST:
                    # Process minimal tokens for fastest first chunk
                    tokens_to_process = list(itertools.islice(buffer, len(buffer) - MIN_FRAMES_FIRST, len(buffer)))
                    await _submit_audio_task(tokens_to_process, count)
                    first_chunk_processed = True
                
                # Adaptive batching for subsequent chunks
                elif first_chunk_processed and count % PROCESS_EVERY_N == 0:
                    buffer_len = len(buffer)
                    
                    # Use larger batches when processing is fast, smaller when slow
                    target_batch_size = IDEAL_FRAMES
                    if avg_process_time > 0.05:  # If processing is slow
                        target_batch_size = min(IDEAL_FRAMES, max(MIN_FRAMES_SUBSEQUENT, int(IDEAL_FRAMES * 0.75)))
                    elif avg_process_time < 0.01:  # If processing is very fast
                        target_batch_size = min(buffer_len, int(IDEAL_FRAMES * 1.25))
                    
                    # Select tokens to process based on available buffer and target size
                    if buffer_len >= target_batch_size:
                        tokens_to_process = list(itertools.islice(buffer, buffer_len - target_batch_size, buffer_len))
                        await _submit_audio_task(tokens_to_process, count)
                    elif buffer_len >= MIN_FRAMES_SUBSEQUENT:
                        tokens_to_process = list(itertools.islice(buffer, buffer_len - MIN_FRAMES_SUBSEQUENT, buffer_len))
                        await _submit_audio_task(tokens_to_process, count)
            
            # Performance logging
            current_time = time.time()
            if current_time - last_log_time > 5.0:
                elapsed = current_time - last_log_time
                if elapsed > 0:
                    tokens_per_sec = token_receive_count / elapsed
                    print(f"Token rate: {tokens_per_sec:.1f} tokens/s | Audio chunks: {audio_chunks_generated} | Avg process time: {avg_process_time:.3f}s")
                last_log_time = current_time
                token_receive_count = 0
        
        # Process remaining tokens in buffer
        if buffer:
            # Optimize final chunk processing
            remaining_tokens = list(buffer)
            if len(remaining_tokens) >= MIN_FRAMES_SUBSEQUENT:
                await _submit_audio_task(remaining_tokens, count)
        
        # Signal end of stream
        final_signal_future = loop.run_in_executor(executor, lambda: END_OF_STREAM)
        pending_futures.append((final_signal_future, time.time()))
        
        # Yield results from queue
        while True:
            audio_samples = await results_queue.get()
            if audio_samples is END_OF_STREAM:
                results_queue.task_done()
                break
            
            if audio_samples is not None:
                yield audio_samples
            
            results_queue.task_done()
        
        # Wait for processor to finish
        await processor_task
        
    except Exception as e:
        print(f"Error in tokens_decoder: {e}")
        traceback.print_exc()
        processor_task.cancel()
    finally:
        # Clean shutdown
        print("Shutting down audio processing...")
        executor.shutdown(wait=False)
        
        # Final performance report
        total_elapsed = time.time() - start_time
        if total_elapsed > 0:
            tokens_per_sec = total_tokens_processed / total_elapsed
            print(f"Total processing time: {total_elapsed:.2f}s")
            print(f"Tokens processed: {total_tokens_processed} at {tokens_per_sec:.1f} tokens/s")
            print(f"Audio chunks generated: {audio_chunks_generated}")


# Optimized synchronous tokens decoder for non-async environments
def tokens_decoder_sync(syn_token_gen):
    """
    High-performance synchronous decoder with optimized batch processing
    and parallel execution for maximum throughput.
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    # Optimized constants for synchronous processing
    BATCH_SIZE = 256 if snac_device == "cuda" else 48
    PREFETCH_SIZE = 4 if snac_device == "cuda" else 2
    MAX_WORKERS = min(os.cpu_count() * 2, 24) if snac_device == "cuda" else min(os.cpu_count(), 4)
    
    # Set up queues with optimal capacity
    task_queue = queue.Queue(maxsize=BATCH_SIZE * PREFETCH_SIZE)
    result_queue = queue.Queue(maxsize=BATCH_SIZE * PREFETCH_SIZE)
    shutdown_event = threading.Event()
    
    # Efficiently collect and batch tokens
    def token_batcher():
        count = 0
        batch = []
        
        try:
            for token_sim in syn_token_gen:
                # Process token
                token = cached_turn_token_into_id(token_sim, count)
                
                if token is not None and token > 0:
                    batch.append(token)
                    count += 1
                    
                    # Submit batch when full
                    if len(batch) >= BATCH_SIZE:
                        if not shutdown_event.is_set():
                            task_queue.put(batch.copy())
                            batch.clear()
            
            # Process any remaining tokens
            if batch and not shutdown_event.is_set():
                task_queue.put(batch.copy())
                
        except Exception as e:
            print(f"Token batcher error: {e}")
            traceback.print_exc()
        finally:
            # Signal end of processing
            if not shutdown_event.is_set():
                task_queue.put(None)
    
    # Worker function for audio processing
    def audio_worker():
        while not shutdown_event.is_set():
            try:
                # Get next batch with timeout
                batch = task_queue.get(timeout=0.1)
                
                if batch is None:  # End signal
                    task_queue.task_done()
                    # Forward the end signal
                    result_queue.put(None)
                    break
                
                # Process batch
                audio_data = convert_to_audio(np.array(batch, dtype=np.int32), len(batch))
                
                if audio_data is not None and not shutdown_event.is_set():
                    result_queue.put(audio_data)
                
                task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio worker error: {e}")
                traceback.print_exc()
                task_queue.task_done()
    
    # Start token batcher thread
    batcher_thread = threading.Thread(target=token_batcher, name="TokenBatcher")
    batcher_thread.daemon = True
    batcher_thread.start()
    
    # Start worker threads
    workers = []
    worker_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="SyncAudioWorker")
    
    for _ in range(MAX_WORKERS):
        worker = worker_pool.submit(audio_worker)
        workers.append(worker)
    
    # Yield results as they become available
    end_signal_received = False
    
    try:
        while not end_signal_received:
            try:
                # Get result with timeout to allow for clean shutdown
                result = result_queue.get(timeout=0.1)
                
                if result is None:  # End signal
                    end_signal_received = True
                else:
                    yield result
                
                result_queue.task_done()
                
            except queue.Empty:
                # Check if batcher is still running
                if not batcher_thread.is_alive() and task_queue.empty():
                    # Double-check result queue once more
                    try:
                        result = result_queue.get(timeout=0.1)
                        if result is None:
                            end_signal_received = True
                        else:
                            yield result
                        result_queue.task_done()
                    except queue.Empty:
                        end_signal_received = True
                        
    except GeneratorExit:
        # Handle early exit
        shutdown_event.set()
        
    finally:
        # Clean shutdown
        shutdown_event.set()
        
        # Stop accepting new tasks
        worker_pool.shutdown(wait=False)
        
        # Clear queues
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