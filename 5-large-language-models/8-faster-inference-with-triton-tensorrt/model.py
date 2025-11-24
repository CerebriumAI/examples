"""
Triton Python Backend for TensorRT-LLM.

This module implements a Triton Inference Server Python backend that uses
TensorRT-LLM's PyTorch backend for optimized LLM inference. 
"""

import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from tensorrt_llm import LLM, SamplingParams, BuildConfig
from tensorrt_llm.plugin.plugin import PluginConfig
from transformers import AutoTokenizer
from pathlib import Path

# Model configuration
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_DIR = f"/persistent-storage/models/{MODEL_ID}"


def ensure_model_downloaded():
    """Check if model exists, download if not available."""
    model_path = Path(MODEL_DIR)
    
    # Check if model directory exists and has content
    if not model_path.exists() or not any(model_path.iterdir()):
        print("Model not found, downloading...")
        try:
            # Import download function from download_model
            from download_model import download_model
            download_model()
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print("✓ Model already exists")


class TritonPythonModel:
    """
    Triton Python Backend model for TensorRT-LLM inference.
    
    This class handles model initialization, inference requests, and cleanup.
    """
    
    def initialize(self, args):
        """
        Initialize the model - called once when Triton loads the model.
        
        Loads tokenizer and initializes TensorRT-LLM with PyTorch backend.
        """
        # Ensure model is downloaded before loading
        ensure_model_downloaded()
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        print("Initializing TensorRT-LLM...")
        
        plugin_config = PluginConfig.from_dict({
            "paged_kv_cache": True,  # Efficient memory usage for KV cache
        })
        
        build_config = BuildConfig(
            plugin_config=plugin_config,
            max_input_len=4096,
            max_batch_size=128,  # Matches Triton max_batch_size in config.pbtxt
        )
        
        self.llm = LLM(
            model=MODEL_DIR,
            build_config=build_config,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        print("✓ Model ready")
    
    def execute(self, requests):
        """
        Execute inference on batched requests.
        
        Triton automatically batches requests (up to max_batch_size: 32).
        This function processes the batch that Triton provides.
        """
        try:
            prompts = []
            sampling_params_list = []
            original_prompts = []  # Store original prompts to strip from output if needed
            
            # Extract data from each request in the batch
            for request in requests:
                try:
                    # Get input text - handle batched tensor structures
                    input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
                    text_array = input_tensor.as_numpy()
                    
                    # Extract text handling different array structures (batched vs non-batched)
                    if text_array.ndim == 0:
                        # Scalar
                        text = text_array.item()
                    elif text_array.dtype == object:
                        # Object dtype array (common for BYTES/STRING with batching)
                        text = text_array.flat[0] if text_array.size > 0 else text_array.item()
                    else:
                        # Regular array - get first element
                        text = text_array.flat[0] if text_array.size > 0 else text_array.item()
                    
                    # Decode if bytes, otherwise use as string
                    if isinstance(text, bytes):
                        text = text.decode('utf-8')
                    elif isinstance(text, np.str_):
                        text = str(text)
                    
                    # Get optional parameters with defaults
                    max_tokens = 1024
                    if pb_utils.get_input_tensor_by_name(request, "max_tokens") is not None:
                        max_tokens_array = pb_utils.get_input_tensor_by_name(request, "max_tokens").as_numpy()
                        max_tokens = int(max_tokens_array.item() if max_tokens_array.ndim == 0 else max_tokens_array.flat[0])
                    
                    temperature = 0.8
                    if pb_utils.get_input_tensor_by_name(request, "temperature") is not None:
                        temp_array = pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy()
                        temperature = float(temp_array.item() if temp_array.ndim == 0 else temp_array.flat[0])
                    
                    top_p = 0.95
                    if pb_utils.get_input_tensor_by_name(request, "top_p") is not None:
                        top_p_array = pb_utils.get_input_tensor_by_name(request, "top_p").as_numpy()
                        top_p = float(top_p_array.item() if top_p_array.ndim == 0 else top_p_array.flat[0])
                    
                    # Format prompt using chat template
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    prompts.append(prompt)
                    original_prompts.append(prompt)  # Store for potential stripping
                    sampling_params_list.append(SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    ))
                except Exception as e:
                    print(f"Error processing request: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    # Use default max_tokens instead of 1 to avoid single token output
                    prompts.append("")
                    original_prompts.append("")
                    sampling_params_list.append(SamplingParams(max_tokens=1024))
            
            # Batch inference
            if not prompts:
                return []
            
            outputs = self.llm.generate(prompts, sampling_params_list)

            # Create responses
            responses = []
            for i, output in enumerate(outputs):
                try:
                    # Extract generated text
                    generated_text = output.outputs[0].text
                    
                    # Remove the prompt from generated text if it's included
                    if original_prompts[i] and original_prompts[i] in generated_text:
                        generated_text = generated_text.replace(original_prompts[i], "").strip()
                    
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor(
                            "text_output",
                            np.array([generated_text.encode('utf-8')], dtype=object)
                        )]
                    ))
                except Exception as e:
                    print(f"Error creating response {i}: {e}", flush=True)
                    responses.append(pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor(
                            "text_output",
                            np.array([f"Error: {str(e)}".encode('utf-8')], dtype=object)
                        )]
                    ))
            
            return responses
            
        except Exception as e:
            print(f"Error in execute: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Return error responses
            return [
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor(
                        "text_output",
                        np.array([f"Batch error: {str(e)}".encode('utf-8')], dtype=object)
                    )]
                )
                for _ in requests
            ]
    
    def finalize(self):
        """
        Cleanup when model is being unloaded.
        
        Shuts down the TensorRT-LLM engine and clears GPU memory.
        """
        if hasattr(self, 'llm'):
            self.llm.shutdown()
            torch.cuda.empty_cache()