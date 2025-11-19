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

# Model configuration
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_DIR = f"/persistent-storage/models/{MODEL_ID}"


class TritonPythonModel:
    """
    Triton Python Backend model for TensorRT-LLM inference.
    
    This class handles model initialization, inference requests, and cleanup.
    """
    
    def initialize(self, args):
        """
        Initialize the model using TensorRT-LLM's PyTorch backend.
        
        This method is called once when the model is loaded. It:
        1. Loads the tokenizer from HuggingFace
        2. Initializes TensorRT-LLM with PyTorch backend (loads model directly)
        
        Args:
            args: Dictionary containing model configuration from Triton
        """
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        print("Initializing TensorRT-LLM with PyTorch backend...")

        
        plugin_config = PluginConfig.from_dict({
            "paged_kv_cache": True,  # Efficient memory usage for KV cache
        })
        
        # Configure build parameters
        build_config = BuildConfig(
            plugin_config=plugin_config,
            max_input_len=4096,      # Maximum input sequence length
            max_batch_size=1,         # Batch size per request
        )
        
        self.llm = LLM(
            model=MODEL_DIR,  # HuggingFace model path
            build_config=build_config,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        print("✓ Model ready")
    
    def execute(self, requests):
        """
        Execute inference requests.
        
        Processes one or more inference requests, generating text responses
        using the TensorRT-LLM model.
        
        Args:
            requests: List of InferenceRequest objects from Triton
            
        Returns:
            List of InferenceResponse objects with generated text
        """
        responses = []
        
        for request in requests:
            try:
                # Extract input text
                input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
                text = input_tensor.as_numpy()[0].decode('utf-8')
                
                # Extract optional parameters (with defaults)
                max_tokens = 1024
                temperature = 0.8
                top_p = 0.95
                
                max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")
                if max_tokens_tensor is not None:
                    max_tokens = int(max_tokens_tensor.as_numpy()[0])
                
                temp_tensor = pb_utils.get_input_tensor_by_name(request, "temperature")
                if temp_tensor is not None:
                    temperature = float(temp_tensor.as_numpy()[0])
                
                top_p_tensor = pb_utils.get_input_tensor_by_name(request, "top_p")
                if top_p_tensor is not None:
                    top_p = float(top_p_tensor.as_numpy()[0])
                
                # Format prompt using Llama chat template
                messages = [{"role": "user", "content": text}]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Configure sampling parameters
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                
                # Generate text
                output = self.llm.generate(prompt, sampling_params)
                generated_text = output.outputs[0].text
                
                # Create response tensor
                output_tensor = pb_utils.Tensor(
                    "text_output",
                    np.array([generated_text.encode('utf-8')], dtype=object)
                )
                
                # Create inference response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor]
                )
                responses.append(inference_response)
                
            except Exception as e:
                # Handle errors gracefully
                print(f"Error processing request: {e}")
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Error: {str(e)}")
                )
                responses.append(error_response)
        
        return responses
    
    def finalize(self):
        """
        Cleanup when model is being unloaded.
        
        Shuts down the TensorRT-LLM engine and clears GPU memory.
        """
        if hasattr(self, 'llm'):
            self.llm.shutdown()
            torch.cuda.empty_cache()

