# DeepSeek with SGLang on Cerebrium

This example demonstrates how to deploy a DeepSeek model using the SGLang framework on Cerebrium's infrastructure, providing an OpenAI-compatible API interface.

## Overview

The implementation uses:

- [SGLang](https://github.com/sgl-project/sglang) - A framework for efficient LLM serving
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek) - An open source LLM
- [Cerebrium](https://cerebrium.ai) - ML deployment platform

The example uses the DeepSeek-R1-Distill-Llama-8B model by default, but can be configured to use the full DeepSeek-R1 model by uncommenting the relevant line in `main.py` and `cerebrium.toml`.

## How it Works

The code:

1. Initializes the SGLang Runtime with the specified DeepSeek model
2. Sets up streaming chat completion functionality
3. Handles chat messages using DeepSeek's chat template
4. Returns responses in an OpenAI-compatible streaming format

Key components:

- Uses SGLang's Runtime for efficient model serving
- Configurable parameters like temperature and top_p for generation
- OpenAI-compatible API with streaming responses
- Response format matches OpenAI's chat completions API structure

## Deployment

To deploy on Cerebrium:

1. Add your Huggingface token as a secret on your Cerebrium Dashboard

2. Deploy using:
   ```bash
   cerebrium deploy
   ```

## Configuration

The default configuration uses:

- Model: DeepSeek-R1-Distill-Llama-8B
- Tensor Parallel Size: 1 (increase to 8 for full R1 model)
- Temperature: 0.8
- Top-p: 0.95

To use the full DeepSeek-R1 model:

1. Uncomment the R1 model line in `main.py`
2. Change tp_size to 8 in the Runtime initialization
3. Change the CPU, Memory and Compute to H200 - this requires you to contact the Cerebrium team

## API Usage

The service provides an OpenAI-compatible API endpoint for chat completions. You can use it as a drop-in replacement for OpenAI's chat completion API.

Each response chunk follows OpenAI's format including:

- Message role and content
- Request ID
- Model information
- Completion status

You can use this API with any OpenAI-compatible client library by simply changing the base URL to your deployed endpoint.
