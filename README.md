<p align="center">
  <a href="https://cerebrium.ai">
    <img src="https://cerebrium-assets.s3.eu-west-1.amazonaws.com/github-examples.png">
  </a>
</p>

Welcome to Cerebrium's official examples repository! This collection of examples helps you get started with building Machine Learning / AI applications on the platform. Whether you're looking to deploy LLMs, process voice data, or handle image and video tasks, you'll find practical, ready-to-use examples here.

## How to Use This Repository

Each example is a self-contained project that demonstrates a specific use case. You can explore the examples in any order, depending on your interests and needs. Each example includes detailed instructions on how to deploy the application on the Cerebrium platform.

Deploy each example by cloning the repo and running the `cerebrium deploy` command in each example folder.

## Categories

We've split each of the examples by category to make them easier to find.

### 1. Getting started 🚀

1. [Deploy your first model](1-getting-started/1-first-cortex-deployment)
2. [Managing secrets and configurations](1-getting-started/2-using-cerebrium-secrets)
3. [CPU-only workloads](1-getting-started/3-cpu-only)

### 2. Advanced Concepts 🧠

1. [Improve inference speed with VLLM](2-advanced-concepts/1-faster-inference-with-vllm)
2. [Deploying Inferentia](2-advanced-concepts/2-inferentia)
3. [Loading model weights faster](2-advanced-concepts/3-loading-model-weights-faster)
4. [Multi-GPU inference](2-advanced-concepts/4-multi-gpu-inference)

### 3. Endpoints 🛤

1. [WebSockets](3-endpoints/1-websockets)
2. [Implement simple output streaming](3-endpoints/2-simple-streaming)

### 4. Integrations 🤝

1. [Create a Langchain QA system](4-integrations/1-langchain-QA)
2. [Create a virtual calendar assistant with Langsmith](4-integrations/2-tool-calling-langsmith)

### 5. Large Language Models 📚

1. [Create an OpenAI compatible endpoint with vLLM](5-large-language-models/1-openai-compatible-endpoint)
2. [Stream results from Falcon 7B to a client](5-large-language-models/2-streaming-endpoint)
3. [Build a code review tool with Winston](5-large-language-models/3-winston)

### 6. Voice 🎤

1. [Transcription service using Whisper](6-voice/1-whisper-transcription)
2. [Create a realtime voice agent](6-voice/2-realtime-voice-agent)
3. [Create a voice agent that leverages current context with RAG](6-voice/3-voice-rag-agent)
4. [Create a WebSockets-based voice agent using Twilio](6-voice/4-twilio-voice-agent)
5. [Clone voices with XTTS](6-voice/5-xtts)
6. [Build your own OpenAI realtime API replacement](6-voice/6-openai-realtime-api-comparison)
7. [Outbound LiveKit agent](6-voice/7-outbound-livekit-agent)
8. [Real-time Voice AI Agent](6-voice/8-multilingual-agent)
9. [Transcribing with FasterWhisper](6-voice/9-faster-whisper)
10. [Sesame Conversational Speech Model (CSM) API](6-voice/10-sesame-voice-api)

### 7. Image & Video 📸

1. [Deploy ComfyUI on Cerebrium](7-image-and-video/1-comfyui)
2. [Build a ControlNet logo detection system](7-image-and-video/2-logo-controlnet)
3. [Refined image generation with SDXL](7-image-and-video/3-sdxl-refiner)
4. [Using SDXL Lightning for image processing](7-image-and-video/4-sdxl-lightning)
5. [Fast stable diffusion for image generation](7-image-and-video/5-fast-stable-diffusion)
6. [Regular stable diffusion for image generation](7-image-and-video/6-regular-stable-diffusion)
7. [How to generate images faster with SDXL](7-image-and-video/7-faster-image-generation)

### 8. Migrations 🚚

1. [Migrate your COG model to SDXL](8-migrations/1-cog-migration-sdxl)

### 9. Application demos 🎬

1. [Create a sales training tool with Mistral](8-application-demos/1-sales-trainer)
2. [Find products for sale using a live video stream](8-application-demos/2-ecommerce-live-stream)
3. [AI commentator](8-application-demos/3-ai-commentator)
4. [AI coding agent](8-application-demos/4-ai-coding-agent)

### 10. Batching 📦

1. [Implement batching with LitServe - CPU version](10-batching/1-litserve-batching-cpu)
2. [Implement batching with LitServe - GPU version](10-batching/2-litserve-batching-gpu)
3. [Batching requests with vLLM](10-batching/3-vllm-batching-gpu)
4. [Batching requests with transformers](10-batching/4-transformers-batching-gpu)

### 11. Python apps 🌐

1. [Deploy FastAPI applications](11-python-apps/1-asgi-fastapi-server)
2. [Create ML web interfaces with Gradio](11-python-apps/2-asgi-gradio-interface)

## How to Contribute 🤝

We love contributions! Here's how you can contribute to our examples repository:

- Fork the repository
- Create a new branch for your example
- Add your example following our template
- Include a clear README with:
  - Description of the example
  - Requirements
  - Step-by-step setup instructions
  - Expected outputs
  - (Optional) Link to a blog post or tutorial video

Share your fork with us on our [Discord](https://discord.gg/ATj6USmeE2) community or on social media. Highly valuable examples for the community will be merged into the master repo.

#### 🎁 Get Free Swag!

For each successful contribution, we'll send you exclusive Cerebrium swag! To be eligible:

- Your PR must be merged
- Fill out the contributor form with your shipping details
- Bonus swag for contributions that include:
  - A blog post explaining your example
  - A tutorial video demonstrating your example

#### 🦮 Contribution Guidelines

- Ensure your example is well-documented
- Make sure that your example deploys successfully
- Add appropriate error handling
- Follow our code style as much as possible
- Test your example thoroughly
- Update the main README.md to include your example

## Support 🛟

- 📚 [Documentation](https://docs.cerebrium.ai)
- 💬 [Discord Community](https://discord.gg/ATj6USmeE2)
- 📧 [Support Email](mailto:support@cerebrium.ai)
