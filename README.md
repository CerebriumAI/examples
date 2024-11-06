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
3. [Deploying inferentia](1-getting-started/3-inferentia)
4. [CPU-only workloads](1-getting-started/4-cpu-only)

### 2. Advanced Concepts 🧠

1. [Deploy ComfyUI on Cerebrium](2-advanced-concepts/1-comfyui)
2. [Improve inference speed with VLLM](2-advanced-concepts/2-faster-inference-with-vllm)
3. [Loading model weights faster](2-advanced-concepts/3-loading-model-weights-faster)
4. [Multi-GPU inference](2-advanced-concepts/4-multi-gpu-inference)
5. [Stream results from Falcon 7B to a client](2-advanced-concepts/5-streaming-endpoint)
6. [How to generate images faster with SDXL](2-advanced-concepts/6-faster-image-generation)
7. [WebSockets](2-advanced-concepts/7-websockets)
8. [Implement simple output streaming](2-advanced-concepts/8-simple-streaming)
9. [Batching requests with vLLM](2-advanced-concepts/9-vllm-batching-gpu)
10. [Batching requests with transformers](2-advanced-concepts/10-transformers-batching-gpu)


### 3. Voice 🎤

1. [Transcription service using Whisper](3-voice/1-whisper-transcription)
2. [Create a realtime voice agent](3-voice/2-realtime-voice-agent)
3. [Create a voice agent that leverages current context with RAG](3-voice/3-voice-rag-agent)
4. [Create a WebSockets-based voice agent using Twilio](3-voice/4-twilio-voice-agent)
5. [Clone voices with XTTS](3-voice/5-xtts)

### 4. Large Language Models 🤖

1. [Create a Langchain QA system](4-large-language-models/1-langchain-QA)
2. [Create a virtual calendar assistant with Langsmith](4-large-language-models/2-tool-calling-langsmith)
3. [Build a code review tool with Winston](4-large-language-models/3-winston)
4. [Create a sales training tool with Mistral](4-large-language-models/4-sales-trainer)
5. [Create an OpenAI compatible endpoint with vLLM](4-large-language-models/5-openai-compatible-endpoint)
6. [Build your own OpenAI realtime API replacement](4-large-language-models/6-openai-realtime-api-comparison)

### 5. Image & Video 📸

1. [Build a ControlNet logo detection system](5-image-and-video/1-logo-controlnet)
2. [Refined image generation with SDXL](5-image-and-video/2-sdxl-refiner)
3. [Using SDXL Lightning for image processing](5-image-and-video/3-sdxl-lightning)
4. [Fast stable diffusion for image generation](5-image-and-video/4-fast-stable-diffusion)
5. [Regular stable diffusion for image generation](5-image-and-video/5-regular-stable-diffusion)
6. [Migrate your COG model to SDXL](5-image-and-video/6-cog-migration-sdxl)
7. [Find products for sale using a live video stream](5-image-and-video/7-ecommerce-live-stream)

### 6. Python apps 🌐

1. [Deploy FastAPI applications](6-python-apps/1-asgi-fastapi-server)
2. [Create ML web interfaces with Gradio](6-python-apps/2-asgi-gradio-interface)
3. [Implement batching with LitServe - CPU version](6-python-apps/3-litserve-batching-cpu)
4. [Implement batching with LitServe - GPU version](6-python-apps/4-litserve-batching-gpu)

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

Share your fork with us on our [Discord](https://discord.gg/ATj6USmeE2) & [Slack](https://join.slack.com/t/cerebriumworkspace/shared_invite/zt-1qojg3eac-q4xyu5O~MeniNIg2jNeadg) communities or on social media. Highly valuable examples for the community will be merged into the master repo.

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
- 💬 [Slack Community](https://join.slack.com/t/cerebriumworkspace/shared_invite/zt-1qojg3eac-q4xyu5O~MeniNIg2jNeadg)
- 📧 [Support Email](support@cerebrium.ai)