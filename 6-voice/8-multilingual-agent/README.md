# Real-time Voice AI Agent

This repository demonstrates how to build a specialized French-speaking voice AI agent. While many voice AI solutions exist, handling multilingual applications effectively remains a challenge. This implementation focuses on achieving high accuracy in French language processing while maintaining low latency (~500ms) and reasonable costs.

Our solution addresses several key challenges in multilingual voice AI:
- Utilizing Cartesia's enhanced TTS capabilities with support for French accents and gendered voices
- Implementing optimized STT (Speech-to-text) processing to minimize Word Error Rate (WER) while maintaining low latency
- Leveraging modern LLM capabilities for French language understanding and response generation

You can visit the full documentation [here](https://docs.cerebrium.ai/v4/examples/realtime-voice-agents)

## Overview

To create this application, we use Pipecat, an open source framework for voice and multimodal conversational AI
that handles some of the functionality we might need such as handling user interruptions, dealing with audio data etc.
We will speak with our voice AI agent via a WebRTC transport, using Daily (the creators of Pipecat) and will deploy this
application on Cerebrium to show how it handles deploying and scaling our application seamlessly.

## Features

- Real-time voice interaction with response times around 500ms
- Flexible integration with various Large Language Models (LLMs), TTS, and STT models
- Utilizes Pipecat for handling voice and multimodal conversational AI
- WebRTC transport using Daily for communication
- Seamless deployment and scaling with Cerebrium

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements

- [Pipecat](https://github.com/daily-co/pipecat) - Open source framework for voice and multimodal conversational AI
- [Daily](https://www.daily.co/) - WebRTC platform for real-time video and audio communication
- [Cerebrium](https://www.cerebrium.ai/) - Platform for deploying and scaling AI applications

