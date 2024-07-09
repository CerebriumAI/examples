# Real-time Voice AI Agent

This repository contains the code to build the demo application [here](https://fastvoiceagent.cerebrium.ai/). We create  a real-time voice AI agent that can respond to any query via speech, in speech, in ~500ms. This is an extremely flexible implementation where you have the ability to swap in any Large Language model, Text-to-speech (TTS) model and Speech-to-text (STT) model of your liking. This is extremely useful for use cases involving voice such as customer service bots, receptionists and many more.

You can visit the full documentation [here](https://docs.cerebrium.ai/v4/examples/realtime-voice-agents)

## Overview

In order to create this application, we use Pipecat, an open source framework for voice and multimodal conversational AI that handles some of the functionality we might need such as handling user interruptions, dealing with audio data etc. We will speak with our voice AI agent via a WebRTC transport, using Daily (the creators of Pipecat) and will deploy this application on Cerebrium to show how it handles deploying and scaling our application seamlessly.

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

