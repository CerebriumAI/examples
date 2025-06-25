# Real-time Voice AI Agent using LiveKit

This repository contains the code to build a real-time voice AI agent that can respond to any query via speech with latency around 500ms. This is an extremely flexible implementation where you have the ability to swap in any Large Language Model (LLM), Text-to-Speech (TTS) model and Speech-to-Text (STT) model of your choice. This makes it ideal for use cases like customer service bots, receptionists and many more.

## Overview
The application is built using LiveKit, which provides both the media transport layer and agent framework needed for real-time voice interactions. LiveKit handles critical functionality like noise/echo cancellation, end-of-turn detection, and user interruptions. The voice agent communicates via WebRTC transport, and the entire application is deployed on Cerebrium for optimal performance and scalability.

The system consists of five core components:
- Speech-to-Text (STT) for transcribing incoming audio
- Large Language Model (LLM) for generating responses  
- Text-to-Speech (TTS) for converting replies to audio
- LiveKit Agent framework for business logic and service orchestration
- LiveKit media server for real-time voice streaming

## Features
- Real-time voice interaction with end-to-end latency around 500ms
- Flexible integration with various LLM, TTS, and STT models
- Built on LiveKit's agent framework for robust conversational AI
- WebRTC-based media transport using LiveKit
- Global deployment and seamless scaling with Cerebrium
- Cost-effective at around $0.03 per minute per call

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Acknowledgements
- LiveKit - Open source WebRTC platform and agent framework
- Cerebrium - Platform for deploying and scaling AI applications globally