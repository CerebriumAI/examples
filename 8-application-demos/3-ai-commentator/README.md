# Real-time AI Commentator

This project demonstrates how to build a real-time AI commentator system that can provide live commentary on video streams. The system combines several technologies to create an engaging experience:

- [LiveKit](https://livekit.io/) for real-time video streaming and WebRTC capabilities
- [Cerebrium](https://www.cerebrium.ai/) for running AI vision models to analyze video frames
- [Cartesia](https://cartesia.ai/) for natural text-to-speech conversion

## How It Works

1. The system receives a video stream through LiveKit's WebRTC infrastructure
2. Video frames are captured and processed at regular intervals (every 2 seconds)
3. The frames are sent to a vision model deployed on Cerebrium that analyzes the content
4. Based on the analysis, the AI generates natural commentary about what it sees
5. The commentary is converted to speech using Cartesia's text-to-speech API
6. The generated audio is streamed back in real-time through LiveKit


For more details on implementation and setup, check out the [full blog post](https://www.cerebrium.ai/blog/creating-a-realtime-ai-commentator-with-cerebrium-livekit-and-cartesia) or the [frontend code repository](https://github.com/CerebriumAI/realtime-ai-commentator).
