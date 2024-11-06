# OpenAI Realtime API Alternative Showcase

This project demonstrates an alternative to the OpenAI Realtime API, offering a faster, more cost-effective, and highly customizable solution. The implementation showcases two main components: a combined service pipeline (`main.py`) and a custom Text-to-Speech service (`rime.py`).

## Features

- Switchable service between OpenAI Realtime and a custom pipeline
- Integration with Daily.co for audio streaming and room management
- Custom Rime TTS service for high-quality voice synthesis
- Flexible context management and function calling
- Real-time voice activity detection (VAD) and transcription

## Main Components

### main.py

The `main.py` file contains the core logic for the alternative service. Key features include:

- Dynamic switching between OpenAI Realtime and custom services
- Integration with Daily.co for audio transport
- Parallel pipeline architecture for efficient processing
- Context management for maintaining conversation state
- Function registration for service switching
- Room creation and token management for Daily.co integration

### rime.py

The `rime.py` file implements a custom Text-to-Speech service using the Rime API. Features include:

- High-quality voice synthesis
- Customizable voice and model selection
- Streaming audio output for low-latency responses
- Metrics generation for performance analysis

## Advantages Over OpenAI Realtime API

1. **Speed**: The custom pipeline can potentially offer faster response times, especially with the Rime TTS service.
2. **Cost-effectiveness**: By using a combination of services and custom implementations, this solution can be more cost-effective for high-volume usage.
3. **Customizability**: The modular design allows for easy integration of different services and models, tailoring the solution to specific needs.
4. **Flexibility**: The ability to switch between services in real-time provides greater control over the conversation flow and quality.

## Getting Started

1. Set up the required Secrets in your Cerebrium Dashboard:
   - `OPENAI_API_KEY`
   - `RIME_API_KEY`
   - `DAILY_TOKEN`

2. Install the required dependencies (listed in `cerebrium.toml`).

3. Run the `main.py` script to start the service.

4. Use the provided functions to create a room and initiate a conversation.

## Customization

The modular design of this project allows for easy customization and extension. You can add new services, modify the pipeline structure, or integrate different APIs to suit your specific requirements.

## Note

This project is a demonstration and may require further optimization and error handling for production use. Always ensure proper security measures are in place when dealing with API keys and user data.
