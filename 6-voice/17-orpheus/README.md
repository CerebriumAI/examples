# Orpheus Text-to-Speech

This service consists of two components that work together to provide text-to-speech capabilities:

## 1. Orpheus Server (Model Host)

Located in `/orpheus-server`

The Orpheus server component hosts the actual text-to-speech model and handles the core speech generation. See the [orpheus-server README](./orpheus-server/README.md) for detailed deployment instructions.

Key features:

- Hosts the Orpheus text-to-speech model
- Handles voice generation processing
- Manages different voice models
- Provides WebSocket endpoints for streaming audio

## 2. Orpheus FastAPI (API Layer)

Located in `/orpheus-fastapi`

The FastAPI component provides a REST API interface for interacting with the Orpheus service. See the [orpheus-fastapi README](./orpheus-fastapi/CEREBRIUM_README.md) for detailed deployment instructions.

Key features:

- RESTful API endpoints for text-to-speech conversion
- Handles request validation and processing
- Manages communication with the Orpheus server
- Provides streaming audio responses

## Deployment Order

1. First deploy the Orpheus server component following its README instructions
2. Once the server is running, deploy the FastAPI component and configure it to connect to your Orpheus server
3. Test the complete setup by making a request to the FastAPI endpoint

## Example Usage

Once both components are deployed, you can convert text to speech using:

```
curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/p-xxxxxx/17-orpheus-fastapi/v1/audio/speech/stream' \
--header 'Authorization: Bearer <AUTH_TOKEN>' \
--header 'Content-Type: application/json' \
--data '{
    "input": "Your text to convert to speech - is this working now",
    "model": "orpheus",
    "voice": "tara",
    "response_format": "wav",
    "speed": 1.0
  }'
```
