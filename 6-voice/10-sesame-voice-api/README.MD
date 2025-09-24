# Sesame Conversational Speech Model (CSM) API

This repository contains a Cerebrium deployment for Sesame AI's Conversational Speech Model (CSM-1B), enabling you to create hyper-realistic AI-generated speech with natural conversational elements like hesitations, filler words, and human-like intonation.

## Overview

Unlike traditional text-to-speech systems that sound robotic, CSM generates remarkably human-like speech that includes natural pauses, "umms", "uhhs," expressive mouth sounds, and subtle intonation changes characteristic of human conversation.

The model combines a Llama 3.2 architecture with specialized audio tokenization, taking into account both the text to be spoken and conversational context to maintain a coherent speaking style.

## Prerequisites

Before getting started, you'll need:

- A [Cerebrium](https://dashboard.cerebrium.ai/register) account
- A [Huggingface](https://huggingface.co/signup) account and API key
- Access to the [CSM-1B model](https://huggingface.co/sesame/csm-1b)
- Access to the [Llama 3.2 1B model](https://huggingface.co/meta-llama/Llama-3.2-1B)

## Repository Structure

```
├── cerebrium.toml       # Cerebrium deployment configuration
├── main.py              # Main API implementation
├── models.py            # Model architecture definitions
├── generator.py         # Speech generation logic
├── requirements.txt     # Python dependencies
├── watermarking.py      # Watermarking for generated audio
├── test.py              # Script to test the deployed API
```

## Setup and Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sesame-csm-cerebrium.git
cd sesame-csm-cerebrium
```

### 2. Install Cerebrium CLI

```bash
pip install cerebrium --upgrade
```

### 3. Set Environment Variables

In your Cerebrium dashboard under the Secrets section, add the following:

- `HF_TOKEN`: Your Huggingface access token
- `HF_HUB_ENABLE_HF_TRANSFER=1`: Enables faster downloads from Huggingface
- `HF_HOME=/persistent-storage/.cache/huggingface/hub`: Sets caching to Cerebrium's persistent volume

### 4. Deploy to Cerebrium

```bash
cerebrium login
cerebrium deploy
```

The deployment process will:

- Upload your files
- Build a container with dependencies
- Provision the A10 GPU
- Deploy your app and test it
- Set up the API endpoint

## Using the API

### API Endpoint

After deployment, your API will be available at:

```
https://api.aws.us-east-1.cerebrium.ai/v4/[YOUR_PROJECT_ID]/10-sesame-voice-api/generate_audio
```

### Request Format

Send a POST request with your text in the following format:

```json
{
  "text": "Your text to be converted to speech goes here. You can include, uh, filler words and they will sound natural."
}
```

### Response Format

The API returns base64-encoded WAV audio:

```json
{
  "audio_data": "base64-encoded-audio-content",
  "format": "wav",
  "encoding": "base64"
}
```

### Testing with the Included Script

This repository includes a `test.py` script that demonstrates how to call the API and save the generated audio. To use it:

1. First, install the required dependencies:

```bash
pip install requests soundfile
```

2. Open `test.py` and update the following variables with your specific information:

```python
# Replace with your actual endpoint and API key
url = "https://api.aws.us-east-1.cerebrium.ai/v4/[YOUR_PROJECT_ID]/10-sesame-voice-api/generate_audio"
api_key = "[YOUR_API_KEY]"  # Replace with your Cerebrium API key
```

3. Optionally, modify the test text to try different phrases:

```python
# The text we want to convert to speech
test_text = "Your custom text goes here. You can include, uh, filler words for natural speech."
```

4. Run the script:

```bash
python test.py
```

5. If successful, you'll see output similar to:

```
Sending text to be converted: "Cerebrium is a, uh, really great cloud platform for deploying your voice models. It's easy to use and the team is very helpful."
Generated audio in 31.23 seconds!
Audio saved to output.wav
Audio length: 7.84 seconds
```

The script will save the generated audio as `output.wav` in your current directory, which you can play with any audio player.

## Configuration Options

In the `main.py` file, you can modify several parameters:

- `speaker`: Choose between speaker 0 and 1 (different voice characteristics)
- `max_audio_length_ms`: Maximum length of generated audio (default: 10,000 ms)
- `temperature`: Controls randomness - higher values produce more variation (default: 0.9)

## Ethical Considerations

Sesame has included audio watermarking to help identify AI-generated speech. This is important for transparency and helps prevent potential misuse. The watermarking is imperceptible to human listeners but can be detected by specialized software.

## Troubleshooting

- **Long Generation Times**: The first request after deployment may take longer as the model is loaded into memory
- **Memory Issues**: If you encounter memory problems, try reducing the `max_audio_length_ms` parameter
- **Authentication Errors**: Ensure your Huggingface token has the correct permissions for the gated models

## License

This project uses components from Sesame AI Labs under the Apache 2.0 license.

## Acknowledgements

- [Sesame AI Labs](https://www.sesame.com/) for creating and releasing CSM
