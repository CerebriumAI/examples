import requests
import json
import base64
import soundfile as sf
import io
import time

# Replace with your actual endpoint and API key
url = "https://api.aws.us-east-1.cerebrium.ai/v4/[YOUR_PROJECT_ID]/10-sesame-api/generate_audio"
api_key = "[YOUR_API_KEY]"  # Replace with your Cerebrium API key

# The text we want to convert to speech
test_text = (
    "Cerebrium is a, uh, really great cloud platform for deploying your voice models. It's easy to use and "
    "the team is very helpful."
)

# Prepare the request
payload = json.dumps({"text": test_text})
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# Time the request
print(f'Sending text to be converted: "{test_text}"')
start_time = time.time()
response = requests.request("POST", url, headers=headers, data=payload)
end_time = time.time()

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
    print(f"Generated audio in {end_time - start_time:.2f} seconds!")

    # Convert base64 to audio file
    audio_data = base64.b64decode(result["result"]["audio_data"])
    audio_buffer = io.BytesIO(audio_data)
    audio, rate = sf.read(audio_buffer)

    # Save to file
    output_file = "output.wav"
    sf.write(output_file, audio, rate)
    print(f"Audio saved to {output_file}")
    print(f"Audio length: {len(audio) / rate:.2f} seconds")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
