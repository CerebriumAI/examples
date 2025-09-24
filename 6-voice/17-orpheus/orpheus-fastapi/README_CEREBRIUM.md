# Orpheus FASTAPI

1. Git clone https://github.com/timonharz/Orpheus-FastAPI
2. Create a file called cerebrium.toml and add the following:

```
[cerebrium.deployment]
name = "17-orpheus-fastapi"
python_version = "3.11"
docker_base_image_url = "debian:bookworm-slim"
disable_auth = true
include = ['./*', 'main.py', 'cerebrium.toml']
exclude = ['.*']

[cerebrium.hardware]
cpu = 4.0
memory = 8.0
compute = "CPU"
provider = "aws"
region = "us-east-1"

[cerebrium.scaling]
min_replicas = 0
max_replicas = 2
cooldown = 30
replica_concurrency = 100
scaling_metric = "concurrency_utilization"

[cerebrium.runtime.custom]
port = 5005
entrypoint = ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5005"]


[cerebrium.dependencies.paths]
pip = "requirements.txt"

[cerebrium.dependencies.apt]
libportaudio2 = "latest"
```

3. Add a empty file called `main.py`. Don't worry about this.
4. To your requirements.txt, uncomment the line to install `torch torchvision torchaudio`
5. Update your .env file with your deployment url from your orpheus server (just fill in your project id) abd upload to your secrets on the cerebrium dashboard
6. Run `cerebrium deploy`
7. You should then be able to make the following CURL request

```
curl --location 'https://api.aws.us-east-1.cerebrium.ai/v4/p-xxxxxx/17-orpheus-fastapi/v1/audio/speech/stream' \
--header 'Authorization: Bearer <AUTH_TOKEN>' \
--header 'Content-Type: application/json' \
--data '{
    "input": "Your text to convert to speech",
    "model": "orpheus",
    "voice": "tara",
    "response_format": "wav",
    "speed": 1.0
  }'
```
