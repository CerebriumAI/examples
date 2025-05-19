# Kokoro Text-to-Speech API

This repository contains a FastAPI implementation of the Kokoro text-to-speech model, which provides high-quality multilingual speech synthesis. The model supports multiple voices and languages, making it suitable for various applications requiring natural-sounding speech output.

## Deployment Instructions

1. Clone the repository: `git clone https://github.com/remsky/Kokoro-FastAPI.git`
2. `cd Kokoro-FastAPI`
3. Create a cerebrium.toml file with the following:
```
[cerebrium.deployment]
name = "13-kokoro"
python_version = "3.11"
docker_base_image_url = "debian:bookworm-slim"
disable_auth = true
include = ['./*', 'main.py', 'cerebrium.toml']
exclude = ['.*']

[cerebrium.hardware]
cpu = 4.0
memory = 12.0
compute = "AMPERE_A10"

[cerebrium.scaling]
min_replicas = 0
max_replicas = 5
cooldown = 30
replica_concurrency = 1
response_grace_period = 900
scaling_metric = "concurrency_utilization"
scaling_target = 100
scaling_buffer = 0
roll_out_duration_seconds = 0

[cerebrium.runtime.custom]
port = 8880
dockerfile_path = "./docker/gpu/Dockerfile"
```

4. Create a main.py - it can be empty.
5. Run: `cerebrium deploy`
6. Your API is up! Update test.py to have your deployment url in the format: https://api.cortex.cerebrium.ai/v4/p-xxxxxx/13-kokoro/v1
7. Run `python test.py` to test your application 