# Fast Stable Diffusion

This example includes the model weights in the base image, which loads faster. The total response time is around 18 seconds.

The key part is the shell_commands in the cerebrium.toml file

```toml
shell_commands = ["export HF_HOME=/cortex/app/.cache/huggingface","python3 -c \"import torch; from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', torch_dtype=torch.float16)\""]
```

This will download the model at build time.

```python
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, local_files_only=True)
```

`local_files_only=True` will ensure that the model is loaded from the cache and not downloaded again.

## Calling the API

```bash
curl --location 'https://dev-api.cortex.cerebrium.ai/v4/YOUR-PROJECT_ID/22-fast-stable-diffusion/predict' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer YOUR_TOKEN' \
--data '{
    "item": {
        "prompt": "A golden retriever puppy sitting in the grass."
    }
}'```