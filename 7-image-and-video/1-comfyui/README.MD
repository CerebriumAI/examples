# ComfyUI API Wrapper

This project provides a FastAPI-based wrapper around ComfyUI, making it easy to run Stable Diffusion workflows programmatically through a RESTful API.

## Overview

The ComfyUI API Wrapper starts a ComfyUI server in the background and exposes endpoints to:
- Run Stable Diffusion workflows with custom parameters
- Check server health status
- Dynamically load and configure models

The API automatically handles model downloading, workflow template substitution, and converting the generated images to base64 for easy integration with other services.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/comfyui-api-wrapper.git
cd comfyui-api-wrapper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Clone ComfyUI repository (if not already included):
```bash
git clone https://github.com/comfyanonymous/ComfyUI
pip install -r ComfyUI/requirements.txt
```

## Usage

### Starting the Server

Run the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8765
```

The server will:
1. Load the workflow template
2. Launch ComfyUI in the background
3. Download required models (if not already present)
4. Begin accepting API requests

### API Endpoints

#### Run a Workflow

```
POST /run
```

This endpoint accepts a JSON payload with parameters that will be substituted into the workflow template.

**Example Request:**

```bash
curl -X POST http://localhost:8765/run \
  -H "Content-Type: application/json" \
  -d '{
    "seed": 12345,
    "positive_prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
    "negative_prompt": "worst quality, low quality, blurry, cropped, lowres",
    "controlnet_image": "https://example.com/image.png"
  }'
```

**Example Response:**

```json
{
  "result": [
    {
      "node_id": "18",
      "data": "iVBORw0KGgoAAAANSUhEUgAA...",
      "format": "png"
    },
    {
      "node_id": "15",
      "data": "R0lGODlhAQABAIAA...",
      "format": "png"
    }
  ],
  "status": "success"
}
```

The `data` field contains the base64-encoded image data that can be directly embedded in HTML or saved to a file.

## Customizing Workflows

The API uses `workflow_api.json` as a template for image generation. You can customize this file to use different models, parameters, or node configurations.

The template uses placeholders in the format `{{parameter_name}}` which are replaced with values from your API request.

### Default Placeholders:

- `{{seed}}` - Random seed for generation stability
- `{{positive_prompt}}` - Text prompt describing what you want to generate
- `{{negative_prompt}}` - Text prompt describing what you want to avoid
- `{{controlnet_image}}` - URL or base64 data of an image for ControlNet

## Advanced Usage

### Custom Models

To use custom models, modify the `model.json` file to include URLs and paths for additional models:

```json
[
  {
    "url": "https://huggingface.co/your-model-url/resolve/main/model.safetensors",
    "path": "models/checkpoints/your_model.safetensors"
  },
  {
    "url": "https://github.com/username/custom-node-repo.git",
    "path": "custom_nodes"
  }
]
```