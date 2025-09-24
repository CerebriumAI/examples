import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from io import BytesIO
from typing import Dict, Any, Tuple, List, Optional

import requests
import websocket
from PIL import Image

# Configure logging
logger = logging.getLogger("comfyui-helpers")

# Constants
COMFYUI_DIR = "ComfyUI"
PERSISTENT_STORAGE_DIR = "/persistent-storage"
REQUEST_TIMEOUT = 30


def create_request_session() -> requests.Session:
    """Create a session with retry capability."""
    session = requests.Session()
    return session


# Global session for reuse
http_session = create_request_session()


def download_model(model_url: str, destination_path: str) -> bool:
    """Download a model file."""
    logger.info(f"Downloading model {model_url} to directory: {destination_path}")
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Stream the download
        response = http_session.get(model_url, stream=True, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Open the destination file and write the content in chunks
        with open(destination_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)

        logger.info(f"Successfully downloaded file to: {destination_path}")
        return True

    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return False


def download_tempfile(
    file_url: str, filename: str
) -> Tuple[Optional[str], Optional[tempfile.NamedTemporaryFile]]:
    """Download file to a temporary location."""
    temp_file = None
    try:
        # Extract file extension
        file_ext = os.path.splitext(filename)[1] or ".tmp"

        # Download
        response = http_session.get(file_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(response.content)
        temp_file.close()

        return temp_file.name, temp_file

    except Exception as e:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        logger.error(f"Error downloading file {file_url}: {str(e)}")
        return None, None


def add_custom_node(git_url: str) -> bool:
    """Clone a custom node repository."""
    try:
        # Parse repo name from URL
        repo_name = git_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        target_dir = f"{COMFYUI_DIR}/custom_nodes/{repo_name}"

        # Check if already cloned
        if os.path.exists(target_dir):
            logger.info(f"Custom node already exists: {target_dir}")
            return True

        # Clone the repository
        logger.info(f"Cloning custom node: {git_url}")
        subprocess.run(
            ["git", "clone", git_url, target_dir, "--recursive", "--depth=1"],
            check=True,
        )

        logger.info(f"Successfully cloned {git_url} to {target_dir}")
        return True

    except Exception as e:
        logger.error(f"Error adding custom node {git_url}: {str(e)}")
        return False


def setup_comfyui(original_working_directory: str, data_dir: str) -> None:
    """Set up ComfyUI with required models and custom nodes."""
    try:
        # Ensure persistent storage directory exists
        os.makedirs(PERSISTENT_STORAGE_DIR, exist_ok=True)

        # Find model.json path
        model_json_path = os.path.join(
            original_working_directory, data_dir, "model.json"
        )
        if not os.path.exists(model_json_path):
            alternate_paths = [
                "model.json",
                os.path.join(original_working_directory, "model.json"),
            ]
            for path in alternate_paths:
                if os.path.exists(path):
                    model_json_path = path
                    break

        if not os.path.exists(model_json_path):
            logger.error("Could not find model.json")
            raise FileNotFoundError("model.json not found")

        # Load model definitions
        with open(model_json_path, "r") as file:
            data = json.load(file)

        # Process each model definition
        for model in data:
            if not isinstance(model, dict) or "url" not in model or "path" not in model:
                continue

            if model.get("path") == "custom_nodes":
                # Install custom nodes
                if "url" in model and model["url"]:
                    add_custom_node(model["url"])
            else:
                # Regular model file
                comfy_destination_path = os.path.join(COMFYUI_DIR, model.get("path"))

                # Skip if already exists
                if os.path.exists(comfy_destination_path):
                    continue

                # Download if needed
                download_model(model.get("url"), comfy_destination_path)

        logger.info("Model setup complete, starting ComfyUI server")

        # Run ComfyUI server
        subprocess.run([sys.executable, "main.py"], cwd=COMFYUI_DIR, check=True)

    except Exception as e:
        logger.error(f"Error setting up ComfyUI: {str(e)}")
        raise RuntimeError(f"Failed to set up ComfyUI: {str(e)}")


def queue_prompt(
    prompt: Dict[str, Any], client_id: str, server_address: str
) -> Dict[str, Any]:
    """Queue a prompt with the ComfyUI server."""
    try:
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode("utf-8")

        url = f"http://{server_address}/prompt"
        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            result = json.loads(response.read())
            logger.info(
                f"Successfully queued prompt, ID: {result.get('prompt_id', 'unknown')}"
            )
            return result

    except Exception as e:
        logger.error(f"Error queuing prompt: {str(e)}")
        raise RuntimeError(f"Error queuing prompt: {str(e)}")


def get_image(
    filename: str, subfolder: str, folder_type: str, server_address: str
) -> bytes:
    """Get an image from the ComfyUI server."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)

    try:
        with urllib.request.urlopen(
            f"http://{server_address}/view?{url_values}", timeout=REQUEST_TIMEOUT
        ) as response:
            return response.read()

    except Exception as e:
        logger.error(f"Error getting image {filename}: {str(e)}")
        raise RuntimeError(f"Error retrieving image: {str(e)}")


def get_history(prompt_id: str, server_address: str) -> Dict[str, Any]:
    """Get execution history for a prompt from the ComfyUI server."""
    try:
        with urllib.request.urlopen(
            f"http://{server_address}/history/{prompt_id}", timeout=REQUEST_TIMEOUT
        ) as response:
            output = json.loads(response.read())
            if prompt_id not in output:
                logger.warning(f"Prompt ID {prompt_id} not found in history")
                return {}
            return output[prompt_id]["outputs"]

    except Exception as e:
        logger.error(f"Error getting history for prompt {prompt_id}: {str(e)}")
        raise RuntimeError(f"Error retrieving execution history: {str(e)}")


def get_images(
    ws, prompt: Dict[str, Any], client_id: str, server_address: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Queue prompt and wait for execution to complete, then retrieve all output images."""
    # Queue the prompt and get its ID
    prompt_id = queue_prompt(prompt, client_id, server_address)["prompt_id"]
    logger.info(f"Queued prompt with ID: {prompt_id}")

    output_images = {}
    timeout = 300  # 5 minutes timeout
    start_time = time.time()

    # Wait for execution to complete via WebSocket
    try:
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Execution timed out after {timeout} seconds")

            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue

            if isinstance(out, str):
                try:
                    message = json.loads(out)
                except json.JSONDecodeError:
                    continue

                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        logger.info(f"Execution completed for prompt {prompt_id}")
                        break
            else:
                continue  # Ignore binary data (previews)

        # Retrieve history data
        history = get_history(prompt_id, server_address)
        if not history:
            return {}

        # Process all nodes in the history
        for node_id in history:
            node_output = history[node_id]

            # Process image outputs
            if "images" in node_output and node_output["images"]:
                outputs = []
                for image in node_output["images"]:
                    # For temporary/preview images, fetch the data
                    if image.get("type") == "temp":
                        try:
                            image_data = get_image(
                                image["filename"],
                                image["subfolder"],
                                image["type"],
                                server_address,
                            )
                            outputs.append(
                                {"filename": image.get("filename"), "data": image_data}
                            )
                        except Exception:
                            continue
                    else:
                        # For saved images, just record the filename
                        outputs.append({"filename": image.get("filename")})

                if outputs:
                    if node_id not in output_images:
                        output_images[node_id] = outputs
                    else:
                        output_images[node_id].extend(outputs)

        return output_images

    except Exception as e:
        logger.error(f"Error during execution or image retrieval: {str(e)}")
        raise RuntimeError(f"Failed to get images: {str(e)}")


def fill_template(workflow: Any, template_values: Dict[str, Any]) -> Any:
    """Fill template placeholders in a workflow."""
    if isinstance(workflow, dict):
        # Process dictionary
        return {
            key: fill_template(value, template_values)
            for key, value in workflow.items()
        }

    elif isinstance(workflow, list):
        # Process list
        return [fill_template(item, template_values) for item in workflow]

    elif (
        isinstance(workflow, str)
        and workflow.startswith("{{")
        and workflow.endswith("}}")
    ):
        # Process placeholder
        placeholder = workflow[2:-2].strip()
        if placeholder in template_values:
            return template_values[placeholder]
        else:
            return workflow

    else:
        # Leave other values unchanged
        return workflow


def convert_request_file_url_to_path(
    template_values: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Any]]:
    """Convert URLs in template values to local file paths."""
    tempfiles = []
    new_template_values = template_values.copy()

    for key, value in template_values.items():
        # Handle URL strings
        if isinstance(value, str) and (
            value.startswith("https://") or value.startswith("http://")
        ):
            if value.endswith("/"):
                value = value[:-1]

            filename = value.split("/")[-1]
            file_destination_path, file_object = download_tempfile(
                file_url=value, filename=filename
            )

            if file_destination_path and file_object:
                tempfiles.append(file_object)
                new_template_values[key] = file_destination_path

    return new_template_values, tempfiles


def convert_outputs_to_base64(
    node_id: str, file_name: str, file_data: Optional[bytes] = None
) -> Dict[str, str]:
    """Convert output files to base64 representation."""
    try:
        if not file_data:
            # Determine file path
            file_path = os.path.join(COMFYUI_DIR, "output", file_name)
            if not os.path.exists(file_path):
                file_path = file_name  # Try as absolute path

            # Read file data
            with open(file_path, "rb") as f:
                file_data = f.read()

        # Convert to base64
        base64_data = base64.b64encode(file_data).decode("utf-8")

        # Try to determine format
        file_ext = os.path.splitext(file_name)[1].lower().strip(".")
        if not file_ext:
            # Try to detect from image data
            try:
                Image.open(BytesIO(file_data))
                file_ext = "png"
            except:
                file_ext = "bin"

        return {"node_id": node_id, "data": base64_data, "format": file_ext}

    except Exception as e:
        logger.error(f"Error converting output to base64: {str(e)}")
        return {"node_id": node_id, "data": "", "format": "error", "error": str(e)}
