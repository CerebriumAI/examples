import io
from typing import Optional
import base64

from pydantic import BaseModel

from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp

import jax

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")

pipeline = FlaxWhisperPipline("openai/whisper-large-v2")


class Item(BaseModel):
    audio: Optional[str]
    file_url: Optional[str]


# Saves a base64 encoded file string to a local file
def save_base64_string_to_file(logger, audio: str, uuid: str):
    logger.info("Converting file...")

    decoded_data = base64.b64decode(audio)

    filename = f"./{uuid}"

    with open(filename, "wb") as file:
        file.write(decoded_data)

    logger.info("Decoding base64 to file was successful")
    return filename

# Downloads a file from a given URL and saves it to a given filename
def download_file_from_url(logger, url: str, uuid: str):
    logger.info("Downloading file...")

    import requests

    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Download was successful")

        filename = f"/models/{uuid}.mp3"
        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    else:
        logger.info(response)
        raise Exception("Download failed")


def predict(item, run_id, logger):
    item = Item(**item)

    if not item.audio and not item.file_url:
        return "audio or file_url field is required."
    
    if item.audio:
        file = save_base64_string_to_file(logger, item.audio, run_id)
    elif item.file_url:
        file = download_file_from_url(logger, item.file_url, run_id)

    outputs = pipeline(file,  task=item.task, return_timestamps=True)

    return {"results": outputs}

