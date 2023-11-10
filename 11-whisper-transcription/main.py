from typing import Optional

from huggingface_hub import hf_hub_download
from pydantic import BaseModel, HttpUrl
from whisper import load_model, transcribe

from util import download_file_from_url, save_base64_string_to_file


class Item(BaseModel):
    audio: Optional[str] = None
    file_url: Optional[HttpUrl] = None
    webhook_endpoint: Optional[HttpUrl] = None


distil_large_v2 = hf_hub_download(
    repo_id="distil-whisper/distil-large-v2", filename="original-model.bin"
)
model = load_model(distil_large_v2)


def predict(item, run_id, logger):
    item = Item(**item)
    input_filename = f"{run_id}.mp3"

    if item.audio is not None:
        file = save_base64_string_to_file(logger, item.audio)
    elif item.file_url is not None:
        file = download_file_from_url(logger, item.file_url, input_filename)
    else:
        return {"statusCode": 500, "message": "Either audio or file_url must be provided"}
    
    logger.info("Transcribing file...")

    result = transcribe(model, audio=file)

    return result
