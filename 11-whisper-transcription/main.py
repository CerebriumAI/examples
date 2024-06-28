from typing import Optional
from pydantic import BaseModel, HttpUrl
from huggingface_hub import hf_hub_download
from whisper import load_model, transcribe
from util import download_file_from_url, save_base64_string_to_file

distil_large_v2 = hf_hub_download(repo_id="distil-whisper/distil-large-v2", filename="original-model.bin")
model = load_model(distil_large_v2)

class Item(BaseModel):
    audio: Optional[str]
    file_url: Optional[HttpUrl]
    webhook_endpoint: Optional[HttpUrl]


def predict(run_id, audio=None, file_url=None, webhook_endpoint=None):
    item = Item(audio=audio, file_url=file_url, webhook_endpoint=webhook_endpoint)
    input_filename = f"{run_id}.mp3"

    if audio is None and file_url is None:
        raise 'Either audio or file_url must be provided'
    else:
        if item.audio is not None:
            file = save_base64_string_to_file(item.audio)
        elif item.file_url is not None:
            file = download_file_from_url(item.file_url, input_filename)
        print("Transcribing file...")

        result = transcribe(model, audio=file)
        return result
