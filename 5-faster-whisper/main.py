from faster_whisper import WhisperModel
from pydantic import BaseModel

DOWNLOAD_ROOT = "/tmp"

model_size = "small"
# model_size = "large-v2"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")


# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")


class Item(BaseModel):
    file_url: str


# Downloads a file from a given URL and saves it to a given filename
def download_file_from_url(url: str, filename: str):
    print("Downloading file...")

    import requests

    response = requests.get(url)
    if response.status_code == 200:
        print("Download was successful")

        with open(f"{DOWNLOAD_ROOT}/{filename}", "wb") as f:
            f.write(response.content)

        return f"{DOWNLOAD_ROOT}/{filename}"

    else:
        print(response)
        raise Exception("Download failed")


def predict(item, run_id, logger):
    item = Item(**item)
    file = download_file_from_url(item.file_url, run_id)

    segments, info = model.transcribe(file, beam_size=5)

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    full_text = ""
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        full_text += " " + segment.text

    return full_text.strip()
