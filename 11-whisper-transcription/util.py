import base64
import uuid

import requests

DOWNLOAD_ROOT = "/tmp/"  # Change this to /persistent-storage/ if you want to save files to the persistent storage


def download_file_from_url(url: str, filename: str):
    print("Downloading file...")

    response = requests.get(url)
    if response.status_code == 200:
        print("Download was successful")

        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    else:
        print(response)
        raise Exception("Download failed")


# Saves a base64 encoded file string to a local file
def save_base64_string_to_file(audio: str):
    print("Converting file...")

    decoded_data = base64.b64decode(audio)

    filename = f"{DOWNLOAD_ROOT}/{uuid.uuid4()}"

    with open(filename, "wb") as file:
        file.write(decoded_data)

    print("Decoding base64 to file was successful")
    return filename
