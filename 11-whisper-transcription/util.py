import base64
import uuid

DOWNLOAD_ROOT = "/tmp/"  # Change this to /persistent-storage/ if you want to save files to the persistent storage


# Downloads a file from a given URL and saves it to a given filename
def download_file_from_url(logger, url: str, filename: str):
    logger.info("Downloading file...")

    import requests

    response = requests.get(url)
    if response.status_code == 200:
        logger.info("Download was successful")

        with open(filename, "wb") as f:
            f.write(response.content)

        return filename

    else:
        logger.info(response)
        raise Exception("Download failed")


# Saves a base64 encoded file string to a local file
def save_base64_string_to_file(logger, audio: str):
    logger.info("Converting file...")

    decoded_data = base64.b64decode(audio)

    filename = f"{DOWNLOAD_ROOT}/{uuid.uuid4()}"

    with open(filename, "wb") as file:
        file.write(decoded_data)

    logger.info("Decoding base64 to file was successful")
    return filename
