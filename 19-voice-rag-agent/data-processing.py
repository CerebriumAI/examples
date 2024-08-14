import os
from pytube import YouTube
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
import httpx
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()


def download_videos(link: str, download_path: str):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    yt = YouTube(link)

    # audio_download = yt.streams.get_audio_only()

    print("Downloading Audio...")
    # audio_download.download(filename=f"{yt.title}.mp3", output_path = download_path)
    download_file_path = f"{download_path}/{yt.title}.mp3"

    return (
        yt.title,
        download_file_path,
    )


def transcribe_file(audio_file: str):
    print("transcribing")
    try:
        # STEP 1 Create a Deepgram client using the API key
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

        print(audio_file)
        with open(audio_file, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        # STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        # STEP 3: Call the transcribe_file method with the text payload and options
        response = deepgram.listen.prerecorded.v("1").transcribe_file(
            payload, options, timeout=httpx.Timeout(300.0, connect=10.0)
        )
        print(response)
        return response.results.channels[0].alternatives[0].transcript

    except Exception as e:
        print(f"Exception: {e}")


def embed_text(text: str):
    print("Embedding")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_text = text_splitter.split_text(text)
    return split_text


def save_embeddings_to_db(title: str, url: str, docs: [str]):
    index_name = "andrej-youtube"
    embeddings = OpenAIEmbeddings()
    # Connect to Pinecone index and insert the chunked docs as contents
    PineconeVectorStore.from_texts(docs, embeddings, index_name=index_name)


if __name__ == "__main__":
    video_links = [
        "https://www.youtube.com/watch?v=l8pRSuU81PU",
        "https://www.youtube.com/watch?v=zduSFxRajkE",
        "https://www.youtube.com/watch?v=zjkBMFhNj_g",
        "https://www.youtube.com/watch?v=kCc8FmEb1nY",
    ]

    for link in video_links:
        title, download_path = download_videos(link, "./videos")
        texts = transcribe_file(download_path)
        docs = embed_text(texts)
        save_embeddings_to_db(title, link, docs=docs)
