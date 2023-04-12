import pytube 
from pydantic import BaseModel
from typing import Optional
from models.common.decorators import worker
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain import OpenAI
import requests
import faiss


class Item(BaseModel):
    url: str
    question: str

def store_segments(segments):
  texts = []
  start_times = []

  for segment in segments:
    text = segment['text']
    start = segment['start']

    # Convert the starting time to a datetime object
    start_datetime = datetime.fromtimestamp(start)

    # Format the starting time as a string in the format "00:00:00"
    formatted_start_time = start_datetime.strftime('%H:%M:%S')

    texts.append("".join(text))
    start_times.append(formatted_start_time)

  return texts, start_times

@worker
def predict(item: Item):

    video = pytube.YouTube(url)
    video.streams.get_highest_resolution().filesize
    audio = video.streams.get_audio_only()
    fn = audio.download(output_path="/content/clips/")

    model = whisper.load_model("small")
    transcription = model.transcribe('/content/tmp.mp3')
    res = transcription['segments']

    texts, start_times = store_segments(res)