import os
from datetime import datetime

import faiss
import pytube
import whisper
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CerebriumAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = whisper.load_model("small")
sentenceTransformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
os.environ["CEREBRIUMAI_API_KEY"] = "c_api_key-xxx"


class Item(BaseModel):
    url: str
    question: str


def store_segments(segments):
    texts = []
    start_times = []

    for segment in segments:
        text = segment["text"]
        start = segment["start"]

        # Convert the starting time to a datetime object
        start_datetime = datetime.fromtimestamp(start)

        # Format the starting time as a string in the format "00:00:00"
        formatted_start_time = start_datetime.strftime("%H:%M:%S")

        texts.append("".join(text))
        start_times.append(formatted_start_time)

    return texts, start_times


def create_embeddings(texts, start_times):
    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(texts):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": start_times[i]}] * len(splits))
    return metadatas, docs


def predict(item, run_id, logger):
    item = Item(**item)

    video = pytube.YouTube(item.url)
    video.streams.get_highest_resolution().filesize
    audio = video.streams.get_audio_only()
    fn = audio.download(output_path="/models/content/", filename= f"{video.title}.mp4")

    transcription = model.transcribe(f"/models/content/{video.title}.mp4")
    res = transcription["segments"]

    texts, start_times = store_segments(res)

    metadatas, docs = create_embeddings(texts, start_times)
    embeddings = HuggingFaceEmbeddings()
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    faiss.write_index(store.index, "docs.index")
    llm = CerebriumAI(
        endpoint_url="https://run.cerebrium.ai/flan-t5-xl-webhook/predict"
    )
    chain = VectorDBQAWithSourcesChain.from_llm(llm=llm, vectorstore=store)

    result = chain({"question": item.question})

    return {"result": result}
