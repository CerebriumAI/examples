import asyncio
import os
import sys
import time
import aiohttp
import requests
from multiprocessing import Process

from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain

from helpers import (
    AudioVolumeTimer,
    TranscriptionTimingLogger,
    LangchainRAGProcessor,
    ElevenLabsTurbo
)

from loguru import logger

from cerebrium import get_secret

os.environ['SSL_CERT'] = ''
os.environ['SSL_KEY'] = ''
os.environ['OPENAI_API_KEY'] = get_secret("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = get_secret("PINECONE_API_KEY")


logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

message_store = {}

embeddings = OpenAIEmbeddings()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]


async def main(room_url: str, token: str):
    
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Andrej Karpathy",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True, ##For local testing, enable and comment out Deepgram sst
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
                # vad_audio_passthrough=True,
            )
        )

        stt = DeepgramSTTService(
            name="STT",
            api_key=None,
            url='ws://127.0.0.1:8082/v1/listen'
        )

        tts = ElevenLabsTurbo(
            aiohttp_session=session,
            api_key=get_secret("ELEVENLABS_API_KEY"), 
            voice_id="uGLvhQYfq0IUmSfqitRE",
        )


        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        vectorstore = PineconeVectorStore.from_existing_index(
            "andrej-youtube", OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 """You are Andrej Karpathy, a Slovak-Canadian computer scientist who served as the director of artificial intelligence and Autopilot Vision at Tesla. \
                 You co-founded and formerly worked at OpenAI, where you specialized in deep learning and computer vision. You publish Youtube videos in which you explain complex \
                 machine learning concepts. Your job is to help people with the content in your Youtube videos given context . Keep your responses concise and relatively simple. \
                Ask for clarification if a user question is ambiguous. Be nice and helpful. Ensure responses contain only words. Check again that you have not included special characters other than '?' or '!'. \
                
                {context}"""
                 ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
        question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
        rag_chain =  create_retrieval_chain(retriever, question_answer_chain)
        
        # chain = prompt | llm
        history_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input",
            output_messages_key="answer")
        lc = LangchainRAGProcessor(chain=history_chain)


        avt = AudioVolumeTimer()
        tl = TranscriptionTimingLogger(avt)

        tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()

        pipeline = Pipeline([
            transport.input(),   # Transport user input
            avt,  # Audio volume timer
            stt,  # Speech-to-text
            tl,  # Transcription timing logger
            tma_in,              # User responses
            lc,                 # LLM
            tts,                 # TTS
            transport.output(),  # Transport bot output
            tma_out,             # Assistant spoken responses
        ])

        task = PipelineTask(pipeline, PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            report_only_initial_ttfb=True,
        ))


        # When the first participant joins, the bot should introduce itself.
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            print(participant["id"])
            transport.capture_participant_transcription(participant["id"])
            lc.set_participant_id(participant["id"])

        # # When the participant leaves, we exit the bot.
        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        # If the call is ended make sure we quit as well.
        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            if state == "left":
                await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)
        await session.close()
    return True

async def start_bot(room_url: str, token: str = None):
    await check_deepgram_model_status()

    try:
        await main(room_url, token)
    except Exception as e:
        logger.error(f"Exception in main: {e}")
        sys.exit(1)  # Exit with a non-zero status code
    
    return {"message": "session finished"}

def create_room():
    url = "https://api.daily.co/v1/rooms/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_secret('DAILY_TOKEN')}"
    }
    data = {
        "properties": {
            "exp": int(time.time()) + 60*5, ##5 mins
            "eject_at_room_exp" : True
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        room_info = response.json()
        token = create_token(room_info['name'])
        if token and 'token' in token:
            room_info['token'] = token['token']
        else:
            print("Failed to create token")
            return {"message": 'There was an error creating your room', "status_code": 500}
        return room_info
    else:
        data = response.json()
        if data.get("error") == "invalid-request-error" and "rooms reached" in data.get("info", ""):
            print("We are currently at capacity for this demo. Please try again later.")
            return {"message": "We are currently at capacity for this demo. Please try again later.", "status_code": 429}
        print(f"Failed to create room: {response.status_code}")
        return {"message": 'There was an error creating your room', "status_code": 500}

def create_token(room_name: str):
    url = "https://api.daily.co/v1/meeting-tokens"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_secret('DAILY_TOKEN')}"
    }
    data = {
        "properties": {
            "room_name": room_name,
            "is_owner": True,
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        token_info = response.json()
        return token_info
    else:
        print(f"Failed to create token: {response.status_code}")
        return None

async def check_deepgram_model_status():
    url = "http://127.0.0.1:8082/v1/status/engine"
    headers = {
        "Content-Type": "application/json"
    }
    max_retries = 5
    async with aiohttp.ClientSession() as session:
        for _ in range(max_retries):
            print("Trying Deepgram local server")
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        print(json_response)
                        if json_response.get('engine_connection_status') == 'Connected':
                            print("Connected to deepgram local server")
                            return True
            except aiohttp.ClientConnectionError:
                print("Connection refused, retrying...")
            await asyncio.sleep(10)
    return False