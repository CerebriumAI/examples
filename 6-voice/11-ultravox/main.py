import os
import time
from loguru import logger
import requests
from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.services.ultravox.stt import UltravoxSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from fastapi import FastAPI

load_dotenv()

# Initialize FastAPI
app = FastAPI()

ultravox_processor = UltravoxSTTService(
    model_name="fixie-ai/ultravox-v0_5-llama-3_1-8b",
    hf_token=os.getenv("HF_TOKEN"),
)


@app.post("/run")
async def run(room_url: str, token: str):
    transport = DailyTransport(
        room_url,
        token,
        "Respond bot",
        DailyParams(
            audio_out_enabled=True,
            transcription_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            vad_audio_passthrough=True,
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.environ.get("CARTESIA_API_KEY"),
        voice_id='97f4b8fb-f2fe-444b-bb9a-c109783a857a',

    )

    # Create pipeline using transport.input() and transport.output()
    pipeline = Pipeline([transport.input(), ultravox_processor, tts, transport.output()])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )
    runner = PipelineRunner()

    logger.info("Starting pipeline...")
    await runner.run(task)


@app.post("/create-room")
def create_room():
    url = "https://api.daily.co/v1/rooms/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('DAILY_TOKEN')}",
    }
    data = {
        "properties": {
            "exp": int(time.time()) + 60 * 5,  ##5 mins
            "eject_at_room_exp": True,
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        room_info = response.json()
        token = create_token(room_info["name"])
        if token and "token" in token:
            room_info["token"] = token["token"]
        else:
            print("Failed to create token")
            return {
                "message": "There was an error creating your room",
                "status_code": 500,
            }
        return room_info
    else:
        data = response.json()
        if data.get("error") == "invalid-request-error" and "rooms reached" in data.get(
                "info", ""
        ):
            print("We are currently at capacity for this demo. Please try again later.")
            return {
                "message": "We are currently at capacity for this demo. Please try again later.",
                "status_code": 429,
            }
        print(f"Failed to create room: {response.status_code}")
        return {"message": "There was an error creating your room", "status_code": 500}


def create_token(room_name: str):
    url = "https://api.daily.co/v1/meeting-tokens"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('DAILY_TOKEN')}",
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


@app.get("/health")
async def health():
    return {"status": "ok"}
