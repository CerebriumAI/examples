# import transformers
# import numpy as np
# import librosa
from daily import *
import time
import requests
import os
from dotenv import load_dotenv
import asyncio
import wave
from model import UltravoxModel

load_dotenv()


# model = UltravoxModel()

async def run(room_url):  # run_id is optional, injected by Cerebrium at runtime
    from translator import Translator

    Daily.init()

    translation_client = Translator(room_url, '')#model)
    client = translation_client.client

    client.set_user_name("translator")
    ##only join if not in call already
    translation_client.join(room_url)
    for participant in client.participants():
        if participant != "local":
            client.set_audio_renderer(
                participant, callback=translation_client.on_audio_frame
            )
    try:
        while translation_client.isRunning():
            print("sleeping")
            time.sleep(10)
    except:
        print("\nIssue detected")

    client.leave()
    return {"message": "Call has finished running"}
    
    # wav = wave.open("./actual_speech.wav", 'rb')
    # sample_rate = wav.getframerate()
    # channels = wav.getnchannels()

    # # Create microphone device
    # mic_device = Daily.create_microphone_device(
    #     "model-mic",
    #     sample_rate=sample_rate,
    #     channels=channels,
    #     non_blocking=True
    # )
    
    # # Create client and configure subscriptions
    # client = CallClient()
    # client.update_subscription_profiles(
    #     {"base": {"camera": "unsubscribed", "microphone": "unsubscribed"}}
    # )
    
    # for participant in client.participants():
    #     if participant != "local":
    #         client.set_video_renderer(
    #             participant, callback=object_detector.on_video_frame
    #         )
    # # Join the meeting
    # client.join(
    #     room_url,
    #     client_settings={
    #         "inputs": {
    #             "camera": False,
    #             "microphone": {"isEnabled": True, "settings": {"deviceId": "model-mic"}},
    #         }
    #     }
    # )
    # while True:
    #     await asyncio.sleep(1)
    # Process audio with the model
    # turns = [
    #     {
    #         "role": "system",
    #         "content": "You are a friendly and helpful character. You love to answer questions for people."
    #     },
    # ]
    # # result = pipe({'audio': audio, 'turns': turns, 'sampling_rate': sr}, max_new_tokens=30)
    
    # # Send the generated audio to the meeting
    # # Note: You'll need to convert the model output to the correct audio format
    # # await mic_device.write_frames(result.audio)
    
    # CHUNK_SIZE = 480  # 10ms chunks at 48kHz
    # chunk = wav.readframes(CHUNK_SIZE)
    
    # await asyncio.sleep(10)  # Sleep for 5 minutes (300 seconds)

    # while chunk:
    #     mic_device.write_frames(chunk)
    #     await asyncio.sleep(0.01)  # Sleep for 10ms between chunks
    #     chunk = wav.readframes(CHUNK_SIZE)
    
    # await asyncio.sleep(10)  # Sleep for 5 minutes (300 seconds)
    # # Cleanup
    # # wav.close()
    # await client.leave()
    # client.release()
    

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


if __name__ == "__main__":
    room = create_room()
    print(room)
    if room and "url" in room:
        asyncio.run(run(room["url"]))
