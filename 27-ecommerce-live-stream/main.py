from daily import *
import time
from cerebrium import get_secret
import requests
from multiprocessing import Process


def predict(room_url: str):
    from detection import ObjectDetection

    bot_name = "Item Detector"

    Daily.init()

    object_detector = ObjectDetection(room_url)
    client = object_detector.client

    client.set_user_name(bot_name)
    ##only join if not in call already
    object_detector.join(room_url)
    for participant in client.participants():
        if participant != "local":
            client.set_video_renderer(
                participant, callback=object_detector.on_video_frame
            )
    try:
        while object_detector.isRunning():
            print("sleeping")
            time.sleep(10)
    except:
        print("\nIssue detected")

    client.leave()
    return {"message": "Call has finished running"}


def start(room_url: str):
    process = Process(target=predict, args=(room_url), daemon=True)
    process.start()
    process.join()  # Wait for the process to complete
    return {"message": "Call finished"}


def create_room():
    url = "https://api.daily.co/v1/rooms/"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_secret('DAILY_TOKEN')}",
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
        "Authorization": f"Bearer {get_secret('DAILY_TOKEN')}",
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
