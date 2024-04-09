
from typing import Optional
from pydantic import BaseModel

import copy
import json
import os
import time
import uuid
from multiprocessing import Process
from typing import Dict

import websocket
from helpers import (
    convert_outputs_to_base64,
    convert_request_file_url_to_path,
    fill_template,
    get_images,
    setup_comfyui,
)

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())
original_working_directory = os.getcwd()
global json_workflow
json_workflow = None

global side_process
side_process = None
if side_process is None:
    side_process = Process(
        target=setup_comfyui,
        kwargs=dict(
            original_working_directory=original_working_directory,
            data_dir="",
        ),
    )
    side_process.start()

# Load the workflow file as a python dictionary
with open(
    os.path.join("./", "workflow_api.json"), "r"
) as json_file:
    json_workflow = json.load(json_file)

# Connect to the ComfyUI server via websockets
socket_connected = False
while not socket_connected:
    try:
        ws = websocket.WebSocket()
        ws.connect(
            "ws://{}/ws?clientId={}".format(server_address, client_id)
        )
        socket_connected = True
    except Exception as e:
        print("Could not connect to comfyUI server. Trying again...")
        time.sleep(5)

print("Successfully connected to the ComfyUI server!")

class Item(BaseModel):
    workflow_values: Optional[Dict] = None

def predict(item, run_id, logger):
    item = Item(**item)

    template_values = item.workflow_values

    template_values, tempfiles = convert_request_file_url_to_path(template_values)
    json_workflow_copy = copy.deepcopy(json_workflow)
    json_workflow_copy = fill_template(json_workflow_copy, template_values)
    outputs = {}  # Initialize outputs to an empty dictionary

    try:
        outputs = get_images(
            ws, json_workflow_copy, client_id, server_address
        )

    except Exception as e:
        print('did it get here')
        print("Error occurred while running Comfy workflow: ", e)

    for file in tempfiles:
        file.close()

    result = []
    print('here')
    for node_id in outputs:
        for unit in outputs[node_id]:
            file_name = unit.get("filename")
            file_data = unit.get("data")
            output = convert_outputs_to_base64(
                node_id=node_id, file_name=file_name, file_data=file_data
            )
            result.append(output)

    return {"result": result}