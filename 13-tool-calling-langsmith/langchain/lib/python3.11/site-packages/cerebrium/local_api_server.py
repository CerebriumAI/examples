"""A local run FastAPI server which receives predict requests as if it was the backend server.


Endpoints:
    - /predict: POST request to get predictions from the model.

Usage:
    - Runs as part of the `cerebrium serve start` command to process requests and send them to the backend server.
    - Otherwise, run this script and send POST requests to the /predict endpoint with the required data.
"""

import os
import json
import traceback
from typing import Any, Dict, Optional, Union
from fastapi import FastAPI, HTTPException, Request
import uvicorn
import threading

SERVE_SESSION_CACHE_FILE = os.path.join(
    os.path.expanduser("~/.cerebrium"), "serve_session.json"
)
fast_app = FastAPI()
shutdown_event = threading.Event()


async def save_predict_data(request: Request, id: Union[str, None]):
    data = await request.body()
    try:
        if data:
            data = json.loads(data)
        else:
            data = {}
    except json.JSONDecodeError:
        return {
            "message": "Could not decode the data in the body of your request. Please ensure that the data is in JSON format."
        }
    except Exception as e:
        return {"message": f"An error occurred while processing your request: {e}"}

    session = await get_session(id=id)

    if session is None or "workingDir" not in session:
        return {
            "message": "No session found. Please start a session using `cerebrium serve start` in your cortex deployment's root directory."
        }
    out_file = os.path.join(session["workingDir"], "_cerebrium_predict.json")
    if os.path.exists(out_file):
        os.remove(out_file)
    with open(out_file, "w") as f:
        f.write(json.dumps(data))
    return {
        "message": "Data sent successfully. You should see the response in your cli logs"
    }


@fast_app.post("/predict")
async def predict(request: Request):
    try:
        return await save_predict_data(request, None)
    except Exception:
        return HTTPException(status_code=500, detail=traceback.format_exc())


@fast_app.get("/")
async def home(request: Request):
    return {
        "message": "Cerebrium local server is running...\nSend POST requests to /predict to upload them to your cerebrium serve instance."
    }


@fast_app.post("/{session_id}/predict")
async def session_predict(request: Request, session_id: str):
    print("Received predict request...")
    try:
        return save_predict_data(request, session_id)
    except Exception:
        return HTTPException(status_code=500, detail=traceback.format_exc())


async def get_session(id: Union[str, None] = None) -> Union[Dict[str, str], None]:
    """
    Retrieve a session based on the provided session ID.

    This function looks up a session by its ID in the serve session cache file. If no ID is provided,
    it attempts to return the most recent session. If the session ID is not found, or if there are no
    sessions available, it raises an HTTPException with appropriate status codes and details.

    Parameters:
    - id (Union[str, None]): The ID of the session to retrieve. If None, the most recent session is retrieved.

    Returns:
    - Union[Dict[str, str], None]: The session data as a dictionary if found, otherwise None.

    Raises:
    - HTTPException: If no session is found with the provided ID, or if there are no sessions available.
    """
    sessions = {}
    if os.path.exists(SERVE_SESSION_CACHE_FILE):
        with open(SERVE_SESSION_CACHE_FILE, "r") as f:
            sessions = json.load(f)

        if id is not None:
            if id in sessions:
                return sessions[id]
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"No session found with id {id}. Please start a session using `cerebrium serve start` in your cortex deployment's root directory.",
                )

        most_recent: Dict[str, Any] = {}
        for _, session in sessions.items():
            if "created" in session:
                if most_recent == {} or session["created"] > most_recent["created"]:
                    most_recent = session
        return most_recent
    return None


def start_local_server(port: int = 8000, reload: bool = False, quiet: bool = True):
    try:
        if not quiet:
            print("Starting server")
        config = uvicorn.Config(
            fast_app,
            host="0.0.0.0",
            port=port,
            log_level="warning" if quiet else "info",
            reload=reload,
        )
        server = uvicorn.Server(config)
        # Start the server in a non-blocking way
        server_thread = threading.Thread(target=lambda: server.run(), daemon=True)
        server_thread.start()
        return server, server_thread  # Return the server instance for further use
    except Exception as e:
        print(f"Error starting local server: {e}")
        raise e


def stop_local_server(server: Optional[uvicorn.Server]):
    shutdown_event.set()  # Signal the server thread to stop
    if server is None:
        print("Server not running.")
        return

    server.should_exit = True


if __name__ == "__main__":
    start_local_server(8001, reload=True, quiet=False)
