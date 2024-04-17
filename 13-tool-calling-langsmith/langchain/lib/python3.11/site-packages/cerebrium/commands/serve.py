import json
import os
import sys
import threading
import time
from typing import Any, Dict, List

import requests
import typer
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from typing_extensions import Annotated, Optional
from yaspin import yaspin  # type: ignore
from yaspin.core import Yaspin  # type: ignore

from cerebrium import __version__ as cerebrium_version
from cerebrium import api, constants, datatypes, local_api_server, utils
from cerebrium.commands.cortex import was_provided
from cerebrium.datatypes import SERVE_SESSION_CACHE_FILE, CerebriumConfig, HttpMethod
from cerebrium.utils.display import dict_pretty_print
from cerebrium.utils.files import FileHashesType
from cerebrium.utils.logging import cerebrium_log, console
from cerebrium.utils.misc import get_function_params
from threading import Thread
from cerebrium.utils.watchdog import monitor_directory_changes
from watchdog.events import FileCreatedEvent

serve_app = typer.Typer(no_args_is_help=True)

# This is used to stop the logs on a different thread
start_event = threading.Event()
stop_event = threading.Event()
file_monitor_stop_event = threading.Event()


@serve_app.command()
def start(
    timeout: Annotated[
        Optional[int],
        typer.Option(
            min=1,
            max=60,
            help="Timeout for serve session in minutes. Defaults to 30 mins",
        ),
    ] = 30,
    port: Annotated[
        int,
        typer.Option(
            help="Port to run the local REST server on for cerebrium serve. Defaults to 7900"
        ),
    ] = 7900,
    gpu: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Hardware to use for the Cortex deployment. "
                "Defaults to 'AMPERE_A6000'. "
                "Can be one of "
                "'TURING_4000', "
                "'TURING_5000', "
                "'AMPERE_A4000', "
                "'AMPERE_A5000', "
                "'AMPERE_A6000', "
                "'AMPERE_A100'"
            ),
        ),
    ] = None,
    cpu: Annotated[
        Optional[int],
        typer.Option(
            min=constants.MIN_CPU,
            max=constants.MAX_CPU,
            help=(
                "Number of vCPUs to use for the Cortex deployment. Defaults to 2. "
                "Can be an integer between 1 and 48."
            ),
        ),
    ] = None,
    memory: Annotated[
        Optional[float],
        typer.Option(
            min=constants.MIN_MEMORY,
            max=constants.MAX_MEMORY,
            help=(
                "Amount of memory(GB) to use for the Cortex deployment. Defaults to 16. "
                "Can be a float between 2.0 and 256.0 depending on hardware selection."
            ),
        ),
    ] = None,
    gpu_count: Annotated[
        Optional[int],
        typer.Option(
            min=1,
            max=constants.MAX_GPU_COUNT,
            help=(
                "Number of GPUs to use for the Cortex deployment. Defaults to 1. "
                "Can be an integer between 1 and 8."
            ),
        ),
    ] = None,
    predict_data: Annotated[
        Optional[str],
        typer.Option(
            "--predict-data",
            "--data",
            help="JSON string containing all the parameters that will be used to run your "
            "deployment's predict function on build to ensure your new deployment will work "
            "as expected before replacing your existing deployment.",
        ),
    ] = None,
    python_version: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Python version to use. "
                "Currently, we support '3.8' to '3.11'. Defaults to '3.10'"
            ),
        ),
    ] = None,
    include: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Comma delimited string list of relative paths to files/folder to include. "
                "Defaults to all visible files/folders in project root."
            ),
        ),
    ] = None,
    exclude: Annotated[
        Optional[str],
        typer.Option(
            help="Comma delimited string list of relative paths to files/folder to exclude. "
            "Defaults to all hidden files/folders in project root.",
        ),
    ] = None,
    force_rebuild: Annotated[
        Optional[bool],
        typer.Option(
            help="Force rebuild. Clears rebuilds deployment from scratch as if it's a clean deployment.",
        ),
    ] = None,
    init_debug: Annotated[
        Optional[bool],
        typer.Option(
            help="Stops the container after initialization.",
        ),
    ] = None,
    log_level: Annotated[
        Optional[str],
        typer.Option(
            help="Log level for the Cortex deployment. Can be one of 'DEBUG' or 'INFO'",
        ),
    ] = None,
    config_file: Annotated[
        Optional[str],
        typer.Option(
            help="Path to cerebrium.toml file. You can generate a config using `cerebrium init-cortex`. The contents of the deployment config file are overridden by the command line arguments.",
        ),
    ] = None,
    disable_predict: Annotated[
        Optional[bool], typer.Option(help="Flag to disable running predict function.")
    ] = None,
    cuda_version: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "CUDA version to use. "
                "Currently, we support 11.8 as '11' and 12.2 as '12'. Defaults to '12'"
            ),
        ),
    ] = None,
):
    """Start a new cerebrium serve session"""
    # Load in the config
    timeout_seconds = timeout * 60 if timeout is not None else 30 * 60

    if config_file is None or not config_file.strip():
        config_file = "cerebrium.toml"
    # func_defaults contains all parameters with their default values
    func_defaults = get_function_params(start)

    # provided_params only include explicitly provided parameters
    provided_params = {
        param: value
        for param, value in locals().items()
        if param in func_defaults and was_provided(param, value, func_defaults)
    }

    # Merge func_defaults with provided_params. Provided values override default values where applicable.
    params = {**func_defaults, **provided_params}
    params["disable_predict"] = True

    # load config toml file and merge with param values
    config_obj = utils.misc.merge_config_with_params("cerebrium.toml", params)

    ##validation is done with types in classes
    config = datatypes.CerebriumConfig(
        scaling=datatypes.CerebriumScaling(**config_obj["scaling"]),
        build=datatypes.CerebriumBuild(**config_obj["build"]),
        deployment=datatypes.CerebriumDeployment(**config_obj["deployment"]),
        hardware=datatypes.CerebriumHardware(**config_obj["hardware"]),
        dependencies=datatypes.CerebriumDependencies(**config_obj["dependencies"]),
        cerebrium_version=cerebrium_version,
    )
    if not os.path.exists("cerebrium.toml"):
        utils.tomls.save_config_to_toml_file(config, "cerebrium.toml")

    backend_params = utils.misc.flatten_cerebrium_config_to_json(config=config)
    backend_params = utils.remove_null_values(backend_params)
    backend_params["function"] = "serve"
    backend_params["source"] = "serve"

    cerebrium_log(
        message="🏗️  Starting served session...",
        level="INFO",
        color="yellow",
    )

    response = api.cerebrium_request(
        http_method=HttpMethod.POST, url="setupApp", payload=backend_params
    )

    if response is None:
        cerebrium_log(
            message="Error starting serve session. Please check your internet connection and try again.\nIf the problem persists, please contact support.",
            level="ERROR",
        )
        exit()

    elif response.status_code != 200:
        cerebrium_log(
            message=f"Error starting serve session:\nStatus code: {response.status_code}\n  {response.json().get('message', response.text)}",
            level="ERROR",
        )
        exit()
    elif "serveId" not in response.json():
        cerebrium_log(
            message="Error starting serve session. Missing information in response.\n"
            "Please check you're using the latest version of the CLI and try again.\n"
            "If the problem persists, please contact support.",
            level="ERROR",
        )
        exit()

    setup_response = response.json()
    # check serveId, uploadUrl, keyName, status are all in the response
    if (
        "serveId" not in setup_response
        or "uploadUrl" not in setup_response
        or "keyName" not in setup_response
        or "status" not in setup_response
    ):
        cerebrium_log(
            message="Error starting serve session. Missing information in response",
            level="ERROR",
        )
        return

    serve_id = setup_response["serveId"]
    print(f"🆔 Serve ID: {serve_id}")
    _status = setup_response["status"]

    _save_serve_session(setup_response, config, timeout_seconds)

    ##if files change, reload
    file_hashes_dict: Dict[str, Dict[str, str]] = {"hashes": {}}
    ##upload files initially - thereafter triggered by directory changes
    update_file_hashes(serve_id, file_hashes_dict, config, False)
    file_monitor_thread = Thread(
        target=monitor_directory_changes,
        args=(
            "./",
            lambda event: (
                update_file_hashes(serve_id, file_hashes_dict, config, False)
                if not (
                    isinstance(event, FileCreatedEvent)
                    and event.src_path.endswith("_cerebrium_predict.json")
                )  ##don't want to show reload with request
                else update_file_hashes(serve_id, file_hashes_dict, config, True)
            ),
            file_monitor_stop_event,
        ),
    )
    file_monitor_thread.start()

    cerebrium_log(
        message=f"Session {serve_id} started successfully",
        level="INFO",
        color="green",
    )

    ##Started API server on new thread
    server, api_server_thread = local_api_server.start_local_server(port)

    timeout = 5
    successful_start = False
    while timeout > 0:
        try:
            response = requests.get(f"http://localhost:{port}", timeout=1)
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            timeout -= 1
            continue
        if response.status_code == 200:
            successful_start = True
            break
    if api_server_thread.is_alive() and successful_start:
        print()
        cerebrium_log(
            f"Local API server started successfully on port {port}\n"
            f"Send a POST request to: http://localhost:{port}/predict\n",
            color="green",
            prefix_seperator="\t",
        )

    else:
        cerebrium_log(f"Failed to start local API server on port {port}", level="ERROR")

    spinner = Spinner("dots", "Building App...", style="gray")

    if constants.env == "local":
        log_thread = threading.Thread(
            target=api.poll_build_logs,
            args=(serve_id, start_event, stop_event, "serve"),
        )
    else:
        modelId = f"{setup_response['projectId']}-{config.deployment.name}"
        log_thread = threading.Thread(
            target=api.stream_logs,
            args=(start_event, stop_event, modelId, serve_id.split(f"{modelId}-")[1]),
        )

    log_thread.start()
    live = Live(spinner, console=console, refresh_per_second=10)

    # Start the Live context using the start() method
    live.start()

    try:
        while True:
            build_status = api.get_build_status(serve_id)
            spinner.text = api.log_build_status(build_status, time.time(), mode="serve")
            if (
                build_status in ["build_failure", "init_failure"]
                or file_monitor_stop_event.is_set()
            ):
                start_event.wait()
                live.update(Text(""))
                live.stop()
                __graceful_shutdown(
                    server,
                    api_server_thread,
                    file_monitor_thread,
                    log_thread,
                    serve_id=serve_id,
                    is_interrupt=True,
                )
                break
            time.sleep(5)
    except KeyboardInterrupt:
        # If user presses Ctrl-C, signal all threads to stop
        live.stop()
        __graceful_shutdown(
            server,
            api_server_thread,
            file_monitor_thread,
            log_thread,
            serve_id=serve_id,
            is_interrupt=True,
        )
        cerebrium_log(
            "User interrupted. Closing all threads...", level="INFO", color="yellow"
        )
    except Exception as e:
        # If any other exception occurs, ensure graceful shutdown
        live.stop()
        __graceful_shutdown(
            server,
            api_server_thread,
            file_monitor_thread,
            log_thread,
            serve_id=serve_id,
            is_interrupt=True,
        )
        cerebrium_log(
            f"An error occurred: {str(e)}. Closing all threads...",
            level="ERROR",
            color="red",
        )
    finally:
        cerebrium_log(
            "Serve ended successfully",
            level="INFO",
            color="green",
            prefix_seperator="\t",
        )
        sys.exit(1)


@serve_app.command()
def end_session(
    id: Annotated[
        str,
        typer.Argument(
            ...,
            help="The id of the serve session to end",
        ),
    ],
):
    """End a running cerebrium serve session"""

    cerebrium_log(
        "Ending serve session...", level="INFO", color="yellow", prefix_seperator="\t"
    )

    try:
        response = api.cerebrium_request(
            http_method=HttpMethod.DELETE,
            url="end-serve",
            payload={"serveId": id},
        )
        if response is None:
            cerebrium_log(
                message="Error ending session. Please check your internet connection and try again.\nIf the problem persists, please contact support.",
                level="ERROR",
                exit_on_error=False,
            )
            return
        if response.status_code != 200:
            try:
                json_response = response.json()
                if "message" in json_response:
                    cerebrium_log(
                        f"Error ending session: {response.json()['message']}",
                        level="ERROR",
                        exit_on_error=False,
                    )
                return
            except json.JSONDecodeError:
                cerebrium_log(
                    f"Error ending session{ ':' +response.text if response.text else ''}",
                    level="ERROR",
                    exit_on_error=False,
                )
                return

    except requests.exceptions.RequestException as e:
        cerebrium_log(f"Error ending session: {e}", level="ERROR", exit_on_error=False)


@serve_app.command()
def predict(
    data: Annotated[
        str,
        typer.Argument(
            ..., help="The data to predict on - in the form of a JSON string"
        ),
    ],
    id: Annotated[str, typer.Option(help="The id of the serve session")] = "",
    disable_wait: Annotated[
        bool, typer.Option(help="Disable waiting for the session to start")
    ] = False,
):
    if os.path.exists(SERVE_SESSION_CACHE_FILE):
        with open(SERVE_SESSION_CACHE_FILE, "r") as f:
            serve_sessions = json.load(f)
    else:
        serve_sessions: Dict[Any, Any] = {}

    if len(serve_sessions) == 0 and not id:
        cerebrium_log(
            message="You have no running serve sessions. Please start a session using the `cerebrium serve start` command or enter a session id to run inference with.",
            level="ERROR",
        )
        sys.exit(1)

    if not id or id == "":
        # get the latest session from the cache
        session_id = max(serve_sessions, key=lambda k: serve_sessions[k]["created"])

    elif len(id) > 50:
        cerebrium_log(
            message="Invalid session id!\nYou can get your session ID using the `cerebrium serve list` command",
            level="ERROR",
        )
        sys.exit(1)
    else:
        session_id = id

    print(f"Using session ID: {session_id}")

    spinner = yaspin(text="Checking session has started", color="yellow")
    session_status = api.get_build_status(build_id=session_id, mode="serve")
    while session_status.lower() not in ["serving", "success"] and not disable_wait:
        if spinner._start_time is None:  # type: ignore
            spinner.start()
        spinner.text = "Waiting for serve session to start..."
        time.sleep(1)
        session_status = api.get_build_status(build_id=session_id, mode="serve")

    if spinner._start_time is not None:  # type: ignore
        spinner.ok("✅")
        spinner.text = "Serve session is active. Sending predict data..."
        spinner.stop()

    # Save the predict data as a file
    try:
        if len(data) < 250:
            print("Running your predict with:", json.dumps(json.loads(data), indent=2))
        else:
            # truncate the data to fit on a terminal nicely
            usr_data = json.loads(data)
            print("Running your predict with:\n")
            dict_pretty_print(ugly=usr_data)

    except json.JSONDecodeError:
        cerebrium_log(
            message="The predict data you've entered is invalid JSON. Please provide valid JSON data",
            level="ERROR",
        )
        sys.exit(1)

    predict_file: str = "_cerebrium_predict.json"

    with open(predict_file, "w") as f:
        f.write(data)

    with open(SERVE_SESSION_CACHE_FILE, "r") as f:
        serve_session = json.load(f)
        upload_url = serve_session[session_id]["uploadUrl"]
        s3_key_name = serve_session[session_id]["keyName"]

    config = datatypes.CerebriumConfig(
        scaling=datatypes.CerebriumScaling(),
        build=datatypes.CerebriumBuild(disable_animation=True, predict_data=data),
        deployment=datatypes.CerebriumDeployment(),
        hardware=datatypes.CerebriumHardware(),
        dependencies=datatypes.CerebriumDependencies(),
        cerebrium_version=cerebrium_version,
    )
    config.file_list = [predict_file]

    api.upload_cortex_files(
        upload_url=upload_url,
        zip_file_name=os.path.basename(s3_key_name),
        config=config,
        quiet=True,
        source="serve",
    )

    if os.path.exists(predict_file):
        os.remove(predict_file)


def update_file_hashes(
    serve_id: str,
    file_hashes_dict: Dict[str, Any],
    config: CerebriumConfig,
    quiet: bool,
):
    # Call _sync_local_serve_files and update the hashes in the shared dictionary
    new_hashes = _sync_local_serve_files(
        id=serve_id, hashes=file_hashes_dict["hashes"], config=config, quiet=quiet
    )
    file_hashes_dict["hashes"] = new_hashes


def _sync_local_serve_files(
    id: str, hashes: FileHashesType, config: CerebriumConfig, quiet: bool = False
) -> FileHashesType:
    """Check if there has been a change in the files in the current directory and sync them with the backend"""
    with open(SERVE_SESSION_CACHE_FILE, "r") as f:
        serve_session = json.load(f)
        upload_url = serve_session[id]["uploadUrl"]
        s3_key_name = serve_session[id]["keyName"]

    new_hashes = utils.files.get_all_file_hashes()

    if new_hashes == hashes:
        return new_hashes

    changed_files: List[str] = [
        file_name
        for file_name in new_hashes
        if file_name not in hashes or new_hashes[file_name] != hashes[file_name]
    ]

    config.file_list = changed_files
    if len(changed_files) > 0:
        plural = len(changed_files) > 1
        if not quiet:
            print(
                f"🏗️ Found {len(changed_files)} file{'s' if plural else ''} that {'have' if plural else 'has'} changed"
            )
            for file in changed_files:
                print(f"🔄 Syncing {file}")
        api.upload_cortex_files(
            upload_url=upload_url,
            zip_file_name=os.path.basename(s3_key_name),
            config=config,
            quiet=quiet,
            source="serve",
        )

    # hash again in  case any files that changed were temp files (_cerebrium_predict.json, etc)
    return utils.files.get_all_file_hashes()


def __graceful_shutdown(
    server: Any,
    api_server_thread: threading.Thread,
    file_monitor_thread: threading.Thread,
    log_thread: threading.Thread,
    serve_id: str,
    is_interrupt: bool = False,
):
    """
    This function is called when the user presses Ctrl+C while streaming logs.
    - stops the spinner
    - sends a kill signal to the backend to stop the serve session
    - prints a message
    - exits the program
    """
    current_thread = (
        threading.current_thread()
    )  ##needed since timeout is done from another thread

    if is_interrupt:
        cerebrium_log(
            "\n\nCtrl+C detected. Shutting down gracefully...", color="yellow"
        )

    end_session(id=serve_id)

    cerebrium_log(
        "Stopping local API server...",
        level="INFO",
        color="yellow",
        prefix_seperator="\t",
    )
    if api_server_thread and api_server_thread.is_alive():
        local_api_server.stop_local_server(server)
        api_server_thread.join()
    if log_thread and log_thread.is_alive():
        stop_event.set()
        log_thread.join()

    if (
        file_monitor_thread
        and file_monitor_thread is not current_thread
        and file_monitor_thread.is_alive()
    ):
        file_monitor_stop_event.set()
        file_monitor_thread.join()

    with open(SERVE_SESSION_CACHE_FILE, "r") as f:
        serve_sessions = json.load(f)
        if serve_id in serve_sessions:
            serve_sessions.pop(serve_id)

    with open(SERVE_SESSION_CACHE_FILE, "w") as f:
        json.dump(serve_sessions, f, indent=2)


def _save_serve_session(
    setup_response: Dict[str, Any],
    config: CerebriumConfig,
    timeout_seconds: int,
):
    """Save the serve session to a file in the ~/.cerebrium directory"""

    serve_session = {
        setup_response["serveId"]: {
            "workingDir": os.getcwd(),
            "status": setup_response["status"],
            "projectId": setup_response["projectId"],
            "uploadUrl": setup_response["uploadUrl"],
            "keyName": setup_response["keyName"],
            "created": time.time(),
            "timeout": timeout_seconds,
            "expires": time.time() + timeout_seconds,
            "config": config.to_dict(),
        }
    }

    # Save the serve session to a file

    if os.path.exists(SERVE_SESSION_CACHE_FILE):
        with open(SERVE_SESSION_CACHE_FILE, "r") as f:
            existing_serve_session = json.load(f)
            existing_serve_session.update(serve_session)
            serve_session = existing_serve_session

    # Remove expired sessions
    for session_id in list(serve_session.keys()):
        try:
            expires: float = float(serve_session[session_id]["expires"])  # type: ignore
        except KeyError:
            continue
        except Exception:
            expires = time.time()

        if expires < time.time():
            serve_session.pop(session_id)

    with open(SERVE_SESSION_CACHE_FILE, "w") as f:
        json.dump(serve_session, f)
