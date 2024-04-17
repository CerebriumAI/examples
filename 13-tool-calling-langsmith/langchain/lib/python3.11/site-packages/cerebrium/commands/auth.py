import os
import time
import webbrowser
from typing import Dict

import yaml
from yaspin import yaspin  # type: ignore
from yaspin.spinners import Spinners  # type: ignore

from cerebrium.api import HttpMethod, cerebrium_request
from cerebrium.core import cli
from cerebrium.datatypes import ENV as env
from cerebrium.utils.logging import cerebrium_log


@cli.command("login")
def login():
    """
    Authenticate user via oAuth and store token in ~/.cerebrium/config.yaml
    """

    auth_response = cerebrium_request(
        HttpMethod.POST, "device-authorization", {}, False
    )
    if auth_response is None:
        cerebrium_log(
            level="ERROR",
            message="There was an error getting your device code. Please try again.\nIf the problem persists, please contact support.",
        )
        exit()
    auth_response = auth_response.json()["deviceAuthResponsePayload"]
    verification_uri = auth_response["verification_uri_complete"]

    print("You will be redirected to a URL to be authenticated")
    webbrowser.open(verification_uri, new=2)

    start_time = time.time()
    response = None
    while time.time() - start_time < 300:  # 5 minutes
        with yaspin(
            Spinners.arc, text="Waiting for authentication...", color="magenta"
        ):
            response = cerebrium_request(
                HttpMethod.POST,
                "token",
                {"device_code": auth_response["device_code"]},
                False,
            )
            assert response is not None
            if response.status_code != 400:
                break
            time.sleep(2)
    assert response is not None
    if response.status_code == 500:
        print(auth_response.json()["message"])
        return
    if response.status_code == 200:
        config_path = os.path.expanduser("~/.cerebrium/config.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        else:
            config: Dict[str, str] = {}

        key_name = ""
        if env == "dev":
            print("❗️❗️Logging in with dev API key❗️❗️")
            key_name = "dev-"
        config[f"{key_name}accessToken"] = response.json()["accessToken"]
        config[f"{key_name}refreshToken"] = response.json()["refreshToken"]
        if env == "local":
            print("❗️❗️Logging in with local API key❗️❗️")
            key_name = "local-"
        config[f"{key_name}accessToken"] = response.json()["accessToken"]
        config[f"{key_name}refreshToken"] = response.json()["refreshToken"]

        print("✅  Logged in successfully.")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        projects_response = cerebrium_request(HttpMethod.GET, "projects", {}, True)
        assert projects_response is not None
        if projects_response.status_code == 200:
            projects = projects_response.json()["projects"]
            config[f"{key_name}project"] = projects[0]["projectId"]
            print(f"Current project context set to ID: {config.get('project')}")

        with open(config_path, "w") as f:
            yaml.dump(config, f)


@cli.command("save-auth-config")
def save_auth_config(access_token: str, refresh_token: str, project_id: str):
    """
    Saves the access token, refresh token, and project ID to the config file. This function is a helper method to allow users to store credentials directly for the framework. Mostly used for CI/CD
    """
    config_path = os.path.expanduser("~/.cerebrium/config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    key_name = ""
    if env == "dev":
        print("❗️❗️Logging in with dev API key❗️❗️")
        key_name = "dev-"
    if env == "local":
        print("❗️❗️Logging in with local API key❗️❗️")
        key_name = "local-"

    config[f"{key_name}accessToken"] = access_token
    config[f"{key_name}refreshToken"] = refresh_token
    config[f"{key_name}project"] = project_id

    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print("✅ Configuration saved successfully.")
