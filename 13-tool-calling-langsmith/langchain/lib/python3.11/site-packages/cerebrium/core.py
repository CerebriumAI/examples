import json
import os
from typing import Dict

import yaml
from typer import Typer

from cerebrium.utils.logging import cerebrium_log

cli = Typer(no_args_is_help=True)

IS_SERVER = os.getenv("IS_SERVER", "false")
if os.path.exists("secrets.json"):
    with open("secrets.json") as f:
        remote_secrets = json.load(f)
elif os.path.exists("secrets.yaml"):
    with open("secrets.yaml") as f:
        remote_secrets = yaml.load(f, Loader=yaml.FullLoader)
else:
    remote_secrets: Dict[str, str] = {}


def get_secret(key: str):
    secret = remote_secrets.get(key, "") if remote_secrets else os.getenv(key, "")
    if secret == "":
        cerebrium_log(
            level="ERROR",
            message=f"Secret not found for key: {key}, please check your environment variables.",
        )
    else:
        return secret
