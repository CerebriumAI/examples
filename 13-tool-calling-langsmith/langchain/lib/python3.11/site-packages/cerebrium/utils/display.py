import json
from typing import Any, Dict, List, Union

import typer
from rich import box
from rich import print as console
from rich.panel import Panel
from rich.table import Table

from cerebrium import datatypes
from cerebrium.utils import requirements

error_messages = {
    "disk quota exceeded": "💾 You've run out of space in your /persistent-storage. \n"
    "You can add more by running the command: `cerebrium storage increase-capacity <the_amount_in_GB>`"
}  # Error messages to check for


def dict_pretty_print(ugly: Union[Dict[Any, Any], List[Any], str]) -> str:
    max_len = 200
    if isinstance(ugly, dict):
        for key, value in ugly.items():
            try:
                stringified = str(value)
                if len(stringified) > max_len:
                    ugly[key] = stringified[:max_len] + "..."
            except Exception:
                ugly[key] = "<Unable to stringify>"
        try:
            # json dump and strip the outer brackets
            pretty = json.dumps(ugly, indent=4)[1:-1]
        except Exception:
            pretty = str(ugly)
    elif isinstance(ugly, list):
        pretty = "["
        for u in ugly:
            try:
                stringified = str(u)
                if len(stringified) > max_len:
                    pretty += stringified[:max_len] + "..." + ", "
            except Exception:
                pretty += "<Unable to stringify>" + ", "
        pretty += "]"
    else:
        pretty = str(ugly)
        if len(pretty) > max_len:
            pretty = pretty[:max_len] + "..."

    return pretty


def confirm_deployment(
    config: datatypes.CerebriumConfig,
    cerebrium_function: str,
    disable_confirmation: bool = False,
):
    """
    Print out a confirmation message for the deployment
    - Display selected hardware options and configuration on a panel
    - Ask user to confirm
    """
    hardware = config.hardware
    deployment = config.deployment
    scaling = config.scaling
    build = config.build
    dependencies = config.dependencies

    if disable_confirmation:
        return True

    def addOptionalRow(key: str, value: str):
        if value:
            deployment_table.add_row(key, str(value))

    deployment_table = Table(box=box.SIMPLE_HEAD)
    deployment_table.add_column("Parameter", style="")
    deployment_table.add_column("Value", style="")

    # TODO this needs to be converted to auto display
    deployment_table.add_row("HARDWARE PARAMETERS", "", style="bold")
    deployment_table.add_row("GPU", str(hardware.gpu))
    deployment_table.add_row("CPU", str(hardware.cpu))
    deployment_table.add_row("Memory", str(hardware.memory))
    if hardware.gpu != "CPU":
        deployment_table.add_row("GPU Count", str(hardware.gpu_count))

    # NOTE Do we want to display these?
    deployment_table.add_row("Region", str(hardware.region))
    deployment_table.add_row("Provider", str(hardware.provider))

    deployment_table.add_row("", "")
    if cerebrium_function == "run":
        deployment_table.add_row("RUN PARAMETERS", "", style="bold")
    else:
        deployment_table.add_row("DEPLOYMENT PARAMETERS", "", style="bold")
    deployment_table.add_row("Python Version", str(deployment.python_version))
    deployment_table.add_row("Include pattern", str(deployment.include))
    deployment_table.add_row("Exclude pattern", str(deployment.exclude))
    deployment_table.add_row("CUDA Version", str(deployment.cuda_version))

    deployment_table.add_row("", "")
    deployment_table.add_row("SCALING PARAMETERS", "", style="bold")
    deployment_table.add_row("Cooldown", str(scaling.cooldown))
    deployment_table.add_row("Minimum Replicas", str(scaling.min_replicas))
    if scaling.max_replicas is not None:
        deployment_table.add_row("Maximum Replicas", str(scaling.max_replicas))

    deployment_table.add_row("", "")
    deployment_table.add_row("BUILD PARAMETERS", "", style="bold")
    deployment_table.add_row("Log Level", str(build.log_level))
    if build.predict_data is not None:
        predict_data = str(build.predict_data)
        if len(predict_data) > 180:
            predict_data = predict_data[:180] + "..."
        deployment_table.add_row("Predict Data", predict_data)

    for key, value in build.__dict__.items():
        if key not in ["predict_data", "log_level"]:
            addOptionalRow(key, value)

    deployment_table.add_row("", "")
    deployment_table.add_row("DEPENDENCIES", "", style="bold")

    deployment_table.add_row(
        "pip",
        "".join(requirements.req_dict_to_str_list(dependencies.pip, for_display=True)),
    )
    deployment_table.add_row(
        "apt",
        "".join(requirements.req_dict_to_str_list(dependencies.apt, for_display=True)),
    )
    deployment_table.add_row(
        "conda",
        "".join(
            requirements.req_dict_to_str_list(dependencies.conda, for_display=True)
        ),
    )

    name = deployment.name
    config_options_panel = Panel.fit(
        deployment_table,
        title=f"[bold]🧠 Deployment parameters for {name} 🧠",
        border_style="yellow bold",
        width=100,
        padding=(1, 2),
    )
    print()
    console(config_options_panel)
    print()
    return typer.confirm(
        "Do you want to continue with the deployment?",
        default=True,
        show_default=True,
    )


def colorise_status_for_rich(status: str) -> str:
    """Takes a status, returns a rich markup string with the correct color"""
    status = " ".join([s.capitalize() for s in status.split("_")])
    color = None
    if status == "Active":
        color = "green"
    elif status == "Cold":
        color = "bright_cyan"
    elif status == "Pending":
        color = "yellow"
    elif status == "Deploying":
        color = "bright_magenta"
    elif "error" in status.lower():
        color = "red"

    if color:
        return f"[bold {color}]{status}[bold /{color}]"
    else:
        return f"[bold]{status}[bold]"


def pretty_timestamp(timestamp: str) -> str:
    """Converts a timestamp from 2023-11-13T20:57:12.640Z to human readable format"""
    return timestamp.replace("T", " ").replace("Z", "").split(".")[0]
