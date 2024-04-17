from typing import Dict, List, Union

import typer
from rich import box
from rich import print as console
from rich.panel import Panel
from rich.table import Table
from typing_extensions import Annotated, Optional

from cerebrium.api import HttpMethod, cerebrium_request
from cerebrium.utils.display import colorise_status_for_rich, pretty_timestamp
from cerebrium.utils.logging import cerebrium_log

app = typer.Typer(no_args_is_help=True)


@app.command("list")
def list():
    """
    List all apps under your current context
    """
    app_response = cerebrium_request(HttpMethod.GET, "get-models", {})
    if app_response is None:
        cerebrium_log(
            level="ERROR",
            message="There was an error getting your apps. Please login and try again.\nIf the problem persists, please contact support.",
            prefix="",
        )
        exit()
    if app_response.status_code != 200:
        cerebrium_log(
            level="ERROR", message="There was an error getting your apps", prefix=""
        )
        return

    apps = app_response.json()

    apps_to_show: List[Dict[str, str]] = []
    for a in apps["models"]:
        # if isinstance(a, list):
        replicas = a.get("replicas", ["None"])
        replicas = [r for r in replicas if r != ""]
        # convert updated at from 2023-11-13T20:57:12.640Z to human readable format
        updated_at = pretty_timestamp(a.get("updatedAt", "None"))

        apps_to_show.append(
            {
                "id": f'{a["projectId"]}-{a["name"]}',
                "name": f'{a["name"]}',
                "status": colorise_status_for_rich(a["status"]),
                "replicas": str(replicas),
                "updatedAt": updated_at,
            }
        )

    # sort by updated date
    apps_to_show = sorted(apps_to_show, key=lambda k: k["updatedAt"], reverse=True)

    # Create the table
    table = Table(title="", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("ModelId")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Replicas", justify="center")
    table.add_column("Last Updated", justify="center")

    for entry in apps_to_show:
        table.add_row(
            entry["id"],
            entry["name"],
            entry["status"],
            "\n".join(entry["replicas"]),
            entry["updatedAt"],
        )

    details = Panel.fit(
        table,
        title="[bold] App Details ",
        border_style="yellow bold",
        width=140,
        padding=(1, 1),
    )
    console(details)


@app.command("get")
def get(
    app_id: Annotated[
        str,
        typer.Argument(
            ...,
            help="The app-id you would like to see the details",
        ),
    ],
):
    """
    Get specific details around a application
    """
    app_response = cerebrium_request(
        HttpMethod.GET, f"get-model-details?modelId={app_id}", {}
    )

    if app_response is None:
        cerebrium_log(
            level="ERROR",
            message=f"There was an error getting the details of app {app_id}. Please login and try again.\nIf the problem persists, please contact support.",
            prefix="",
        )
        exit()
    if app_response.status_code != 200:
        cerebrium_log(
            level="ERROR",
            message=f"There was an error getting the details of app {app_id}.\n{app_response.json()['message']}",
            prefix="",
        )
        return

    json_response = app_response.json()

    table = make_detail_table(json_response)
    details = Panel.fit(
        table,
        title=f"[bold] App Details for {app_id} [/bold]",
        border_style="yellow bold",
        width=100,
        padding=(1, 1),
    )
    print()
    console(details)
    print()


@app.command("delete")
def delete(
    name: Annotated[str, typer.Argument(..., help="Name of the Cortex deployment.")],
):
    """
    Delete a model or training job from Cerebrium
    """
    print(f'Deleting model "{name}" from Cerebrium...')
    delete_response = cerebrium_request(
        HttpMethod.DELETE, "delete-model", {"name": name}
    )
    if delete_response is None:
        cerebrium_log(
            level="ERROR",
            message=f"There was an error deleting {name}. Please login and try again.\nIf the problem persists, please contact support.",
            prefix="",
        )
        exit()
    if delete_response.status_code == 200:
        print("✅ Model deleted successfully.")
    else:
        print(f"❌ Model deletion failed.\n{delete_response.json()['message']}")


@app.command("scale")
def model_scaling(
    name: Annotated[str, typer.Argument(..., help="The name of your model.")],
    cooldown: Annotated[
        Optional[int],
        typer.Option(
            ...,
            min=0,
            help=(
                "Update the cooldown period of your deployment. "
                "This is the number of seconds before your app is scaled down to 0."
            ),
        ),
    ] = None,
    min_replicas: Annotated[
        Optional[int],
        typer.Option(
            ...,
            min=0,
            help=(
                "Update the minimum number of replicas to keep running for your deployment."
            ),
        ),
    ] = None,
    max_replicas: Annotated[
        Optional[int],
        typer.Option(
            ...,
            min=1,
            help=(
                "Update the maximum number of replicas to keep running for your deployment."
            ),
        ),
    ] = None,
):
    """
    Change the cooldown, min and max replicas of your deployment via the CLI
    """
    if (
        max_replicas is not None
        and min_replicas is not None
        and max_replicas <= min_replicas
    ):
        cerebrium_log(
            message="Maximum replicas must be greater than or equal to minimum replicas.",
            level="ERROR",
        )

    print(f"Updating scaling for model '{name}'...")
    if cooldown is not None:
        print(f"\tSetting cooldown to {cooldown} seconds...")
    if min_replicas is not None:
        print(f"\tSetting minimum replicas to {min_replicas}...")
    if max_replicas is not None:
        print(f"\tSetting maximum replicas to {max_replicas}...")

    body = {}
    if cooldown is not None:
        body["cooldownPeriodSeconds"] = cooldown
    if min_replicas is not None:
        body["minReplicaCount"] = min_replicas
    if max_replicas is not None:
        body["maxReplicaCount"] = max_replicas
    if not body:
        print("Nothing to update...")
        print("Cooldown, minReplicas and maxReplicas are all None ✅")

    body["name"] = name
    update_response = cerebrium_request(HttpMethod.POST, "update-model-scaling", body)

    if update_response is None:
        cerebrium_log(
            level="ERROR",
            message=f"There was an error scaling {name}. Please login and try again.\nIf the problem persists, please contact support.",
            prefix="",
        )
        exit()

    if update_response.status_code == 200:
        print("✅ Model scaled successfully.")
    else:
        cerebrium_log(
            level="ERROR", message=f"There was an error scaling {name}", prefix=""
        )


def make_detail_table(data: Dict[str, Union[str, int, List[str]]]):
    def get(key: str):
        return str(data.get(key)) if data.get(key) else "Data Unavailable"

    def addRow(
        leader: str,
        key: str = "",
        value: Union[str, None] = None,
        ending: str = "",
        optional: bool = False,
    ):
        if value is None:
            if key not in data:
                ending = ""
            if optional:
                if data.get(key):
                    table.add_row(leader, get(key) + ending)
            else:
                table.add_row(leader, get(key) + ending)
        else:
            table.add_row(leader, str(value))

    # Create the tables
    table = Table(box=box.SIMPLE_HEAD)
    table.add_column("Parameter", style="")
    table.add_column("Value", style="")
    table.add_row("MODEL", "", style="bold")
    table.add_row("Name", str(data.get("name")))
    addRow("Average Runtime", "averageModelRunTimeSeconds", ending="s")
    addRow("Cerebrium Version", "cerebriumVersion")
    addRow("Created At", "createdAt", pretty_timestamp(get("createdAt")))
    if get("createdAt") != get("updatedAt"):
        addRow("Updated At", "updatedAt", pretty_timestamp(get("updatedAt")))

    table.add_row("", "")
    table.add_row("HARDWARE", "", style="bold")
    table.add_row("GPU", get("hardware"))
    addRow("CPU", "cpu", ending=" cores")
    addRow("Memory", "memory", ending=" GB")
    if get("hardware") != "CPU" and "hardware" in data:
        addRow("GPU Count", "gpuCount")

    table.add_row("", "")
    table.add_row("SCALING PARAMETERS", "", style="bold")
    addRow("Cooldown Period", key="cooldownPeriodSeconds", ending="s")
    addRow("Minimum Replicas")
    if "maxReplicaCount" in data:
        addRow("Maximum Replicas", key="maxReplicaCount", optional=True)

    table.add_row("", "")
    table.add_row("STATUS", "", style="bold")
    addRow("Status", "status", value=colorise_status_for_rich(get("status")))
    addRow("Last Build Status", value=colorise_status_for_rich(get("lastBuildStatus")))
    addRow("Last Build Version", value=get("latestBuildVersion"), optional=True)

    pods = data.get("pods", "")
    if isinstance(pods, List):
        pods = "\n".join(pods)
    if data.get("pods"):
        table.add_row("", "")

        table.add_row(
            "[bold]LIVE PODS[/bold]", str(pods) if pods else "Data Unavailable"
        )

    return table
