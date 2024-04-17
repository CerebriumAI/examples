from typing import Annotated, Optional

import typer
from cerebrium import datatypes

from cerebrium.api import HttpMethod, cerebrium_request
from cerebrium.utils.logging import cerebrium_log

storage_app = typer.Typer(no_args_is_help=True)


@storage_app.command("get-capacity")
def capacity():
    """A utility to view persistent storage capacity"""
    response = cerebrium_request(
        HttpMethod.GET,
        "get-storage-capacity",
    )
    if response is None:
        cerebrium_log(
            level="ERROR",
            message="There was an error getting your storage capacity. Please login and try again.\nIf the problem persists, please contact support.",
            prefix="",
        )
        exit()
    if response.status_code == 200:
        storage = response.json()["capacity"]
        print(f"📦 Storage capacity: {storage} GB")
    else:
        cerebrium_log(
            level="ERROR",
            message=f"There was an error getting your storage capacity.\n{response.json()['message']}",
            prefix="",
        )


@storage_app.command("increase-capacity")
def increase_capacity(
    increase: Annotated[
        int,
        typer.Argument(
            help="Increase storage capacity by the given number of GB. Warning: storage cannot be decreased once allocated and this will increase your monthly bill.",
            min=0,
        ),
    ] = 0,
    region: Optional[str] = typer.Option(
        default=None,
        help="The region to increase storage capacity in. If not provided, the default region will be used.",
    ),
):
    """A utility to increase persistent storage capacity."""
    if increase > 150:
        raise ValueError(
            "Increase value cannot be more than 150GB at a time. Otherwise, contact support if this is needed."
        )
    if region is not None and region not in ["aws", "coreweave"]:
        raise ValueError("Region must be one of: [aws, coreweave]")
    print(f"📦 Increasing storage capacity by {increase}GB...")
    response = cerebrium_request(
        HttpMethod.POST,
        "increase-storage-capacity",
        (
            {"increaseInGB": increase}
            if region is None
            else {"increaseInGB": increase, "region": region}
        ),
    )
    if response is None:
        cerebrium_log(
            level="ERROR",
            message="There was an error updating your storage capacity. Please login and try again.\nIf the problem persists, please contact support.",
            prefix="",
        )
        exit()
    if response.status_code == 200:
        new_size = response.json()["capacity"]
        print(
            f"✅ Storage capacity{' in '+region if region else ''} successfully increased to {new_size} GB."
        )
    else:
        cerebrium_log(
            level="ERROR",
            message=f"There was an error updating your storage capacity.\n{response.json()['message']}",
            prefix="",
        )
