import os

import typer
import yaml
from typing_extensions import Annotated

import cerebrium.utils as utils
from cerebrium.api import HttpMethod, cerebrium_request, env
from cerebrium.utils.logging import cerebrium_log

project_app = typer.Typer(no_args_is_help=True)
env = os.getenv("ENV", "prod")


@project_app.command("current")
def current():
    """
    Get the current project you are working in
    """
    print(f"projectId: {utils.get_current_project_context()}")


@project_app.command("list")
def list_projects():
    """
    List all your projects
    """
    projects_response = cerebrium_request(HttpMethod.GET, "projects", {})
    if projects_response is None:
        cerebrium_log(
            level="ERROR",
            message="There was an error getting your projects. Please login again and, if the problem persists, contact support.",
            prefix="",
        )
        return

    if projects_response.status_code != 200:
        cerebrium_log(
            level="ERROR",
            message="There was an error getting your projects",
            prefix="",
        )
        return

    if projects_response.status_code == 200:
        projects = projects_response.json()["projects"]
        for project in projects:
            print(f"{project['projectId']} : {project['name']}")
        print("")
        print(
            f"You can set your current project context by running 'cerebrium project set {projects[0]['projectId']}"
        )


@project_app.command("set")
def set_project(
    project_id: Annotated[
        str,
        typer.Argument(
            help="The projectId of the project you would like to work in",
        ),
    ],
):
    """
    Set the project context you are working in.
    """
    config_path = os.path.expanduser("~/.cerebrium/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    key_name = ""
    if env == "dev":
        key_name = "dev-"
    elif env == "local":
        key_name = "local-"
    config[f"{key_name}project"] = project_id
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Project context successfully set to : {project_id}")
