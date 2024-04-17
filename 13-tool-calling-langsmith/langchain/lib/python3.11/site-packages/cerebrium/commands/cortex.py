import inspect
import json
import os
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import requests

import typer
from rich import print as console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from termcolor import colored
from typing_extensions import Annotated
from yaspin import yaspin  # type: ignore

import cerebrium.api as api
import cerebrium.utils.sync_files as sync_files
from cerebrium import __version__ as cerebrium_version
from cerebrium import constants, datatypes, utils, verification
from cerebrium.api import HttpMethod, cerebrium_request
from cerebrium.core import cli
from cerebrium.utils import logging
from cerebrium.utils.config import archive_file, update_config_from_files
from cerebrium.utils.logging import cerebrium_log
from cerebrium.utils.misc import get_function_params
from cerebrium.utils.sync_files import upload_files_to_s3, upload_marker_file_and_delete

_EXAMPLE_MAIN = """
from typing import Optional
from pydantic import BaseModel


class Item(BaseModel):
    prompt: str
    your_param: Optional[str] = None # an example optional parameter


def predict(item, run_id, logger):
    item = Item(**item)

    my_results = {"prediction": item.prompt, "your_optional_param": item.your_param}
    my_status_code = 200 # if you want to return some status code

    return {"my_result": my_results, "status_code": my_status_code} # return your results
"""


def was_provided(param_name: str, param_value: Any, defaults: Dict[str, Any]) -> bool:
    """Check if a parameter was explicitly provided by comparing against its default value."""
    return defaults.get(param_name, object()) != param_value


@cli.command("init")
def init(
    init_dir: Annotated[
        str,
        typer.Argument(
            help="Directory where you would like to init a Cortex project.",
        ),
    ] = ".",
    name: Annotated[
        Optional[str], typer.Option(help="Name of the Cortex deployment.")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Flag to overwrite contents of the init_dir.")
    ] = False,
    pip: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Optional list of requirements. "
                "Example: \"['transformers', 'torch==1.31.1']\""
            ),
        ),
    ] = None,
    apt: Annotated[
        str,
        typer.Option(
            help=("Optional list of apt packages. For example: \"['git', 'ffmpeg' ]\""),
        ),
    ] = "",
    conda: Annotated[str, typer.Option(help="Optional list of conda packages.")] = "",
    gpu: Annotated[
        str,
        typer.Option(
            help=(
                "Hardware to use for the Cortex deployment. "
                "Defaults to 'GPU'. "
                f"Can be one of: {datatypes.HardwareOptions.available_hardware()}"
            ),
        ),
    ] = constants.DEFAULT_GPU_SELECTION,
    cpu: Annotated[
        int,
        typer.Option(
            min=constants.MIN_CPU,
            max=constants.MAX_CPU,
            help=(
                "Number of vCPUs (cores) to use for the Cortex deployment. "
                "Defaults to 2. Can be an integer between 1 and 48"
            ),
        ),
    ] = constants.DEFAULT_CPU,
    memory: Annotated[
        float,
        typer.Option(
            min=constants.MIN_MEMORY,
            max=constants.MAX_MEMORY,
            help=(
                "Amount of memory (in GB) to use for the Cortex deployment. "
                "Defaults to 14.5GB. "
                "Can be a float between 2.0 and 256.0 depending on hardware selection."
            ),
        ),
    ] = constants.DEFAULT_MEMORY,
    gpu_count: Annotated[
        int,
        typer.Option(
            min=0,
            max=constants.MAX_GPU_COUNT,
            help=(
                "Number of GPUs to use for the Cortex deployment. "
                "Defaults to 1. Can be an integer between 1 and 8."
            ),
        ),
    ] = constants.DEFAULT_GPU_COUNT,
    include: Annotated[
        str,
        typer.Option(
            help=(
                "Comma delimited string list of relative paths to files/folder to include. "
                "Defaults to all visible files/folders in project root."
            ),
        ),
    ] = constants.DEFAULT_INCLUDE,
    exclude: Annotated[
        str,
        typer.Option(
            help=(
                "Comma delimited string list of relative paths to files/folder to exclude. "
                "Defaults to all hidden files/folders in project root."
            ),
        ),
    ] = constants.DEFAULT_EXCLUDE,
    log_level: Annotated[
        Optional[str],
        typer.Option(
            help="Log level for the Cortex deployment. Can be one of 'DEBUG' or 'INFO'",
        ),
    ] = constants.DEFAULT_LOG_LEVEL,
    disable_animation: Annotated[
        Optional[bool],
        typer.Option(
            help="Whether to use TQDM and yaspin animations.",
        ),
    ] = bool(os.getenv("CI")),
    cuda_version: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "CUDA version to use. "
                "Currently, we support 11.8 as '11' and 12.2 as '12'. Defaults to '12'"
            ),
        ),
    ] = constants.DEFAULT_CUDA_VERSION,
    provider: Annotated[
        str,
        typer.Option(
            help=(
                "Provider to use for the Cortex deployment. "
                "Defaults to 'coreweave'. "
                "Can be one of "
                "'aws', "
                "'coreweave', "
            ),
        ),
    ] = "",
):
    """
    Initialize an empty Cerebrium Cortex project.
    """

    if gpu:
        vals = datatypes.HardwareOptions.available_hardware()
        if gpu not in vals:
            cerebrium_log(message=f"Hardware must be one of {vals}", level="ERROR")
        gpu = getattr(datatypes.HardwareOptions, gpu).name

    if not name:
        name = os.path.basename(os.path.abspath(init_dir))
        name = name.replace("_", "-")
        # remove all non alpha-numerical digits that aren't '-'
        name = re.sub(r"[^a-zA-Z0-9-]", "", name)
        name = name.lower()

    print(f"Initializing Cerebrium Cortex project in {init_dir}")
    pip = (
        pip
        if pip
        else (
            "['transformers', 'torch>=2.0.0', 'pydantic']"
            if cuda_version == "12"
            else "['transformers', 'torch<2.0.0', 'pydantic']"
        )
    )

    if not os.path.exists(init_dir):
        os.makedirs(init_dir)
    elif os.listdir(init_dir) and not overwrite:
        cerebrium_log(
            level="WARNING",
            message="Directory is not empty. "
            "Use an empty directory or use the `--overwrite` flag.",
            prefix_seperator="\t",
        )

    if not os.path.exists(os.path.join(init_dir, "main.py")):
        with open(os.path.join(init_dir, "main.py"), "w") as f:
            f.write(_EXAMPLE_MAIN)

    config = {
        "name": name,
        "gpu": gpu,
        "cpu": cpu,
        "memory": memory,
        "log_level": log_level,
        "include": include,
        "exclude": exclude,
        "cooldown": constants.DEFAULT_COOLDOWN,
        "gpu_count": gpu_count,
        "min_replicas": constants.DEFAULT_MIN_REPLICAS,
        "disable_predict": False,
        "force_rebuild": False,
        "disable_confirmation": False,
        "cuda_version": cuda_version,
        "provider": provider,
    }
    if disable_animation is not None:
        config["disable_animation"] = disable_animation

    requirements_list = pip.strip("[]").split(",")
    requirements_list = [r.strip().strip("'").strip('"') for r in requirements_list]
    pkg_list = apt.strip("[]").split(",")
    pkg_list = [p.strip().strip("'").strip('"') for p in pkg_list]
    conda_pkglist = conda.strip("[]").split(",")
    conda_pkglist = [c.strip().strip("'").strip('"') for c in conda_pkglist]
    # if any of the lists only contain an empty string, set to empty list
    pip_dict = utils.requirements.req_list_to_dict(requirements_list)
    apt_dict = utils.requirements.req_list_to_dict(pkg_list)
    conda_dict = utils.requirements.req_list_to_dict(conda_pkglist)

    utils.tomls.legacy_to_toml_structure(
        name=name,
        legacy_params=config,
        config_file=os.path.join(init_dir, "cerebrium.toml"),
        pip=pip_dict,
        apt=apt_dict,
        conda=conda_dict,
        overwrite=overwrite,
    )

    print("🚀 Cerebrium Cortex project initialized successfully!")


@cli.command("deploy")
def deploy(
    name: Annotated[
        Optional[str], typer.Option(help="Name of the Cortex deployment.")
    ] = None,
    disable_syntax_check: Annotated[
        Optional[bool], typer.Option(help="Flag to disable syntax check.")
    ] = None,
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
    min_replicas: Annotated[
        Optional[int],
        typer.Option(
            min=0,
            max=200,
            help=(
                "Minimum number of replicas to create on the Cortex deployment. "
                "Defaults to 0."
            ),
        ),
    ] = None,
    max_replicas: Annotated[
        Optional[int],
        typer.Option(
            min=1,
            max=200,
            help=(
                "A hard limit on the maximum number of replicas to allow. "
                "Defaults to 2 for free users. "
                "Enterprise and standard users are set to maximum specified in their plan"
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
    cooldown: Annotated[
        Optional[int],
        typer.Option(
            help="Cooldown period in seconds before an inactive replica of your deployment is scaled down. Defaults to 60s.",
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
    disable_confirmation: Annotated[
        Optional[bool],
        typer.Option(
            "--disable-confirmation",
            "-q",
            "-y",
            help="Whether to disable the confirmation prompt before deploying.",
        ),
    ] = None,
    disable_predict: Annotated[
        Optional[bool], typer.Option(help="Flag to disable running predict function.")
    ] = None,
    disable_animation: Annotated[
        Optional[bool],
        typer.Option(
            help="Whether to use TQDM and yaspin animations.",
        ),
    ] = None,
    disable_build_logs: Annotated[
        Optional[bool],
        typer.Option(help="Whether to disable build logs during a deployment."),
    ] = None,
    hide_public_endpoint: Annotated[
        Optional[bool],
        typer.Option(
            help="Whether to hide the public endpoint of the deployment when printing the logs.",
        ),
    ] = None,
    cuda_version: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "CUDA version to use. "
                "Currently, we support 11.8 as '11' and 12.2 as '12'. Defaults to '12'"
            ),
        ),
    ] = "",
    provider: Annotated[
        Optional[str],
        typer.Option(
            help=f"Provider to deploy to. Can be one of: 'aws' or 'coreweave'. This is auto-populated if left unspecified"
        ),
    ] = None,
    region: Annotated[
        Optional[str],
        typer.Option(
            help=f"Region to deploy to. Can be one of: 'us-east-1' or 'us-west-1'. This is auto-populated if left unspecified"
        ),
    ] = None,
):
    """
    Deploy a Cortex deployment to Cerebrium
    """

    if config_file is None or not config_file.strip():
        config_file = "cerebrium.toml"
    # func_defaults contains all parameters with their default values
    func_defaults = get_function_params(deploy)

    # provided_params only include explicitly provided parameters
    provided_params = {
        param: value
        for param, value in locals().items()
        if param in func_defaults and was_provided(param, value, func_defaults)
    }

    # Merge func_defaults with provided_params. Provided values override default values where applicable.
    final_params = {**func_defaults, **provided_params}

    # load config toml file and merge with param values
    config_obj = utils.misc.merge_config_with_params("cerebrium.toml", final_params)

    ##validation is done with types in classes
    cerebrium_config = datatypes.CerebriumConfig(
        scaling=datatypes.CerebriumScaling(**config_obj["scaling"]),
        build=datatypes.CerebriumBuild(**config_obj["build"]),
        deployment=datatypes.CerebriumDeployment(**config_obj["deployment"]),
        hardware=datatypes.CerebriumHardware(**config_obj["hardware"]),
        dependencies=datatypes.CerebriumDependencies(**config_obj["dependencies"]),
        cerebrium_version=cerebrium_version,
    )
    # Check if there has been any changes to the requirements, apt or conda packages. If so, update the config

    if update_config_from_files(
        config=cerebrium_config,
        archive_files=False,
        config_file=config_file,
        quiet=True,
    ):
        cerebrium_log(
            message="Environment requirements changed.\n Automatically updated the config file with the new requirements",
            color="green",
        )

    build_status, setup_response = package_app(
        cerebrium_config, datatypes.OperationType.DEPLOY
    )
    if "success" == build_status:
        project_id = setup_response["projectId"]
        jwt = setup_response["jwt"]

        if constants.env == "prod":
            endpoint = f"https://run.cerebrium.ai/v3/{project_id}/{cerebrium_config.deployment.name}/predict"
        else:
            endpoint = f"https://dev-run.cerebrium.ai/v3/{project_id}/{cerebrium_config.deployment.name}/predict"

        dashboard_url = f"{api.dashboard_url}/projects/{project_id}/models/{project_id}-{cerebrium_config.deployment.name}"

        info_string = (
            f"🔗 [link={dashboard_url}]View your deployment dashboard here[/link]\n"
            f"🔗 [link={dashboard_url}?tab=builds]View builds here[/link]\n"
            f"🔗 [link={dashboard_url}?tab=runs]View runs here[/link]\n\n"
            f"🛜  Endpoint:\n{endpoint}"
        )

        dashboard_info = Panel(
            info_string,
            title=f"[bold green]🚀 {cerebrium_config.deployment.name} is now live! 🚀 ",
            border_style="green",
            width=100,
            padding=(1, 2),
        )

        console(Group(dashboard_info))

        curl_command = colored(
            f"curl -X POST {endpoint} \\\n"
            "     -H 'Content-Type: application/json'\\\n"
            f"     -H 'Authorization: {jwt}'\\\n"
            '     --data \'{"prompt": "Hello World!"}\'',
            "green",
        )
        print(
            "\n💡You can call the endpoint with the following curl command:\n"
            f"{curl_command}"
        )
    elif build_status in ["build_failure", "init_failure"]:
        console(
            Text(
                "Unfortunately there was an issue with your deployment",
                style="red",
            )
        )


@cli.command("build")
def build(
    name: Annotated[str, typer.Option(help="Name of the Cortex deployment.")] = "",
    disable_syntax_check: Annotated[
        bool, typer.Option(help="Flag to disable syntax check.")
    ] = False,
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
    python_version: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Python version to use. "
                "Currently, we support '3.8' to '3.11'. Defaults to '3.10'"
            ),
        ),
    ] = None,
    predict: Annotated[
        Optional[str],
        typer.Option(
            help="JSON string containing all the parameters that will be used to run your "
            "deployment's predict function on build to ensure your new deployment will work "
            "as expected before replacing your existing deployment.",
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
            help="Comma delimited string list of relative paths to files/folder to exclude. Defaults to all hidden files/folders in project root.",
        ),
    ] = None,
    force_rebuild: Annotated[Optional[bool], typer.Option()] = False,
    config_file: Annotated[
        Optional[str],
        typer.Option(
            help="Path to cerebrium.toml file. You can generate a config using `cerebrium init-cortex`. The contents of the deployment config file are overridden by the command line arguments.",
        ),
    ] = None,
    log_level: Annotated[
        Optional[str],
        typer.Option(
            help="Log level for the Cortex build. Can be one of 'DEBUG' or 'INFO'"
        ),
    ] = "",
    disable_confirmation: Annotated[
        Optional[bool],
        typer.Option(
            "--disable-confirmation",
            "-q",
            "-y",
            help="Whether to disable the confirmation prompt before deploying.",
        ),
    ] = None,
    disable_predict: Annotated[
        Optional[bool], typer.Option(help="Flag to disable running predict function.")
    ] = None,
    disable_animation: Annotated[
        Optional[bool],
        typer.Option(
            help="Whether to use TQDM and yaspin animations.",
        ),
    ] = None,
    disable_build_logs: Annotated[
        Optional[bool],
        typer.Option(help="Whether to disable build logs during a deployment."),
    ] = None,
    hide_public_endpoint: Annotated[
        Optional[bool],
        typer.Option(
            help="Whether to hide the public endpoint of the deployment when printing the logs.",
        ),
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
    provider: Annotated[
        Optional[str],
        typer.Option(
            help=f"Provider to deploy to. Can be one of: 'aws' or 'coreweave'. This is auto-populated if left unspecified"
        ),
    ] = None,
    region: Annotated[
        Optional[str],
        typer.Option(
            help=f"Region to deploy to. Can be one of: 'us-east-1' or 'us-central-1'. This is auto-populated if left unspecified"
        ),
    ] = None,
):
    """
    Build and run your Cortex files on Cerebrium to verify that they're working as expected.
    """
    if config_file is None or not config_file.strip():
        config_file = "cerebrium.toml"

    # func_defaults contains all parameters with their default values
    func_defaults = {
        k: v.default
        for k, v in inspect.signature(deploy).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    # provided_params only include explicitly provided parameters
    provided_params = {
        param: value
        for param, value in locals().items()
        if param in func_defaults and was_provided(param, value, func_defaults)
    }
    # Merge func_defaults with provided_params. Provided values override default values where applicable.
    final_params = {**func_defaults, **provided_params}

    # load config toml file and merge with param values
    config_obj = utils.misc.merge_config_with_params("cerebrium.toml", final_params)

    ##validation is done with types in classes
    cerebrium_config = datatypes.CerebriumConfig(
        scaling=datatypes.CerebriumScaling(**config_obj["scaling"]),
        build=datatypes.CerebriumBuild(**config_obj["build"]),
        deployment=datatypes.CerebriumDeployment(**config_obj["deployment"]),
        hardware=datatypes.CerebriumHardware(**config_obj["hardware"]),
        dependencies=datatypes.CerebriumDependencies(**config_obj["dependencies"]),
        cerebrium_version=cerebrium_version,
    )

    if update_config_from_files(config=cerebrium_config, archive_files=True):
        cerebrium_log(message="Updated the config file with the new requirements")
        archive_file(config_file)
        # write the updated config to the config file
        utils.tomls.save_config_to_toml_file(
            cerebrium_config, config_file.strip() or "cerebrium.toml", quiet=True
        )

    build_status, _ = package_app(cerebrium_config, datatypes.OperationType.RUN)
    if "success" in build_status:
        console(
            Text(
                "Unfortunately there was an issue with running your deployment",
                style="red",
            )
        )


def package_app(
    cerebrium_config: datatypes.CerebriumConfig, app_type: datatypes.OperationType
):
    # Get the files in the users directory
    cerebrium_config.file_list = utils.determine_includes(
        include=cerebrium_config.deployment.include,
        exclude=cerebrium_config.deployment.exclude,
    )
    if (
        "./main.py" not in cerebrium_config.file_list
        and "main.py" not in cerebrium_config.file_list
    ):
        cerebrium_log(
            "⚠️ main.py not found. Please ensure your project has a main.py file.",
            level="ERROR",
        )

    if not cerebrium_config.build.disable_syntax_check:
        verification.run_pyflakes(files=cerebrium_config.file_list, print_warnings=True)

    if cerebrium_config.build.disable_predict:
        cerebrium_config.build.predict_data = None

    cerebrium_config.partial_upload = False
    # If files are larger than 10MB, use partial_upload and localFiles otherwise upload with app zip
    if (
        utils.files.check_deployment_size(cerebrium_config.file_list, 10)
        or len(cerebrium_config.file_list) > 500
    ):
        if len(cerebrium_config.file_list) < 1000:
            print("📦 Large upload, only uploading files that have changed...")
            cerebrium_config.partial_upload = True
        else:
            cerebrium_log(
                "⚠️ 1000+ files detected. Partial sync not possible. Try reduce the number of files or file size for faster deployments.",
                level="ERROR",
            )
            exit()

    if cerebrium_config.partial_upload:
        setup_response = do_partial_upload(cerebrium_config, type=app_type)
    else:
        params = utils.misc.flatten_cerebrium_config_to_json(config=cerebrium_config)
        params["function"] = app_type.value
        setup_response = _setup_request(cerebrium_config, params)

    build_id = setup_response["buildId"]
    print(f"🆔 Build ID: {build_id}")
    build_status = str(setup_response["status"])

    spinner = None

    if build_status == "pending":
        if not cerebrium_config.partial_upload:
            api.upload_cortex_files(
                upload_url=setup_response["uploadUrl"],
                zip_file_name=os.path.basename(setup_response["keyName"]),
                config=cerebrium_config,
            )

        build_status = "Build pending..."
        start_time = time.time()

        spinner = (
            Spinner("dots", "Building App...", style="gray")
            if not cerebrium_config.build.disable_animation
            else None
        )

        # This is used to stop the logs on a different thread
        start_event = threading.Event()
        stop_event = threading.Event()

        if constants.env == "local":
            log_thread = threading.Thread(
                target=api.poll_build_logs,
                args=(
                    setup_response["buildId"],
                    start_event,
                    stop_event,
                ),
            )
        else:
            log_thread = threading.Thread(
                target=api.stream_logs,
                args=(
                    start_event,
                    stop_event,
                    f'{setup_response["projectId"]}-{cerebrium_config.deployment.name}',
                    setup_response["buildId"],
                ),
            )

        log_thread.start()
        live = Live(spinner, console=logging.console, refresh_per_second=10)

        # Start the Live context using the start() method
        live.start()
        build_status = ""
        try:
            while True:
                old_build_status = build_status
                build_status = api.get_build_status(setup_response["buildId"])
                if spinner:
                    spinner.text = api.log_build_status(build_status, start_time)
                elif old_build_status != build_status:
                    print(api.log_build_status(build_status, start_time, mode="build"))

                if build_status in ["success", "build_failure", "init_failure"]:
                    start_event.wait()
                    live.update(Text(""))
                    stop_event.set()
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            # If user presses Ctrl-C, signal all threads to stop
            live.stop()
            __graceful_shutdown(
                build_id=build_id,
                stop_event=stop_event,
                log_thread=log_thread,
                is_interrupt=True,
            )
        finally:
            # Stop the Live instance after the loop
            live.stop()

        log_thread.join()
    elif build_status == "running":
        print("🤷 No file changes detected. Not fetching logs")
    else:
        if spinner:
            spinner.stop(text="Build failed")
        cerebrium_log("ERROR", "Build failed.")

    return build_status, setup_response


def do_partial_upload(
    cerebrium_config: datatypes.CerebriumConfig,
    type: datatypes.OperationType,
) -> Dict[str, Any]:
    """
    Partial uploads need to be done in a different flow to the normal upload.

    This function will:
    - Create all utility files (requirements.txt, apt.txt, conda.txt, _cerebrium_predict.json, etc) and remove any conflicting files
    - Get the hashes of all the files
    - Compare the hashes to the hashes of the last deployment
    - Upload the files that have changed
    - Create a marker file with the hashes of the new files

    """
    temp_file_list: List[str] = cerebrium_config.file_list.copy()

    # make utility files
    sync_files.make_cortex_util_files(working_dir=os.getcwd(), config=cerebrium_config)
    current_dir_files = os.listdir(os.getcwd())

    temp_file_list.extend(
        [f for f in constants.INTERNAL_FILES if f in current_dir_files],
    )
    cerebrium_config.local_files = sync_files.gather_hashes(temp_file_list)

    params = utils.misc.flatten_cerebrium_config_to_json(config=cerebrium_config)
    params["function"] = type.value

    setup_response = _setup_request(cerebrium_config, params)

    if setup_response["status"] == "pending":
        uploaded_count = upload_files_to_s3(
            setup_response["uploadUrls"], quiet=cerebrium_config.build.disable_animation
        )
        upload_marker_file_and_delete(
            setup_response["markerFile"],
            uploaded_count,
            setup_response["buildId"],
            cerebrium_config.local_files,
        )

    return setup_response


def _setup_request(
    cerebrium_config: datatypes.CerebriumConfig,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    # Include the predict data in the content hash to trigger a rebuild if the predict changes
    files_hash = utils.files.content_hash(
        cerebrium_config.file_list, strings=cerebrium_config.build.predict_data
    )
    params["upload_hash"] = files_hash

    if not utils.display.confirm_deployment(
        cerebrium_config, "deploy", cerebrium_config.build.disable_confirmation
    ):
        sys.exit()

    setup_response = cerebrium_request(HttpMethod.POST, "setupApp", params)
    if setup_response is None:
        cerebrium_log(
            level="ERROR",
            message="❌ There was an error deploying your app. Please login and try again. If the error continues to persist, contact support.",
            prefix="",
        )
        exit()

    if setup_response.status_code != 200:
        cerebrium_log(
            message=f"❌ There was an error deploying your app\n{setup_response.json()['message']}",
            prefix="",
            level="ERROR",
        )
        exit()
    print("✅ App setup complete!")

    return setup_response.json()


def __graceful_shutdown(
    build_id: str,
    stop_event: threading.Event,
    log_thread: Optional[threading.Thread],
    is_interrupt: bool = False,
):
    """
    This function is called when the user presses Ctrl+C while streaming logs.
    - stops the spinner
    - sends a kill signal to the backend to stop the build job
    - prints a message
    - exits the program
    """
    stop_event.set()
    if is_interrupt:
        cerebrium_log(
            "\n\nCtrl+C detected. Shutting down current build...", color="yellow"
        )

    try:
        response = api.cerebrium_request(
            http_method=HttpMethod.DELETE,
            url="build",
            payload={"buildId": build_id},
        )
        if response is None:
            cerebrium_log(
                message="Error ending build. Please check your internet connection and try again.\nIf the problem persists, please contact support.",
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

    if log_thread is not None:
        log_thread.join()
