"""
This file houses all the utils and commands for updating the configuration of a cerebrium deployment
"""

import inspect
import json
import os
from typing import Annotated, Any, Dict, Optional

import typer
import yaml

from cerebrium import __version__ as cerebrium_version
from cerebrium import datatypes, utils
from cerebrium.commands.cortex import was_provided
from cerebrium.constants import MAX_CPU, MAX_GPU_COUNT, MAX_MEMORY, MIN_CPU, MIN_MEMORY
from cerebrium.utils.config import archive_file, update_config_from_files
from cerebrium.utils.logging import cerebrium_log
from cerebrium.utils.misc import merge_config_with_params

config_app = typer.Typer(no_args_is_help=True)


@config_app.command("upgrade-yaml")
def upgrade_yaml(
    name: Annotated[str, typer.Option(help="Name of the Cortex deployment.")] = "",
    config_file: Annotated[
        str,
        typer.Option(
            help="Path to cerebrium.yaml file. You can generate a config using `cerebrium init-cortex`. The contents of the deployment config file are overwritten by the command line arguments.",
        ),
    ] = "",
):
    """Upgrade your legacy config.yaml file to the new cerebrium.toml"""
    if not config_file:
        config_file = os.path.join(os.getcwd(), "config.yaml")
    if not os.path.exists(config_file):
        cerebrium_log(
            level="ERROR",
            message=f"Config file {config_file} does not exist.",
        )

    config = yaml.safe_load(open(config_file, "r"))
    if not name and "name" in config:
        name = config["name"]
    elif not name:
        name = os.path.basename(os.getcwd())
        # remove any capital letters
        name = name.lower()
        name = name.replace(" ", "-").replace("_", "-")
        # remove any non-alphanumeric characters
        name = "".join(e for e in name if e.isalnum() or e == "-")

    utils.tomls.legacy_to_toml_structure(
        name=name,
        legacy_params=config,
        config_file=config_file,
        save_to_file=True,
        disable_confirmation=True,
    )

    print("🚀 Cerebrium project upgraded successfully!")


@config_app.command("update")
def update(
    config_file: Annotated[
        str, typer.Option(help="The path to the configuration file")
    ] = "cerebrium.toml",
    name: Annotated[str, typer.Option(help="Name of the Cortex deployment.")] = "",
    disable_syntax_check: Annotated[
        bool, typer.Option(help="Flag to disable syntax check.")
    ] = False,
    gpu: Annotated[
        str,
        typer.Option(
            help=(
                "Hardware to use for the Cortex deployment. "
                "Defaults to 'AMPERE_A10'. "
                "Can be one of "
                "'AMPERE_A10'"
                "'ADA_L4'"
                "'TURING_4000', "
                "'TURING_5000', "
                "'AMPERE_A4000', "
                "'AMPERE_A5000', "
                "'AMPERE_A6000', "
                "'AMPERE_A100'"
            ),
        ),
    ] = "",
    cpu: Annotated[
        Optional[int],
        typer.Option(
            min=MIN_CPU,
            max=MAX_CPU,
            help=(
                "Number of vCPUs to use for the Cortex deployment. Defaults to 2. "
                "Can be an integer between 1 and 48."
            ),
        ),
    ] = None,
    memory: Annotated[
        Optional[float],
        typer.Option(
            min=MIN_MEMORY,
            max=MAX_MEMORY,
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
            max=MAX_GPU_COUNT,
            help=(
                "Number of GPUs to use for the Cortex deployment. Defaults to 1. "
                "Can be an integer between 1 and 8."
            ),
        ),
    ] = None,
    python_version: Annotated[
        str,
        typer.Option(
            help=(
                "Python version to use. "
                "Currently, we support '3.8' to '3.11'. Defaults to '3.10'"
            ),
        ),
    ] = "",
    include: Annotated[
        str,
        typer.Option(
            help=(
                "Comma delimited string list of relative paths to files/folder to include. "
                "Defaults to all visible files/folders in project root."
            ),
        ),
    ] = "",
    exclude: Annotated[
        str,
        typer.Option(
            help="Comma delimited string list of relative paths to files/folder to exclude. Defaults to all hidden files/folders in project root.",
        ),
    ] = "",
    force_rebuild: Annotated[Optional[bool], typer.Option()] = False,
    log_level: Annotated[
        Optional[str],
        typer.Option(
            help="Log level for the Cortex build. Can be one of 'DEBUG' or 'INFO'"
        ),
    ] = None,
    disable_confirmation: Annotated[
        bool,
        typer.Option(
            help="Whether to disable the confirmation prompt before deploying.",
        ),
    ] = False,
    disable_predict: Annotated[
        Optional[bool], typer.Option(help="Flag to disable running predict function.")
    ] = True,
    disable_animation: Annotated[
        Optional[bool],
        typer.Option(
            help="Whether to use TQDM and yaspin animations.",
        ),
    ] = None,
    disable_build_logs: Annotated[
        bool, typer.Option(help="Whether to disable build logs during a deployment.")
    ] = False,
    hide_public_endpoint: Annotated[
        bool,
        typer.Option(
            help="Whether to hide the public endpoint of the deployment when printing the logs.",
        ),
    ] = False,
    cuda_version: Annotated[
        str,
        typer.Option(
            help=(
                "CUDA version to use. "
                "Currently, we support 11.8 as '11' and 12.2 as '12'. Defaults to '12'"
            ),
        ),
    ] = "12",
):
    """
    Update the dependencies of your cerebrium deployment in the cerebrium.toml file.
    """

    # func_defaults contains all parameters with their default values
    func_defaults = {
        k: v.default
        for k, v in inspect.signature(update).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    # provided_params only include explicitly provided parameters
    provided_params = {
        param: value
        for param, value in locals().items()
        if param in func_defaults and was_provided(param, value, func_defaults)
    }
    # Merge func_defaults with provided_params. Provided values override default values where applicable.
    final_params: Dict[str, Any] = {**func_defaults, **provided_params}

    # load config toml file and merge with param values
    config_obj = merge_config_with_params(config_file=config_file, params=final_params)

    ##validation is done with types in classes
    cerebrium_config = datatypes.CerebriumConfig(
        scaling=datatypes.CerebriumScaling(**config_obj["scaling"]),
        build=datatypes.CerebriumBuild(**config_obj["build"]),
        deployment=datatypes.CerebriumDeployment(**config_obj["deployment"]),
        hardware=datatypes.CerebriumHardware(**config_obj["hardware"]),
        dependencies=datatypes.CerebriumDependencies(**config_obj["dependencies"]),
        cerebrium_version=cerebrium_version,
    )

    if update_config_from_files(
        config=cerebrium_config, warning_message=False, archive_files=True
    ):
        archive_file(config_file)
        # write the updated config to the config file
        utils.tomls.save_config_to_toml_file(cerebrium_config, config_file)

    print("🚀 Cerebrium Cortex config updated successfully!")
