import copy
import os
from typing import Any, Dict

import toml
import typer
from termcolor import colored

from cerebrium import constants, datatypes
from cerebrium.utils.logging import cerebrium_log
from cerebrium.utils.requirements import update_from_file


def legacy_to_toml_structure(
    name: str,
    legacy_params: Dict[str, Any],
    config_file: str,
    save_to_file: bool = True,
    pip: Dict[str, str] = {},
    apt: Dict[str, str] = {},
    conda: Dict[str, str] = {},
    disable_confirmation: bool = False,
    overwrite: bool = False,
) -> datatypes.CerebriumConfig:
    # Tomls have the following format so they're less intimidating:
    # [cerebrium.hardware]
    # {all the hardware params like cpu, memory, etc}
    # [cerebrium.scaling]
    # {all the scaling params. Min/max replicas, cooldown, etc}
    # [cerebrium.deployment]
    # {all the deployment params. Python version, include, exclude, etc}
    # [cerebrium.requirements] (optional pip requirements)
    # {all the pip requirements}
    # [cerebrium.conda_requirements] (optional conda requirements)
    # {all the conda requirements}
    # [cerebrium.pkglist] (optional pkglist)

    """Upgrade legacy config file to use a more intuitive toml format"""
    legacy = ".yaml" in config_file or ".yml" in config_file

    upgrade_to_toml = False
    if legacy and not disable_confirmation:
        upgrade_prompt = colored(
            "Upgrade legacy config to toml?",
            "yellow",
        )
        if typer.confirm(upgrade_prompt):
            upgrade_to_toml = True
    dir_path = os.path.dirname(os.path.realpath(config_file))
    legacy_config: Dict[str, Any] = legacy_params or {}

    # Hardware
    hardware = datatypes.CerebriumHardware(
        gpu=legacy_config.get(
            "hardware" if legacy else "gpu", constants.DEFAULT_GPU_SELECTION
        ),
        cpu=legacy_config.get("cpu", constants.DEFAULT_CPU),
        memory=legacy_config.get("memory", constants.DEFAULT_MEMORY),
        gpu_count=legacy_config.get("gpu_count", constants.DEFAULT_GPU_COUNT),
    )

    deployment = datatypes.CerebriumDeployment(
        name=legacy_config.get("name", os.path.basename(os.getcwd())),
        python_version=legacy_config.get(
            "python_version", constants.DEFAULT_PYTHON_VERSION
        ),
        include=legacy_config.get("include", constants.DEFAULT_INCLUDE),
        exclude=legacy_config.get("exclude", constants.DEFAULT_EXCLUDE),
    )

    default_predict_data = '{"prompt": "Here is some example predict data for your config.yaml which will be used to test your predict function on build."}'
    build = datatypes.CerebriumBuild(
        predict_data=legacy_config.get("predict_data", default_predict_data),
        disable_predict=legacy_config.get("disable_predict", False)
        or legacy_config.get("disable_predict_data", False),
        disable_build_logs=legacy_config.get("disable_build_logs", False),
        disable_animation=legacy_config.get("disable_animation", False),
        force_rebuild=legacy_config.get("force_rebuild", False),
        disable_confirmation=legacy_config.get("disable_confirmation", False),
        hide_public_endpoint=legacy_config.get("hide_public_endpoint", False),
        disable_syntax_check=legacy_config.get("disable_syntax_check", False),
        shell_commands=legacy_config.get("shell_commands") or [],
    )

    # Scaling
    scaling = datatypes.CerebriumScaling(
        min_replicas=legacy_config.get("min_replicas", constants.DEFAULT_MIN_REPLICAS),
        max_replicas=legacy_config.get("max_replicas", constants.DEFAULT_MAX_REPLICAS),
        cooldown=legacy_config.get("cooldown", constants.DEFAULT_COOLDOWN),
    )

    # Requirements
    dependencies = {"pip": pip, "conda": conda, "apt": apt}
    if (
        os.path.exists(os.path.join(dir_path, "requirements.txt"))
        and os.stat(os.path.join(dir_path, "requirements.txt")).st_size != 0
        and legacy
    ):
        dependencies = update_from_file(
            file="requirements.txt",
            toml_requirements=dependencies,
            key="pip",
            confirm=(not legacy) and (not disable_confirmation),
        )
    else:
        dependencies["pip"] = pip
    if (
        os.path.exists(os.path.join(dir_path, "pkglist.txt"))
        and os.stat(os.path.join(dir_path, "pkglist.txt")).st_size != 0
        and legacy
    ):
        dependencies = update_from_file(
            "pkglist.txt",
            dependencies,
            "apt",
            confirm=(not legacy) and (not disable_confirmation),
        )

    else:
        dependencies["apt"] = apt  # no versions for apt. So we just add the list

    if (
        os.path.exists(os.path.join(dir_path, "conda_pkglist.txt"))
        and os.stat(os.path.join(dir_path, "conda_pkglist.txt")).st_size != 0
        and legacy
    ):
        dependencies = update_from_file(
            "conda_pkglist.txt",
            dependencies,
            "conda",
            confirm=(not legacy) and (not disable_confirmation),
        )
    else:
        dependencies["conda"] = conda

    config = datatypes.CerebriumConfig(
        hardware=hardware,
        deployment=deployment,
        scaling=scaling,
        build=build,
        dependencies=datatypes.CerebriumDependencies(**dependencies),
    )

    if name:
        config.deployment.name = name
    elif not config.deployment.name:
        config.deployment.name = os.path.basename(dir_path)

    if save_to_file or upgrade_to_toml:
        config_file = os.path.join(dir_path, "cerebrium.toml")
        save_config_to_toml_file(config, config_file, overwrite=overwrite)
        # move old config file to config.yaml.legacy
        if legacy:
            cwd = os.getcwd()
            if os.path.exists(os.path.join(cwd, "config.yaml")):
                os.rename(
                    os.path.join(cwd, "config.yaml"),
                    os.path.join(cwd, "config.yaml.legacy"),
                )
            elif os.path.exists(config_file):
                os.rename(config_file, config_file + ".legacy")

    return config


def save_config_to_toml_file(
    config: datatypes.CerebriumConfig,
    file: str,
    overwrite: bool = False,
    quiet: bool = False,
):
    # Write to file
    config_dict = copy.deepcopy(config).to_dict()
    if "local_files" in config_dict:
        config_dict.pop("local_files")
    if "cerebrium_version" in config_dict:
        config_dict.pop("cerebrium_version")
    if "api_key" in config_dict:
        config_dict.pop("api_key")
    if "partial_upload" in config_dict:
        config_dict.pop("partial_upload")

    if "init_debug" in config_dict.get("build", {}):
        config_dict["build"].pop("init_debug")

    if "file_list" in config_dict:
        config_dict.pop("file_list")

    for k, v in config_dict.items():
        if hasattr(v, "to_dict"):
            config_dict[k] = v.to_dict()

    # sort the keys
    keys = list(config_dict.keys())
    keys.sort()
    config_dict = {k: config_dict[k] for k in keys}
    # make sure "requirements" is last key
    config_dict["dependencies"] = config_dict.pop("dependencies")
    if "region_name" in config_dict.get("hardware", {}):
        # region_name is the real name of the region for the backendregion is the name of the region for the user
        config_dict["hardware"].pop("region_name")

    config_dict = {"cerebrium": config_dict}

    if os.path.splitext(file)[1] != ".toml":
        file = os.path.splitext(file)[0] + ".toml"

    if os.path.exists(file):
        if not overwrite and not quiet:
            cerebrium_log(
                level="WARNING",
                message="cerebrium.toml already exists. Not writing.",
                prefix_seperator="\t",
            )
            return None

    with open(file, "w") as f:
        toml.dump(config_dict, f)

    comment = (
        "# This file was automatically generated by Cerebrium as a "
        "starting point for your project. \n"
        "# You can edit it as you wish.\n"
        "# If you would like to learn more about your Cerebrium config, "
        "please visit https://docs.cerebrium.ai/cerebrium/environments/config-files#config-file-example"
    )

    # prepend comment to file
    with open(file, "r") as f:
        content = f.read()
    with open(file, "w") as f:
        f.write(f"{comment}\n\n{content}")
