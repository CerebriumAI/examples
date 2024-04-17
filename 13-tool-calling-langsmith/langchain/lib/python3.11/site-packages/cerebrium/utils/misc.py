import fnmatch
import inspect
import os
from typing import Any, Callable, Dict, List, Union

import toml
import yaml

from cerebrium import datatypes
from cerebrium.utils.files import content_hash
from cerebrium.utils.logging import cerebrium_log

env = os.getenv("ENV", "prod")
NullishDictType = Dict[str, Any]


def shell_commands_to_file(
    shell_commands: datatypes.RequirementsType, shell_file: str
) -> None:
    """Takes requirements from a TOML file and writes the shell commands to a shell script file"""

    shell_file_directory = os.path.dirname(shell_file)
    os.makedirs(shell_file_directory, exist_ok=True)

    with open(shell_file, "w") as f:
        f.write("set -e \n")
        for command in shell_commands:
            f.writelines(command + "\n")


def determine_includes(include: str, exclude: str, quiet: bool = False):
    include_set = include.strip("[]").split(",")
    include_set.extend(
        [
            "./main.py",
        ]
    )

    include_set = [i.strip() for i in include_set]
    include_set = set(map(ensure_pattern_format, include_set))

    exclude_set = exclude.strip("[]").split(",")
    exclude_set = [e.strip() for e in exclude_set]
    exclude_set = set(map(ensure_pattern_format, exclude_set))

    file_list: List[str] = []
    for root, _, files in os.walk("./"):
        for file in files:
            full_path = os.path.join(root, file)
            if any(
                fnmatch.fnmatch(full_path, pattern) for pattern in include_set
            ) and not any(
                fnmatch.fnmatch(full_path, pattern) for pattern in exclude_set
            ):
                file_list.append(full_path)
    return file_list


def get_current_project_context():
    """
    Get the current project context and project name
    """
    config_path = os.path.expanduser("~/.cerebrium/config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            key_name = ""
            if env != "prod":
                key_name = f"{env}-"
            if config.get(f"{key_name}project"):
                return config.get(f"{key_name}project")
    print("No current project context found.")
    return None, None


def ensure_pattern_format(pattern: str):
    if not pattern:
        return pattern
    if not pattern.startswith("./"):
        pattern = f"./{pattern}"
    elif pattern.startswith("/"):
        cerebrium_log(
            prefix="ValueError",
            level="ERROR",
            message="Pattern cannot start with a forward slash. Please use a relative path.",
        )
    if pattern.endswith("/"):
        pattern = f"{pattern}*"
    elif os.path.isdir(pattern) and not pattern.endswith("/"):
        pattern = f"{pattern}/*"
    return pattern


def update_with_defaults(
    params: Dict[str, Union[str, None]], defaults: Dict[str, Union[str, None]]
):
    for key, val in defaults.items():
        if params.get(key) is None or params.get(key) == "":
            params[key] = val

    return params


def assign_param(
    param_dict: Dict[str, Any],
    key: str,
    new_value: Any,
    default_value: Union[str, None] = None,
):
    valid = (
        (isinstance(new_value, bool) and new_value is False)
        or bool(new_value)
        or isinstance(new_value, (int, float))
    )
    param_dict[key] = new_value if valid else param_dict.get(key, default_value)
    return param_dict


def remove_null_values(param_dict: NullishDictType):
    new_dict: NullishDictType = {}
    for key, val in param_dict.items():
        if isinstance(val, dict):
            val = remove_null_values(val)  # type: ignore
        if val is not None:
            if not (isinstance(val, str) and val == ""):
                new_dict[key] = val
    return new_dict


def get_function_params(function: Callable[..., Any]) -> dict[str, Any]:
    """
    Get the parameters of a function
    """
    return {
        k: v.default
        for k, v in inspect.signature(function).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def merge_config_with_params(
    config_file: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Load the configuration from a TOML file and override it with any non-default parameter values provided.
    This version is enhanced to explicitly handle the nested structure of the cerebrium.toml configuration.

    Args:
    - config_file: Path to the TOML configuration file.
    - params: Keyword arguments corresponding to the deploy function parameters.


    returns: A dictionary with the merged configuration.
    """

    # Load the configuration from the TOML file
    if not os.path.exists(config_file):
        cerebrium_log(
            level="WARNING",
            message=f"Could not find {config_file}. Using default parameters and creating a new one instead.",
        )
        toml_config: Dict[str, Any] = {}
    else:
        with open(config_file, "r") as f:
            toml_config = toml.load(f)
            toml_config = toml_config["cerebrium"]

    # Define the top-level sections in the TOML configuration
    top_level_sections = {
        "build": get_function_params(datatypes.CerebriumBuild.__init__),
        "deployment": get_function_params(datatypes.CerebriumDeployment.__init__),
        "hardware": get_function_params(datatypes.CerebriumHardware.__init__),
        "scaling": get_function_params(datatypes.CerebriumScaling.__init__),
        "dependencies": get_function_params(datatypes.CerebriumDependencies.__init__),
    }

    # Iterate over each top-level section in the TOML configuration
    for section, defaults in top_level_sections.items():
        # check if the key is in the params
        for key, val in defaults.items():
            if section not in toml_config:
                toml_config[section] = defaults
            elif (
                key in params
                and (params.get(key) is not None)
                and params.get(key) != ""
            ):
                toml_config.get(section, {})[key] = params.get(key)
            elif toml_config.get(section, {}).get(key) is None:
                # Replace none with default value
                toml_config[section][key] = val
    return toml_config


def flatten_cerebrium_config_to_json(
    config: datatypes.CerebriumConfig,
) -> Dict[str, Any]:
    """
    Takes a CerebriumConfig class object and flattens it into a flat JSON-like dictionary.

    :param config: CerebriumConfig object
    :return: Flat dictionary representing the config
    """

    hashes = {
        "requirements_hash": content_hash(
            files=[], strings=str(config.dependencies.pip)
        ),
        "pkglist_hash": content_hash(files=[], strings=str(config.dependencies.apt)),
        "conda_pkglist_hash": content_hash(
            files=[], strings=str(config.dependencies.conda)
        ),
    }

    raw = config.to_dict()
    raw.pop("dependencies")

    # Need two copies as raw will change sizes.
    params = raw.copy()
    for k, v in raw.items():
        if isinstance(v, dict):
            # remove section title. eg, all params in "build" go into root dict
            section = params.pop(k)
            for k2, v2 in section.items():
                params[k2] = v2

    params.update(hashes)
    params["hardware"] = params.pop("gpu")  # so we use the same name as the backend.
    params["region"] = params.pop("region_name")
    # Remove keys with None values to clean up the dictionary
    params: Dict[str, Any] = {k: v for k, v in params.items() if v is not None}
    return params
