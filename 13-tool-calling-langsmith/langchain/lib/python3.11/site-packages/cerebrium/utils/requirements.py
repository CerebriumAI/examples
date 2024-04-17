import os
import re
from typing import Dict, List

import typer
from termcolor import colored

from cerebrium.datatypes import RequirementsType
from cerebrium.utils.logging import cerebrium_log


def parse_requirements(file: str):
    """Takes a pip requirements file or a pkglist file and returns a list of packages"""
    if not os.path.exists(file):
        cerebrium_log(
            level="ERROR",
            message=f"Could not find {file}. Please create it and try again.",
        )
    with open(file, "r") as f:
        requirements = f.read()
    requirements_list = requirements.split("\n")
    requirements_list = [r.strip() for r in requirements_list]

    # ignore comments
    requirements_list = [r for r in requirements_list if not r.startswith("#")]
    # remove empty lines
    requirements_list = [r for r in requirements_list if r != ""]

    # if there's version numbers, we return a dict of package: version
    # otherwise we return a list of packages
    requirements_dict = req_list_to_dict(requirements_list)
    return requirements_dict


def req_list_to_dict(requirements: List[str]) -> Dict[str, str]:
    """Takes a list of requirements and returns a dict of package: version"""
    requirements_dict: Dict[str, str] = {}
    if len(requirements) == 0:
        return requirements_dict
    for r in requirements:
        # find on "==" or ">=" or "<=" or "~=" or "!=" or ">" or "<"
        search = re.search(r"==|>=|<=|~=|!=|>|<", r)
        if search is None:
            package, version = r, "latest"
        else:
            idx = search.start()
            package, version = r[:idx], r[idx:]
        requirements_dict[package] = version
    return requirements_dict


def req_dict_to_str_list(
    requirements: RequirementsType, for_display: bool = False, is_conda: bool = False
) -> List[str]:
    """Takes a dict of requirements and returns a list of requirements to be written to a file"""
    reqs: List[str] = []
    # if version starts with ==, >=, <=, ~=, !=, >, <, we don't add the ==
    # find >=, <=, ~=, !=, >, <
    pattern = re.compile(r"==|>=|<=|~=|!=|>|<")
    if isinstance(requirements, list):
        requirements = req_list_to_dict(requirements)
    for package, version in requirements.items():
        if str(version).lower() == "latest" and not for_display:
            version = ""
        if pattern.search(version):
            reqs.append(f"{package}{version}\n")
        else:
            if version == "":
                reqs.append(f"{package}\n")
            else:
                version = version.strip("=")
                if version.startswith("git+"):
                    reqs.append(f"{version}\n")
                else:
                    reqs.append(f"{package}{'=' if is_conda else '=='}{version}\n")

    return reqs


def requirements_to_file(
    requirements: RequirementsType,
    file: str,
    is_conda: bool = False,
    allow_empty: bool = False,
) -> None:
    """Takes a dict/list of requirements and writes them to a file"""
    reqs = req_dict_to_str_list(requirements, is_conda=is_conda)
    if not allow_empty and len(reqs) == 0:
        return
    with open(file, "w") as f:
        f.writelines(reqs)


def shell_commands_to_file(
    shell_commands: RequirementsType,
    shell_file: str,
    allow_empty: bool = False,
) -> None:
    """Takes requirements from a TOML file and writes the shell commands to a shell script file"""
    if not allow_empty and len(shell_commands) == 0:
        return

    shell_file_directory = os.path.dirname(shell_file)
    os.makedirs(shell_file_directory, exist_ok=True)

    with open(shell_file, "w") as f:
        f.write("set -e \n")
        for command in shell_commands:
            f.writelines(command + "\n")


def update_from_file(
    file: str,
    toml_requirements: Dict[str, Dict[str, str]],
    key: str,
    confirm: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Update the requirements dictionary from a file"""
    new_requirements = parse_requirements(file)

    if new_requirements != toml_requirements.get(key):
        if confirm:
            if typer.confirm(
                colored(
                    f"Update {key} requirements in the cerebrium.toml?",
                    "yellow",
                )
            ):
                toml_requirements[key] = new_requirements
            else:
                return toml_requirements
        else:
            toml_requirements[key] = new_requirements

    return toml_requirements
