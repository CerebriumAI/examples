import hashlib
import os
import zipfile
from typing import Dict, List, Union

from cerebrium.constants import INTERNAL_FILES

FileHashesType = Dict[str, str]  # Dict[file-path, hash]


def content_hash(
    files: Union[List[str], str],
    strings: Union[List[Union[str, None]], str, None] = None,
) -> str:
    """
    Hash the content of each file, avoiding metadata.
    """

    files = files if isinstance(files, list) else [files]
    h = hashlib.sha256()
    if files:
        for file in files:
            if os.path.exists(file):
                with open(file, "rb") as f:
                    h.update(f.read())
            else:
                return "FILE_DOESNT_EXIST"

    if not isinstance(strings, list):
        strings = [strings]
    for string in strings:
        if isinstance(string, str):
            h.update(string.encode())
    if files or strings:
        return h.hexdigest()
    return "NO_FILES"


def check_deployment_size(files: Union[str, List[str]], max_size_mb: int = 100):
    """
    Check if the sum of all files is less than max_size MB
    """
    files = files if isinstance(files, list) else [files]
    total_size = 0
    for file in files:
        if os.path.exists(file):
            total_size += os.path.getsize(file)

    return total_size > max_size_mb * 1024 * 1024


def get_all_file_hashes() -> FileHashesType:
    """Get the hashes of all the files in the current directory"""
    file_hashes: FileHashesType = {}
    for root, _, files in os.walk("."):
        for file in files:
            file_path = os.path.join(root, file)
            file_hashes[file_path] = content_hash(file_path)
    return file_hashes


def create_zip_file(
    zip_file_name: str,
    file_list: List[str],
    temp_dir: str,
):
    """
    Create a zip file with the given files

    Args:
        zip_file_name (str): Name of the zip file to be created
        file_list (List[str]): List of files to be added to the zip file
        temp_dir (str): Temporary directory to store the zip file
    """

    tmp_dir_files = os.listdir(temp_dir)

    for f in INTERNAL_FILES:
        if f in tmp_dir_files and f in file_list:
            file_list.remove(f)

    with zipfile.ZipFile(zip_file_name, "w") as zip_file:
        print("🗂️  Zipping files...")
        for f in file_list:
            if os.path.isfile(f):
                zip_file.write(f)

        for f in INTERNAL_FILES:
            if f in tmp_dir_files:
                zip_file.write(os.path.join(temp_dir, f), arcname=os.path.basename(f))
