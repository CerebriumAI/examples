import datetime
import hashlib
import io
import json
import os
import requests
from typing import Dict, List, Literal, Optional, TypedDict
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from concurrent.futures import ThreadPoolExecutor, as_completed

from cerebrium import datatypes
from cerebrium.utils.logging import cerebrium_log
from cerebrium.utils.requirements import requirements_to_file, shell_commands_to_file

debug = os.environ.get("LOG_LEVEL", "INFO") == "DEBUG"


class UploadURLsResponse(TypedDict):
    uploadUrls: Dict[str, str]
    deleteKeys: List[str]
    markerFile: str


def get_md5(file_path: str, max_size_mb: int = 100) -> str:
    """Return MD5 hash of a file if smaller than threshold. Else, hash the os.stat info"""
    hasher = hashlib.md5()
    if os.stat(file_path).st_size > max_size_mb * 1024 * 1024:
        file_stats = os.stat(file_path)
        large_file_info = f"{file_path}-{ file_stats.st_mtime}-{file_stats.st_size}"
        hasher.update(str(large_file_info).encode())
        return hasher.hexdigest()

    with open(file_path, "rb") as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()


def gather_hashes(file_list: List[str], base_dir: str = "") -> List[datatypes.FileData]:
    """Gather the MD5 hashes of the local files including subdirectories."""
    local_files_payload: List[datatypes.FileData] = []

    for file in file_list:
        if file.startswith("./"):
            file = file[2:]
        if base_dir and file.startswith(base_dir):
            file_name = os.path.relpath(file, base_dir)
        else:
            file_name = file
        if os.path.islink(file):
            file = os.readlink(file)
        if os.path.isfile(file):
            file_hash = get_md5(file)
            local_files_payload.append(
                {
                    "fileName": file_name,
                    "hash": file_hash,
                    "dateModified": os.stat(file).st_mtime,
                    "size": os.stat(file).st_size,
                }
            )

    return local_files_payload


def upload_file(
    upload_url: str, file_name: str, file_path: str, pbar: Optional[tqdm]
) -> int:
    """Function to upload a single file."""

    try:
        if os.stat(file_path).st_size == 0:
            upload_response = requests.put(upload_url, data=b"")
        else:
            with open(file_path, "rb") as file:
                wrapped_file = (
                    CallbackIOWrapper(pbar.update, file, "read") if pbar else file
                )
                upload_response = requests.put(
                    upload_url,
                    data=wrapped_file,  # type: ignore
                    timeout=60,
                    stream=True,
                )
        if upload_response.status_code != 200:
            raise Exception(
                f"Failed to upload {file_name}. Status code: {upload_response.status_code}"
            )
        return 1
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        return 0


def upload_files_to_s3(
    upload_urls: Dict[str, str],
    base_dir: str = "",
    workers: int = 5,
    quiet: bool = False,
) -> int:
    try:
        workers = max(os.cpu_count() or workers, 10)
    except (KeyError, TypeError):
        pass
    file_keys = list(upload_urls.keys())
    if len(file_keys) == 0:
        return 0

    workers = min(workers, len(file_keys) or 1)  # don't want more workers than files
    working_dir = base_dir or os.getcwd()
    if working_dir[-1] != "/":
        working_dir = working_dir + "/"

    for path in file_keys:
        if not quiet:
            print(f"➕ Adding {path}")
    print(f"Uploading {len(upload_urls)} files...")
    # Get the working paths in the tempfile dir.
    # Necessary because the file paths in the upload_urls are relative to the working directory
    # Skipping "upload.complete" files - this is uploaded after all other files
    working_paths = [
        os.path.join(working_dir, file)
        for file in file_keys
        if file in upload_urls and file != "upload.complete"
    ]

    # Need to follow links so that we stat the actual file, not the symlink.
    real_paths = {
        path.replace(working_dir, ""): (
            path if not os.path.islink(path) else os.readlink(path)
        )
        for path in working_paths
    }

    # Calculate total size of all files
    total_size = sum(
        os.path.getsize(path) for path in real_paths if os.path.isfile(path)
    )

    if quiet:
        uploaded_count = _parallel_upload(workers, upload_urls, real_paths, None)
    else:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Upload Progress",
        ) as pbar:
            uploaded_count = _parallel_upload(workers, upload_urls, real_paths, pbar)

    return uploaded_count


def _parallel_upload(
    workers: int,
    upload_urls: Dict[str, str],
    real_paths: Dict[str, str],
    pbar: Optional[tqdm],
) -> int:
    """Upload files in parallel for faster uploads"""
    uploaded_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                upload_file,
                upload_urls[key],
                key,
                real_path,
                pbar,
            )
            for key, real_path in real_paths.items()
            if key in upload_urls
        ]
        for future in as_completed(futures):
            uploaded_count += future.result()

    return uploaded_count


def upload_marker_file_and_delete(
    url: str,
    uploaded_count: int,
    build_id: str,
    files_and_hashes: List[datatypes.FileData],
) -> None:
    """Upload the marker file with JSON content without actually writing anything to disk."""

    # Construct the marker file content
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    marker_content = {
        "date": current_date,
        "filesUploaded": uploaded_count,
        "buildId": build_id,
        "fileList": files_and_hashes,
    }

    # Convert the dictionary to a JSON formatted string
    json_content = json.dumps(marker_content)

    # Simulate the marker file in memory
    marker_file_content = json_content.encode()  # Convert to bytes
    marker_file = io.BytesIO(marker_file_content)

    upload_response = requests.put(url, data=marker_file)
    if upload_response.status_code != 200:
        marker_file_name = "upload.complete"

        raise Exception(
            f"Failed to upload {marker_file_name}. Status code: {upload_response.status_code}"
        )
    print("Upload complete.")


def make_cortex_util_files(
    working_dir: str,
    config: datatypes.CerebriumConfig,
    source: Literal["serve", "cortex"] = "cortex",
):
    # Remove requirements.txt, pkglist.txt, conda_pkglist.txt from file_list if they exist. Will be added from config
    files_to_remove = [
        "requirements.txt",
        "conda_pkglist.txt",
        "pkglist.txt",
        "./requirements.txt",
        "./conda_pkglist.txt",
        "./pkglist.txt",
    ]
    config.file_list = [f for f in config.file_list if f not in files_to_remove]

    # write a predict config file containing the prediction parameters
    predict_file = os.path.join(
        working_dir, "_cerebrium_predict.json"
    )  # use a file to avoid storing large files in the model objects in ddb
    if (
        config.build.predict_data is not None
        and not os.path.exists("_cerebrium_predict.json")
        and not source == "serve"
    ):
        with open(predict_file, "w") as f:
            f.write(
                config.build.predict_data
            )  # predict data has been validated as a json string previously

    # Create files temporarily for upload
    requirements_files = [
        ("requirements.txt", config.dependencies.pip),
        ("pkglist.txt", config.dependencies.apt),
        ("conda_pkglist.txt", config.dependencies.conda),
        ("shell_commands.sh", config.build.shell_commands),
    ]

    for file_name, reqs in requirements_files:
        if reqs:
            if file_name == "shell_commands.sh":
                shell_commands_to_file(reqs, os.path.join(working_dir, file_name))
            else:
                requirements_to_file(
                    reqs,
                    os.path.join(working_dir, file_name),
                    is_conda=file_name == "conda_pkglist.txt",
                )
