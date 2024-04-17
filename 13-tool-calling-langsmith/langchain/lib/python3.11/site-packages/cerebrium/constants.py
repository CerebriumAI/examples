import os

MAX_MEMORY = 256
MAX_GPU_COUNT = 8
MAX_CPU = 48

MIN_CPU = 1
MIN_MEMORY = 2

DEFAULT_COOLDOWN = 60
DEFAULT_CPU = 3
DEFAULT_MEMORY = 12
DEFAULT_MIN_REPLICAS = 0
DEFAULT_MAX_REPLICAS = 5
DEFAULT_GPU_SELECTION = "AMPERE_A10"
DEFAULT_PYTHON_VERSION = "3.11"
DEFAULT_GPU_COUNT = 1
DEFAULT_CUDA_VERSION = "12"
DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_INCLUDE = "[./*, main.py, cerebrium.toml]"
DEFAULT_EXCLUDE = "[./example_exclude]"

INTERNAL_FILES = [
    "requirements.txt",
    "pkglist.txt",
    "conda_pkglist.txt",
    "_cerebrium_predict.json",
    "shell_commands.sh",
]

env = os.getenv("ENV", "prod")
