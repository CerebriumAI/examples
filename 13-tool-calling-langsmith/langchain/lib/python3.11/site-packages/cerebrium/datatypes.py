import enum
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict, Union, Optional

from cerebrium import constants

PythonVersionType = Literal["3.9", "3.10", "3.11"]
LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
ENV: str = os.getenv("ENV", "prod")

task: Literal["deploy", "build", "serve"] = "deploy"

SERVE_SESSION_CACHE_FILE = os.path.join(
    os.path.expanduser("~/.cerebrium"), "serve_session.json"
)

CerebriumDeploymentType = Literal["deploy", "run", "serve"]
RequirementsType = Union[Dict[str, str], List[str]]


class HttpMethod(enum.Enum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"


class PythonVersion(enum.Enum):
    PYTHON_3_9 = "3.9"
    PYTHON_3_10 = "3.10"
    PYTHON_3_11 = "3.11"


class OperationType(enum.Enum):
    DEPLOY = "deploy"
    BUILD = "build"
    RUN = "runner"


class FileData(TypedDict):
    fileName: str
    hash: str
    dateModified: float
    size: int


class ProviderOption:
    options = {
        "AWS": {"region": {"us-east-1": "US_EAST_1"}},
        "COREWEAVE": {"region": {"us-east-1": "LGA1", "us-central-1": "LAS1"}},
    }

    def __init__(self, name: str):
        if name.upper() not in self.options:
            raise ValueError(
                f"{name} is not a valid provider option. Available options are: AWS, COREWEAVE"
            )
        self.name = name
        self.regions = self.options[name.upper()]["region"]

    def get(self, name: str):
        if name.upper() == "AWS":
            return self.options["AWS"]
        elif name.upper() == "COREWEAVE":
            return self.options["COREWEAVE"]
        else:
            raise ValueError(
                f"{name} is not a valid provider option. Available options are: AWS, COREWEAVE"
            )


##These are hardware options from compute providers
@dataclass
class Hardware:
    def __init__(
        self,
        name: str,
        VRAM: int,
        gpu_model: str,
        max_memory: float = 128.0,
        max_cpu: int = 36,
        max_gpu_count: int = constants.MAX_GPU_COUNT,
        has_nvlink: bool = False,
        provider_names: List[str] = [],
    ):
        self.name = name
        self.gpu_model = gpu_model
        self.max_memory = max_memory
        self.max_cpu = max_cpu
        self.max_gpu_count = max_gpu_count
        self.VRAM = VRAM
        self.has_nvlink = has_nvlink
        self.provider_names = [p.lower() for p in provider_names]

        # Names for providers and hardware are entered in order of priority
        self.providers = [ProviderOption(name) for name in provider_names]
        self.region_names = [
            region for provider in self.providers for region in provider.regions
        ]

    def get_provider(self, name: str) -> ProviderOption:
        """Check if there's a provider with the given name. Return if there is. Else, raise a value error"""
        for provider in self.providers:
            if provider.name == name:
                return provider
        raise ValueError(
            f"{name} is not a valid provider option. Available options are: {self.provider_names}"
        )

    def validate(self, cpu: int, memory: float, gpu_count: int) -> str:
        if not all(
            isinstance(i, (int, float)) and i >= 0  # type: ignore
            # Unnecessary isinstance call; "int | float" is always an instance of "int | float"
            # But we need to ensure that the type because of user inputs
            for i in [cpu, memory, gpu_count]
        ):
            raise TypeError("CPU, memory, and GPU count must be positive numbers.")

        message = ""
        if cpu > self.max_cpu:
            message += f"CPU must be at most {self.max_cpu} for {self.name}.\n"
        if cpu < constants.MIN_CPU:
            message += f"CPU must be at least {constants.MIN_CPU} for {self.name}.\n"
        if memory > self.max_memory:
            message += f"Memory must be at most {self.max_memory} GB for {self.name}.\n"
        if memory < constants.MIN_MEMORY:
            message += (
                f"Memory must be at least {constants.MIN_MEMORY} GB for {self.name}.\n"
            )
        if gpu_count > self.max_gpu_count:
            message += f"Number of GPUs must be at most {self.max_gpu_count} for {self.name}.\n"
        if gpu_count < 1:
            message += f"Number of GPUs must be at least 1 for {self.name}.\n"

        if self.name == "CPU":
            # Memory is dependent on the number of CPUs.
            # Memory must be at most 4 times the number of CPUs
            if memory > 4 * cpu:
                message += "Memory must be at most 4 times the number of CPUs for CPU based deployments.\n"

        return message

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, sort_keys=False)


class HardwareOptions:
    # NOTE: Names for providers and hardware are entered in order of priority
    CPU: Hardware = Hardware(
        name="CPU",
        max_memory=128.0,
        max_cpu=36,
        VRAM=0,
        gpu_model="",
        provider_names=["coreweave", "aws"],
    )
    GPU: Hardware = Hardware(
        name="TURING_4000",
        max_memory=256.0,
        max_cpu=48,
        VRAM=8,
        gpu_model="Quadro RTX 4000",
        provider_names=["coreweave"],
    )
    TURING_4000: Hardware = Hardware(
        name="TURING_4000",
        max_memory=256.0,
        max_cpu=48,
        VRAM=8,
        gpu_model="Quadro RTX 4000",
        provider_names=["coreweave"],
    )
    TURING_5000: Hardware = Hardware(
        name="TURING_5000",
        max_memory=256.0,
        max_cpu=48,
        VRAM=8,
        gpu_model="RTX 5000",
        provider_names=["coreweave"],
    )
    AMPERE_A4000: Hardware = Hardware(
        name="AMPERE_A4000",
        max_memory=256.0,
        max_cpu=48,
        VRAM=16,
        gpu_model="RTX A4000",
        provider_names=["coreweave"],
    )
    AMPERE_A5000: Hardware = Hardware(
        name="AMPERE_A5000",
        max_memory=256.0,
        max_cpu=48,
        VRAM=24,
        gpu_model="RTX A5000",
        provider_names=["coreweave"],
    )

    AMPERE_A6000: Hardware = Hardware(
        name="AMPERE_A6000",
        max_memory=256.0,
        max_cpu=48,
        VRAM=48,
        gpu_model="RTX A6000",
        provider_names=["coreweave"],
    )
    AMPERE_A100: Hardware = Hardware(
        name="AMPERE_A100",
        max_memory=256.0,
        max_cpu=48,
        VRAM=80,
        has_nvlink=True,
        gpu_model="A100",
        provider_names=["coreweave", "aws"],
    )
    AMPERE_A100_40GB: Hardware = Hardware(
        name="AMPERE_A100_40GB",
        max_memory=256.0,
        max_cpu=48,
        VRAM=40,
        has_nvlink=True,
        gpu_model="A100 40GB",
        provider_names=["coreweave", "aws"],
    )
    AMPERE_A10: Hardware = Hardware(
        name="AMPERE_A10",
        max_memory=124.0,
        max_cpu=30,
        VRAM=24,
        has_nvlink=False,
        gpu_model="A10",
        provider_names=["aws"],
    )
    ADA_L4: Hardware = Hardware(
        name="ADA_L4",
        max_memory=16.0,
        max_cpu=4,
        VRAM=24,
        has_nvlink=False,
        gpu_model="L4",
        provider_names=["aws"],
    )

    @classmethod
    def available_hardware(cls):
        return list(cls.__annotations__.keys())


class CerebriumScaling:
    def __init__(
        self,
        min_replicas: int = constants.DEFAULT_MIN_REPLICAS,
        max_replicas: Union[int, None] = constants.DEFAULT_MAX_REPLICAS,
        cooldown: int = constants.DEFAULT_COOLDOWN,
    ):
        self.validate(min_replicas, max_replicas, cooldown)
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.cooldown = cooldown

    def validate(
        self, min_replicas: int, max_replicas: Union[int, None], cooldown: int
    ):
        if not all(x >= 0 for x in [min_replicas, cooldown]):
            raise ValueError(
                "Min replicas, cooldown, and max replicas must be positive integers"
            )
        if max_replicas is None:
            max_replicas = constants.DEFAULT_MAX_REPLICAS
        if max_replicas < 1:
            raise ValueError("Max replicas must be at least 1.")
        if min_replicas > max_replicas:
            raise ValueError("Max replicas must be larger than min replicas.")

    def __str__(self):
        return json.dumps(self.__to_dict__(), indent=4, sort_keys=False)

    def __to_dict__(self):
        return self.__dict__

    def get(self, key: str, default: Any = None):
        return self.__dict__.get(key, default)


class CerebriumBuild:
    def __init__(
        self,
        predict_data: Union[str, None] = None,
        force_rebuild: bool = False,
        hide_public_endpoint: bool = False,
        disable_animation: bool = False,
        disable_build_logs: bool = False,
        disable_syntax_check: bool = False,
        disable_predict: bool = False,
        disable_confirmation: bool = False,
        init_debug: bool = False,
        log_level: LogLevelType = "INFO",
        shell_commands: List[str] = [],
    ):
        self.predict_data = predict_data
        self.force_rebuild = force_rebuild
        self.hide_public_endpoint = hide_public_endpoint
        self.disable_animation = disable_animation
        self.disable_build_logs = disable_build_logs
        self.disable_syntax_check = disable_syntax_check
        self.disable_predict = disable_predict
        self.init_debug = init_debug
        self.log_level = log_level
        self.disable_confirmation = disable_confirmation
        self.shell_commands = shell_commands

    def __str__(self):
        return json.dumps(self.__to_dict__(), indent=4, sort_keys=False)

    def __to_dict__(self):
        return self.__dict__

    def get(self, key: str, default: Any = None):
        return self.__dict__.get(key, default)


class CerebriumDeployment:
    def __init__(
        self,
        name: str = "my-model",
        python_version: PythonVersionType = "3.10",
        include: str = constants.DEFAULT_INCLUDE,
        exclude: str = constants.DEFAULT_EXCLUDE,
        cuda_version: str = constants.DEFAULT_CUDA_VERSION,
    ):
        self.name = self.validate_name(name)
        self.python_version = python_version
        self.include = include
        self.exclude = exclude
        self.cuda_version = str(cuda_version)  # Avoids accidental int or float

    def validate_name(self, name: str, env: str = "prod") -> str:
        if not name:
            raise ValueError("No name provided.")
        max_length = 32 if env == "prod" else 63 - 28 - 2 * len(env)
        if len(name) > max_length:
            raise ValueError(f"Name must be at most {max_length} characters.")
        if not re.match("^[a-z0-9\\-]*$", name):
            raise ValueError(
                "Name must only contain lower case letters, numbers, and dashes."
            )
        return name

    def __str__(self):
        return json.dumps(self.__to_dict__(), indent=4, sort_keys=False)

    def __to_dict__(self):
        return self.__dict__

    def get(self, key: str, default: Any = None):
        return self.__dict__.get(key, default)


class CerebriumHardware:
    def __init__(
        self,
        gpu: str = constants.DEFAULT_GPU_SELECTION,
        cpu: int = constants.DEFAULT_CPU,
        memory: float = constants.DEFAULT_MEMORY,
        gpu_count: int = constants.DEFAULT_GPU_COUNT,
        provider: Optional[str] = "",
        region: Optional[str] = "",
    ):
        if gpu.upper() == "NONE":
            # use CPU only
            gpu = "CPU"
        if gpu.upper() == "CPU":
            gpu_count = 0

        try:
            gpuOption: Hardware = getattr(HardwareOptions, gpu.upper())
        except AttributeError:
            available_gpus = ", ".join(HardwareOptions.available_hardware())
            raise ValueError(
                f"{gpu.upper()} is not a valid GPU option. Available options are: {available_gpus}"
            )
        provider = provider if provider else gpuOption.provider_names[0]
        region = region if region else gpuOption.region_names[0]
        self.validate(gpuOption, cpu, memory, gpu_count, provider, region)

        # NOTE: region_name is the real name of the region for the backend
        self.region_name = gpuOption.get_provider(provider).regions[region]
        self.region = region
        self.provider = provider
        self.gpu = gpu.upper()
        self.cpu = cpu
        self.memory = memory
        self.gpu_count = gpu_count

    def validate(
        self,
        gpu: Hardware,
        cpu: int,
        memory: float,
        gpu_count: int,
        provider: Optional[str],
        region: Optional[str],
    ):
        message: List[str] = []
        if not all(x >= 0 for x in [cpu, memory]) or (
            gpu.name not in ["CPU", None] and gpu_count <= 0
        ):
            message.append(
                "CPU, memory must be positive values and GPU count must be positive unless GPU is 'CPU' or None."
            )
        if cpu < constants.MIN_CPU:
            message.append(f"CPU must be at least {constants.MIN_CPU}.")

        if memory < constants.MIN_MEMORY:
            message.append(f"Memory must be at least {constants.MIN_MEMORY} GB.")

        if cpu > gpu.max_cpu:
            message.append(
                f"CPU must be <= {gpu.max_cpu} for the selected hardware option."
            )

        if gpu_count > constants.MAX_GPU_COUNT:
            message.append(f"GPU count must be <= {constants.MAX_GPU_COUNT}.")

        if memory > gpu.max_memory:
            message.append(
                f"Memory must be <= {gpu.max_memory} GB for the selected hardware option."
            )
        if not provider or provider not in gpu.provider_names:
            message.append(
                f"{provider} is not a valid provider for {gpu.name}. Please use one of {gpu.provider_names}."
            )
        elif region not in gpu.get_provider(provider).regions:
            message.append(
                f"{region} is not a valid region for {gpu.name} with on {provider}. Please use one of {list(gpu.get_provider(provider).regions.keys())}."
            )
        if len(message) > 0:
            raise ValueError("\n".join(message))

    def __str__(self):
        return json.dumps(self.__to_dict__(), indent=4, sort_keys=False)

    def __to_dict__(self):
        return self.__dict__

    def get(self, key: str, default: Any = None):
        return self.__dict__.get(key, default)


class CerebriumDependencies:
    def __init__(
        self,
        pip: Dict[str, str] = {},
        conda: Dict[str, str] = {},
        apt: Dict[str, str] = {},
    ):
        self.pip = pip
        self.conda = conda
        self.apt = apt

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=False)

    def to_dict(self):
        return self.__dict__


class CerebriumConfig:
    def __init__(
        self,
        scaling: CerebriumScaling = CerebriumScaling(),
        build: CerebriumBuild = CerebriumBuild(),
        deployment: CerebriumDeployment = CerebriumDeployment(),
        hardware: CerebriumHardware = CerebriumHardware(),
        dependencies: CerebriumDependencies = CerebriumDependencies(),
        local_files: List[FileData] = [],
        cerebrium_version: str = "",
        partial_upload: bool = False,
        file_list: List[str] = [],
    ):
        self.scaling = scaling
        self.build = build
        self.deployment = deployment
        self.hardware = hardware
        self.dependencies = dependencies
        self.local_files = local_files
        self.cerebrium_version = cerebrium_version
        self.partial_upload = partial_upload
        self.file_list = file_list

    def get(self, key: str, default: Any = None):
        return self.__dict__.get(key, default)

    def __str__(self):
        # Convert to dict. All the nested classes need to be converted to dict
        # before converting to string
        dictified = self.to_dict()
        return json.dumps(dictified, indent=4, sort_keys=False)

    def to_dict(self):
        dictified = self.__dict__.copy()
        for key, value in dictified.items():
            if hasattr(value, "__dict__"):
                dictified[key] = value.__dict__
        return dictified

    def __to_json__(self):
        return json.dumps(self.to_dict())  # type: ignore
