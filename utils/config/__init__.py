"""Configuration management: RunConfig schema, CLI parser, and yaml archive.

:class:`ConfigParser` derives the CLI from the RunConfig dataclass tree and
returns a typed RunConfig instance; :mod:`archive` round-trips it to yaml.
EarlyStopping (algorithm) and the runtime containers (TrainingConfig/...) live
in utils.training, not here.
"""

from .archive import (
    load_run_config_archive,
    load_run_metadata,
    save_run_config_archive,
)
from .config_parser import (
    ConfigParser,
    build_node,
    expand_short_flags,
    parse_run_archive,
    peek_flag_value,
    reject_model_flags,
    require_dataset,
)
from .data_config import (
    DataLoaderConfig,
    create_optimized_dataloader,
)
from .run_config import (
    CompileConfig,
    DownloadConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    GeneralConfig,
    ModelConfig,
    ProcessConfig,
    RunConfig,
    RunDataConfig,
    config_to_dict,
)

__all__ = [
    "CompileConfig",
    "ConfigParser",
    "DataLoaderConfig",
    "DownloadConfig",
    "EarlyStoppingConfig",
    "ExperimentConfig",
    "GeneralConfig",
    "ModelConfig",
    "ProcessConfig",
    "RunConfig",
    "RunDataConfig",
    "build_node",
    "config_to_dict",
    "create_optimized_dataloader",
    "expand_short_flags",
    "load_run_config_archive",
    "load_run_metadata",
    "parse_run_archive",
    "peek_flag_value",
    "reject_model_flags",
    "require_dataset",
    "save_run_config_archive",
]
