"""RunConfig: the typed dataclass tree parsed by :class:`ConfigParser`.

CLI flags, ``--config`` yaml, and archived ``run_config.yaml`` all round-trip
through this schema. Framework-level knobs live in the fixed sub-configs below;
per-model hyperparameters live in :class:`ModelConfig` subclasses registered via
``@register_model_config``.

Runtime ``rc`` is a :class:`RunConfig` instance. The polymorphic ``model`` node
is the concrete subclass at schema-build time (see
:func:`build_run_config_schema`), never the base.

Field help lives in each class ``Args:`` docstring; enumerated fields use
:class:`typing.Literal` for validated choices.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration.

    Args:
        monitor: Metric to monitor.
        mode: Optimization mode ('max' for auc/acc, 'min' for rmse/loss).
        patience: Epochs with no improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
    """

    monitor: Literal["auc", "acc", "rmse", "loss", "auprc"] = "auc"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0


@dataclass
class GeneralConfig:
    """Framework-level general knobs: logging, device, seed, tracking.

    Args:
        log_dir: Directory to save logs and models (default runs/<timestamp>).
        checkpoint_path: Path to a checkpoint for resuming training.
        device: Device to use ('cuda' or 'cpu'; auto-detect when null).
        seed: Random seed for reproducibility.
        deterministic: Enable deterministic algorithms (on by default; may
            reduce speed on GPU).
        cloud_tracking: Enable cloud experiment tracking (SwanLab or W&B;
            backend chosen via the ``KT_TRACKING_BACKEND`` env var, default
            SwanLab).
        log_batch_metrics: Log per-batch loss to batch_metrics_<phase>.csv.
        skip_test: Skip test-set evaluation after training.
        cache: Enable disk cache for model data preparation.
        save_last_checkpoint: Save a full checkpoint (``last_checkpoint.pth``)
            at the end of every epoch.
        pin_memory: DataLoader host-memory pinning (null = auto: pin only
            on CUDA devices).
        progress: Progress rendering for the training loop. ``auto`` renders
            when stdout is a TTY; ``rich`` always renders; ``none`` disables
            (headless runs, in-process sweeps).
    """

    log_dir: str | None = None
    checkpoint_path: str | None = None
    device: str | None = None
    seed: int = field(default=42, metadata={"preprocess_ui": True})
    deterministic: bool = True
    cloud_tracking: bool = True
    log_batch_metrics: bool = False
    skip_test: bool = False
    cache: bool = False
    save_last_checkpoint: bool = True
    pin_memory: bool | None = None
    progress: Literal["auto", "rich", "none"] = "auto"


@dataclass
class CompileConfig:
    """``torch.compile`` execution knobs (framework-level, model-agnostic).

    Args:
        compile: Enable torch.compile for model optimization.
        compile_mode: Compilation mode.
        compile_fullgraph: Require the whole function be one capturable graph.
        compile_dynamic: Use dynamic shape tracing (null = PyTorch auto-detects).
        compile_backend: Compilation backend.
    """

    compile: bool = False
    compile_mode: Literal[
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ] = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool | None = None
    compile_backend: str = "inductor"


@dataclass
class ExperimentConfig:
    """Experiment identity used for run naming and archive lookup.

    Args:
        model_name: Registered model name (selects the trainer + ModelConfig).
    """

    model_name: str = ""


@dataclass
class RunDataConfig:
    """Dataset selection, split, sequence bounds, and sampling.

    Args:
        dataset: Dataset name.
        data_base_path: Path to the data files.
        fold: Fold index for K-fold cross-validation.
        kfold: Number of folds (>=2 to enable K-fold).
        test_ratio: Held-out test ratio.
        min_seq_len: Minimum sequence length.
        max_seq_len: Maximum sequence length.
        sample_size: Absolute sample count (null disables sampling).
        sample_ratio: Sample ratio 0.0-1.0 (overrides sample_size).
        sample_strategy: Sampling strategy.
        sample_attempts_bins: Attempt-count bin edges (e.g. [20, 100]).
        sample_correct_bins: Correct-rate bin edges (e.g. [0.4, 0.8]).
    """

    dataset: str = ""
    data_base_path: str = "./data"
    fold: int = 0
    kfold: int = field(default=5, metadata={"preprocess_ui": True})
    test_ratio: float = 0.2
    min_seq_len: int = field(default=3, metadata={"preprocess_ui": True})
    max_seq_len: int = field(default=200, metadata={"preprocess_ui": True})
    sample_size: int | None = field(default=None, metadata={"preprocess_ui": True})
    sample_ratio: float | None = field(default=None, metadata={"preprocess_ui": True})
    sample_strategy: Literal["random", "stratified", "time"] = field(
        default="random", metadata={"preprocess_ui": True}
    )
    sample_attempts_bins: list[int] = field(
        default_factory=lambda: [20, 100], metadata={"preprocess_ui": True}
    )
    sample_correct_bins: list[float] = field(
        default_factory=lambda: [0.4, 0.8], metadata={"preprocess_ui": True}
    )


@dataclass
class DownloadConfig:
    """Options for the data_process.py download subcommand.

    Args:
        data_url: Override URL for downloading.
        force: Force re-download even if the file exists.
        max_retries: Maximum download retries.
        num_threads: Parallel download threads.
    """

    data_url: str | None = None
    force: bool = False
    max_retries: int = 3
    num_threads: int = 4


@dataclass
class ProcessConfig:
    """Options for the data_process.py process subcommand.

    Args:
        extra: Extra processing steps (e.g. windowlate).
    """

    extra: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Base class for per-model hyperparameter configs.

    Each model subclasses this and registers via ``@register_model_config``.
    The training knobs below are declared here as a contract (so the framework
    can always read ``rc.model.epochs`` / ``rc.model.batch_size`` / etc.);
    subclasses override the defaults with model-specific values and may carry
    an ``optuna`` field-metadata key consumed by the Optuna search-space derive.

    Args:
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
    """

    epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class RunConfig:
    """Top-level run configuration tree.

    The ``model`` node is polymorphic: the runtime schema is assembled with the
    concrete :class:`ModelConfig` subclass, never structured on this base.

    Args:
        general: Framework-level general knobs.
        compile: torch.compile execution knobs.
        early_stopping: Early-stopping knobs.
        experiment: Experiment identity.
        data: Dataset / split / sampling knobs.
        model: Per-model hyperparameters (concrete ModelConfig subclass).
    """

    general: GeneralConfig = field(default_factory=GeneralConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: RunDataConfig = field(default_factory=RunDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# Fixed framework nodes; ``model`` is filled per model_name at build time.
_FRAMEWORK_NODES: dict[str, type] = {
    "general": GeneralConfig,
    "compile": CompileConfig,
    "early_stopping": EarlyStoppingConfig,
    "experiment": ExperimentConfig,
    "data": RunDataConfig,
}


def build_run_config_schema(model_name: str) -> dict[str, type]:
    """Return ``{node_name: dataclass_cls}`` for the concrete model's tree.

    Binds the polymorphic ``model`` node to the concrete registered
    :class:`ModelConfig` subclass; the framework nodes are the fixed five.

    Raises:
        KeyError: If no ModelConfig is registered for ``model_name`` (raised
            by the registry, listing the available names).
    """
    from ..core import MODEL_CONFIGS  # lazy: avoid any import-time cycle

    model_cls = MODEL_CONFIGS.get(model_name)
    return {**_FRAMEWORK_NODES, "model": model_cls}


def config_to_dict(config) -> dict:
    """Recursively convert a config dataclass (instance or node) to a plain dict.

    Used for yaml serialization, metric logging, cache keys, and anywhere a
    config must cross into plain-Python land.
    """
    from dataclasses import asdict, is_dataclass

    if config is None:
        return {}
    return asdict(config) if is_dataclass(config) else dict(config)


__all__ = [
    "_FRAMEWORK_NODES",
    "CompileConfig",
    "DownloadConfig",
    "EarlyStoppingConfig",
    "ExperimentConfig",
    "GeneralConfig",
    "ModelConfig",
    "ProcessConfig",
    "RunConfig",
    "RunDataConfig",
    "build_run_config_schema",
    "config_to_dict",
]
