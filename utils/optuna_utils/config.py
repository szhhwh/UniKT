"""Optuna configuration and parameter space utilities."""

from dataclasses import dataclass, field
from typing import Any

import yaml
from optuna.pruners import (
    BasePruner,
    HyperbandPruner,
    MedianPruner,
    PatientPruner,
    PercentilePruner,
    SuccessiveHalvingPruner,
    ThresholdPruner,
    WilcoxonPruner,
)
from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    GPSampler,
    GridSampler,
    NSGAIISampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)
from optuna.trial import Trial

# auc/acc/auprc are maximise, rmse/loss are minimise
_METRIC_DIRECTIONS: dict[str, str] = {
    "auc": "maximize",
    "acc": "maximize",
    "auprc": "maximize",
    "rmse": "minimize",
    "loss": "minimize",
}


def direction_for_metric(metric_name: str) -> str:
    """Return the Optuna optimisation direction for a given metric name.

    Args:
        metric_name: Name of the metric (e.g. 'auc', 'acc', 'rmse', 'loss').

    Returns:
        'maximize' or 'minimize'.

    Raises:
        ValueError: If the metric name is not recognised.
    """
    direction = _METRIC_DIRECTIONS.get(metric_name.lower())
    if direction is None:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. "
            f"Expected one of: {sorted(_METRIC_DIRECTIONS)}"
        )
    return direction


@dataclass
class HyperparameterSpace:
    """Definition of a single hyperparameter search space."""

    name: str
    type: str  # 'int', 'float', 'categorical'
    low: float | None = None
    high: float | None = None
    log: bool | None = None  # Logarithmic sampling for numeric params
    step: float | None = None  # Step size for integer params
    choices: list[Any] | None = None  # Choices for categorical params
    default: Any | None = None

    def validate(self) -> None:
        """Validate the parameter space configuration completeness."""
        if self.type == "int":
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Integer parameter '{self.name}' requires 'low' and 'high'"
                )
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}': low must be less than high")
            if self.log and self.step is not None:
                raise ValueError(
                    f"Parameter '{self.name}': 'step' and 'log' are mutually exclusive"
                )
            if self.step is not None and self.step <= 0:
                raise ValueError(f"Parameter '{self.name}': step must be positive")
        elif self.type == "float":
            if self.low is None or self.high is None:
                raise ValueError(
                    f"Float parameter '{self.name}' requires 'low' and 'high'"
                )
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}': low must be less than high")
            if self.log and self.step is not None:
                raise ValueError(
                    f"Parameter '{self.name}': 'step' and 'log' are mutually exclusive"
                )
            if self.step is not None and self.step <= 0:
                raise ValueError(f"Parameter '{self.name}': step must be positive")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError(
                    f"Categorical parameter '{self.name}' requires 'choices'"
                )
        else:
            raise ValueError(f"Unsupported parameter type: {self.type}")

        if self.default is not None:
            if self.type in ("int", "float"):
                if not (self.low <= self.default <= self.high):
                    raise ValueError(
                        f"Parameter '{self.name}': default {self.default} "
                        f"out of range [{self.low}, {self.high}]"
                    )
            elif (
                self.type == "categorical"
                and self.choices is not None
                and self.default not in self.choices
            ):
                raise ValueError(
                    f"Parameter '{self.name}': default {self.default} "
                    f"not in choices {self.choices}"
                )

    def suggest(self, trial: Trial) -> Any:
        """Sample a parameter value from the Optuna trial.

        Args:
            trial: Optuna trial object.

        Returns:
            The sampled parameter value.
        """
        self.validate()

        if self.type == "int":
            assert self.low is not None and self.high is not None  # validated above
            return trial.suggest_int(
                self.name,
                low=int(self.low),
                high=int(self.high),
                step=int(self.step) if self.step else 1,
                log=self.log or False,
            )
        elif self.type == "float":
            assert self.low is not None and self.high is not None  # validated above
            return trial.suggest_float(
                self.name,
                low=float(self.low),
                high=float(self.high),
                step=float(self.step) if self.step else None,
                log=self.log or False,
            )
        elif self.type == "categorical":
            assert self.choices is not None  # validated above
            return trial.suggest_categorical(self.name, self.choices)


@dataclass
class OptunaConfig:
    """Optuna search configuration."""

    # Sampler configuration
    sampler: str = "tpe"  # 'tpe','random','grid','cmaes','gp','nsgaii','qmc' (nsgaii: multi-objective)
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)
    seed: int = 42  # Seed for stochastic samplers (tpe/random/cmaes/gp/nsgaii/qmc); grid exhaustive

    # Pruner configuration
    pruner: str = "median"  # 'median','percentile','successive_halving','hyperband','threshold','wilcoxon','patient', None
    pruner_kwargs: dict[str, Any] = field(default_factory=dict)

    # Search configuration
    n_trials: int = 100
    n_jobs: int = 1  # Number of parallel jobs
    timeout: int | None = None  # Timeout in seconds
    directions: list[str] = field(
        default_factory=lambda: ["maximize"]
    )  # 'maximize' or 'minimize'

    # Storage and logging
    study_name: str | None = None
    db_url: str | None = None  # Persistence DB URL, e.g. "sqlite:///study.db"
    save_dir: str | None = None
    verbose: int = 1  # 0=quiet, 1=normal, 2=verbose

    def get_sampler(self) -> BaseSampler:
        """Create a sampler based on the configuration.

        Returns:
            An Optuna BaseSampler instance.

        Raises:
            ValueError: If the sampler name is not recognised.
        """
        sampler_name = self.sampler.lower()
        kwargs = self.sampler_kwargs.copy()
        # Stochastic samplers are seeded for reproducibility (grid is exhaustive)
        if sampler_name != "grid" and "seed" not in kwargs:
            kwargs["seed"] = self.seed

        if sampler_name == "tpe":
            return TPESampler(**kwargs)
        elif sampler_name == "random":
            return RandomSampler(**kwargs)
        elif sampler_name == "grid":
            if "search_space" not in kwargs:
                raise ValueError(
                    "GridSampler requires 'search_space' in sampler_kwargs, "
                    'e.g. {"lr": [1e-3, 1e-4], "layers": [1, 2]}'
                )
            return GridSampler(**kwargs)
        elif sampler_name == "cmaes":
            return CmaEsSampler(**kwargs)
        elif sampler_name == "gp":
            return GPSampler(**kwargs)
        elif sampler_name == "nsgaii":
            return NSGAIISampler(**kwargs)
        elif sampler_name == "qmc":
            return QMCSampler(**kwargs)
        else:
            raise ValueError(f"Unsupported sampler: {sampler_name}")

    def get_pruner(self) -> BasePruner | None:
        """Create a pruner based on the configuration.

        Returns:
            An Optuna BasePruner instance, or None if no pruner is configured.

        Raises:
            ValueError: If the pruner name is not recognised.
        """
        if self.pruner is None:
            return None

        pruner_name = self.pruner.lower()
        kwargs = self.pruner_kwargs.copy()

        if pruner_name == "median":
            return MedianPruner(**kwargs)
        elif pruner_name == "percentile":
            if "percentile" not in kwargs:
                raise ValueError(
                    "PercentilePruner requires 'percentile' (0-100) in pruner_kwargs"
                )
            return PercentilePruner(**kwargs)
        elif pruner_name == "successive_halving":
            return SuccessiveHalvingPruner(**kwargs)
        elif pruner_name == "hyperband":
            return HyperbandPruner(**kwargs)
        elif pruner_name == "threshold":
            return ThresholdPruner(**kwargs)
        elif pruner_name == "wilcoxon":
            return WilcoxonPruner(**kwargs)
        elif pruner_name == "patient":
            return PatientPruner(**kwargs)
        else:
            raise ValueError(f"Unsupported pruner: {pruner_name}")


def load_optuna_config(config_path: str) -> OptunaConfig:
    """Load an :class:`OptunaConfig` from a yaml file.

    The yaml is parsed with PyYAML into a dict and passed to the
    ``OptunaConfig`` constructor (so methods like ``get_sampler`` are preserved
    and unknown keys fail loudly).

    Args:
        config_path: Path to the yaml configuration file.

    Returns:
        An OptunaConfig instance.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return OptunaConfig(**cfg)


def _default_fits(space: HyperparameterSpace, default: Any) -> bool:
    """Whether a field default is compatible with the search space (for enqueue)."""
    if default is None:
        return False
    if space.type in ("int", "float"):
        return (
            space.low is not None
            and space.high is not None
            and space.low <= default <= space.high
        )
    if space.type == "categorical":
        return space.choices is not None and default in space.choices
    return False


def param_spaces_from_model_config(model_name: str) -> list[HyperparameterSpace]:
    """Derive the optuna search space from a ModelConfig's field metadata.

    Each searchable field carries an ``"optuna"`` metadata dict
    (``{"type": "int"|"float"|"categorical", "low", "high", "log", "step",
    "choices"}``). The field default becomes the enqueued-trial default when it
    lies within the space; otherwise it is omitted (default=None).

    Args:
        model_name: Registered model name.

    Returns:
        A list of :class:`HyperparameterSpace` (one per field with optuna metadata).

    Raises:
        KeyError: If no ModelConfig is registered for ``model_name``.
    """
    from dataclasses import MISSING
    from dataclasses import fields as dc_fields

    from utils.core import MODEL_CONFIGS

    model_cls = MODEL_CONFIGS.get(model_name)

    spaces: list[HyperparameterSpace] = []
    for f in dc_fields(model_cls):
        spec = f.metadata.get("optuna")
        if not spec:
            continue
        default = f.default_factory() if f.default_factory is not MISSING else f.default
        space = HyperparameterSpace(name=f.name, **spec)
        space.default = default if _default_fits(space, default) else None
        spaces.append(space)
    return spaces


__all__ = [
    "HyperparameterSpace",
    "OptunaConfig",
    "direction_for_metric",
    "load_optuna_config",
    "param_spaces_from_model_config",
]
