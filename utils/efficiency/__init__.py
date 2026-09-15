"""Model efficiency benchmark module.

Standalone efficiency benchmark: build model + data, then measure model scale,
compute cost, inference efficiency, training efficiency (peak memory +
throughput), and system/environment metadata under controlled conditions for a
fair, reproducible efficiency comparison.

Stages are pluggable: each lives under :mod:`utils.efficiency.stages`, registers
via ``@register_efficiency_stage("name")``, and is auto-discovered. Entry:
``python efficiency.py -m MODEL -d DATASET``.
"""

from . import stages  # noqa: F401  — triggers static stage discovery
from .config import (
    EfficiencyEntryConfig,
    GeneralEfficiencyConfig,
    get_efficiency_config_cls,
)
from .environment import (
    EnvironmentInfo,
    ResourceSampler,
    ResourceStats,
    ResourceSummary,
    collect_environment,
)
from .report import EfficiencyReport
from .session import EfficiencySession
from .stages.base import EfficiencyStage, StageContext
from .stages.inference import InferenceMetrics, benchmark_inference
from .stages.profile import ModelProfile, profile_model
from .stages.trace import TraceProfile, benchmark_trace
from .stages.training import TrainingMetrics, benchmark_training
from .sweep import EfficiencySweep, SweepPoint, SweepReport, batch_size_sweep

__all__ = [
    "EfficiencyEntryConfig",
    "EfficiencyReport",
    "EfficiencySession",
    "EfficiencyStage",
    "EfficiencySweep",
    "EnvironmentInfo",
    "GeneralEfficiencyConfig",
    "InferenceMetrics",
    "ModelProfile",
    "ResourceSampler",
    "ResourceStats",
    "ResourceSummary",
    "StageContext",
    "SweepPoint",
    "SweepReport",
    "TraceProfile",
    "TrainingMetrics",
    "batch_size_sweep",
    "benchmark_inference",
    "benchmark_trace",
    "benchmark_training",
    "collect_environment",
    "get_efficiency_config_cls",
    "profile_model",
]
