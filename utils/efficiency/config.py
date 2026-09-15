"""Efficiency benchmark config: entry knobs + general node + one sub-node per stage.

The ``EfficiencyConfig`` dataclass is composed at schema-build time from
:class:`EfficiencyEntryConfig` (flat ``--efficiency.*`` entry knobs),
:class:`GeneralEfficiencyConfig`, plus each registered stage's own ``config_cls``
(mirroring ``build_run_config_schema``'s model-name → subclass binding). This is
the repo's first nested config node, so reconstruction recurses
(see :func:`utils.config.config_parser.build_node`).

Exposed via ``ConfigParser(extra_nodes={"efficiency": get_efficiency_config_cls()})``
in ``efficiency.py``. Lives outside the shared :class:`RunConfig` tree so it never
appears in ``train.py``'s flags.
"""

from dataclasses import dataclass, field, make_dataclass


@dataclass
class EfficiencyEntryConfig:
    """Entry-point knobs exposed as flat ``--efficiency.*`` flags.

    Mirrors ``EvaluateConfig``'s run_dir/checkpoint contract so archive-restoring
    entry points name their run directory the same way (``--evaluate.run_dir``,
    ``--case.run_dir``, ``--efficiency.run_dir``).

    Args:
        run_dir: Trained run dir; seed rc from its run_config.yaml and benchmark
            its checkpoint.
        checkpoint: Checkpoint filename inside ``run_dir``.
        weights: Standalone checkpoint to load after building the model
            (fresh-mode alternative to ``run_dir``).
    """

    run_dir: str | None = None
    checkpoint: str = "best_model.pth"
    weights: str | None = None


@dataclass
class GeneralEfficiencyConfig:
    """Cross-stage knobs (owned by no single stage).

    Args:
        modes: Comma-separated stages to run (empty = all discovered stages).
        batch_sizes: Comma-separated batch sizes to sweep (empty = single run).
            Each size rebuilds a fresh trainer for a clean CUDA allocator.
        compile_modes: Comma-separated compile states to sweep (empty = single
            run). Each entry is ``off`` (compile disabled, baseline) or a valid
            ``torch.compile`` mode (``default`` / ``reduce-overhead`` /
            ``max-autotune`` / ``max-autotune-no-cudagraphs``). Combines with
            ``batch_sizes`` into a Cartesian-product sweep.
        warmup_iters: Discarded iters before timing (cuDNN autotune / clock ramp);
            shared by the inference/train/trace stages.
        resource_sample_interval: Background resource sampling interval (s).
        output_dir: Where to write efficiency_report.json; default = exp dir.
        benchmark_batch_size: Override rc.model.batch_size so every model is
            measured under one batch size; None keeps the model's default.
            Throughput is per-interaction but its wall-time still scales with GPU
            utilization, which depends on batch size — uniformize to compare.
        benchmark_seq_len: Override rc.data.max_seq_len so every model pads to
            one length; None keeps the dataset default.
    """

    modes: str = ""
    batch_sizes: str = ""
    compile_modes: str = ""
    warmup_iters: int = 50
    resource_sample_interval: float = 0.05
    output_dir: str | None = None
    benchmark_batch_size: int | None = None
    benchmark_seq_len: int | None = None


@dataclass
class DefaultStageConfig:
    """Fallback config for a stage that declares no ``config_cls``."""


_EFFICIENCY_CONFIG_CLS: type | None = None


def build_efficiency_config_schema() -> type:
    """Compose (once, cached) the ``EfficiencyConfig`` dataclass.

    Builds ``general`` plus one sub-node per registered efficiency stage, bound
    to that stage's ``config_cls``. Cached because ``make_dataclass`` returns a
    new type each call and all callers (``extra_nodes``, ``asdict``,
    ``dataclass_fields``) must share the same class object.
    """
    global _EFFICIENCY_CONFIG_CLS
    if _EFFICIENCY_CONFIG_CLS is not None:
        return _EFFICIENCY_CONFIG_CLS

    from utils.core import EFFICIENCY_STAGES, get_supported_stages

    stage_fields = [
        (
            "general",
            GeneralEfficiencyConfig,
            field(default_factory=GeneralEfficiencyConfig),
        )
    ]
    for name in get_supported_stages():
        stage_cls = EFFICIENCY_STAGES.get(name)
        cfg_cls = getattr(stage_cls, "config_cls", DefaultStageConfig)
        stage_fields.append((name, cfg_cls, field(default_factory=cfg_cls)))
    _EFFICIENCY_CONFIG_CLS = make_dataclass(
        "EfficiencyConfig", stage_fields, bases=(EfficiencyEntryConfig,)
    )
    return _EFFICIENCY_CONFIG_CLS


def get_efficiency_config_cls() -> type:
    """Return the cached composed ``EfficiencyConfig`` class (builds on first call)."""
    return build_efficiency_config_schema()
