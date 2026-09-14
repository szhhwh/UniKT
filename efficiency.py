"""UniKT model efficiency benchmark.

Two mutually exclusive entry modes:
    - ``-m/-d``                       build the model fresh (random weights, or
                                      ``--efficiency.general.weights`` to load a file) + data
    - ``--efficiency.general.run_dir`` seed the RunConfig from a trained run's
                                      ``run_config.yaml`` and benchmark its checkpoint

Usage:
    python efficiency.py -m GIKT -d assistments09
    python efficiency.py -m SAKT -d assistments09 --efficiency.general.weights runs/.../best_model.pth
    python efficiency.py --efficiency.general.run_dir runs/normal/GIKT_assist09_..._fold0_bs128
    python efficiency.py -m AKT -d assistments09 --efficiency.general.modes inference --efficiency.inference.iters 500
    python efficiency.py -m SAKT -d assistments09 --efficiency.general.compile_modes off,default,reduce-overhead
    python efficiency.py -m SAKT -d assistments09 --efficiency.general.batch_sizes 32,64 --efficiency.general.compile_modes off,default
"""

import sys
from pathlib import Path

import model  # noqa: F401  — triggers trainer/model-config discovery
from utils.config import ConfigParser, build_node, peek_flag_value
from utils.config.config_parser import _reject_model_flags
from utils.core import add_file_handler, get_logger
from utils.data_process import get_data_source
from utils.efficiency import EfficiencySession, EfficiencySweep
from utils.efficiency.config import get_efficiency_config_cls
from utils.efficiency.session import build_target
from utils.experiment_manager import ExperimentManager, ExperimentType

logger = get_logger(__name__)


def main() -> None:
    """Build the model, run all enabled efficiency stages, and print the report."""
    rc, eff_cfg = _parse()
    # Suppress trainer side effects irrelevant to benchmarking.
    rc.general.cloud_tracking = False
    rc.general.checkpoint_path = None
    rc.general.skip_test = True

    weights_path = _resolve_weights(eff_cfg)
    logger.info(
        f"[Benchmark] model={rc.experiment.model_name} dataset={rc.data.dataset}"
    )
    _apply_benchmark_overrides(rc, eff_cfg)

    if eff_cfg.general.batch_sizes or eff_cfg.general.compile_modes:
        _run_sweep(rc, eff_cfg, weights_path)
    else:
        _run_single_efficiency(rc, eff_cfg, weights_path)


def _run_single_efficiency(rc, eff_cfg, weights_path: str | None) -> None:
    """Build one trainer and run a single efficiency session."""
    exp_manager = ExperimentManager.from_run_config(rc, ExperimentType.EFFICIENCY)
    add_file_handler(Path(exp_manager.get_log_dir()) / "run.log")
    output_dir = eff_cfg.general.output_dir or exp_manager.get_log_dir()
    logger.info(f"[Benchmark] output_dir={output_dir}")

    data_src = get_data_source(rc)
    if weights_path:
        logger.info(f"[Benchmark] loading weights: {weights_path}")
    target = build_target(rc, data_src, exp_manager, weights_path)
    EfficiencySession(
        target=target, rc=rc, eff_cfg=eff_cfg, output_dir=output_dir
    ).run().print_console()


def _run_sweep(rc, eff_cfg, weights_path: str | None) -> None:
    """Sweep a set of batch sizes, rebuilding the trainer per size."""
    data_src = get_data_source(rc)
    EfficiencySweep(
        rc=rc, eff_cfg=eff_cfg, data_src=data_src, weights_path=weights_path
    ).run()


def _parse() -> tuple:
    """Parse RunConfig + EfficiencyConfig; in run_dir mode seed from the archive."""
    EfficiencyConfig = get_efficiency_config_cls()

    run_dir = _peek_run_dir()
    default_config = None
    if run_dir:
        _reject_model_flags(
            sys.argv[1:],
            prog="efficiency.py",
            run_dir_flag="--efficiency.general.run_dir",
        )  # run_dir mode reconstructs the model from the archive
        archive = Path(run_dir) / "run_config.yaml"
        if not archive.exists():
            raise SystemExit(f"[Benchmark] run_config.yaml not found in {run_dir}")
        default_config = archive

    rc, ns = ConfigParser(
        prog="efficiency.py",
        description="UniKT Model Efficiency Benchmark",
        extra_nodes={"efficiency": EfficiencyConfig},
        default_config=default_config,
    ).parse_with_extras()
    eff_cfg = build_node(EfficiencyConfig, ns["efficiency"])
    return rc, eff_cfg


def _peek_run_dir() -> str | None:
    """Read --efficiency.general.run_dir before ConfigParser (default_config path needs it)."""
    return peek_flag_value(sys.argv[1:], "--efficiency.general.run_dir")


def _resolve_weights(eff_cfg) -> str | None:
    general = eff_cfg.general
    if general.weights:
        path = Path(general.weights)
    elif general.run_dir:
        path = Path(general.run_dir) / general.checkpoint
    else:
        return None
    # Fail fast before the expensive model+data build.
    if not path.exists():
        raise SystemExit(f"[Benchmark] checkpoint not found: {path}")
    return str(path)


def _apply_benchmark_overrides(rc, eff_cfg) -> None:
    """Force a uniform batch_size / max_seq_len so throughput compares across models.

    Throughput normalizes per interaction, but its wall-time denominator still
    scales with GPU utilization (batch size) and padding cost (seq_len). Without
    uniform inputs a bs=6 model and a bs=10000 model cannot be ranked by
    interactions/s. Opt-in: None keeps the model/dataset default. Applied before
    the sweep branch, so a batch-size sweep still wins on rc.model.batch_size
    while seq_len stays uniform across sweep points.
    """
    bs = eff_cfg.general.benchmark_batch_size
    seq = eff_cfg.general.benchmark_seq_len
    if bs is None and seq is None:
        return
    if bs is not None:
        rc.model.batch_size = bs
    if seq is not None:
        rc.data.max_seq_len = seq
    logger.info(
        "[Benchmark] uniform input override: "
        f"batch_size={rc.model.batch_size} seq_len={rc.data.max_seq_len}"
    )


if __name__ == "__main__":
    main()
