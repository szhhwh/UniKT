"""Evaluate a trained model on the test set.

The archived ``run_config.yaml`` inside the run directory seeds the RunConfig;
any reflective RunConfig flag overrides it (``--general.device``,
``--model.batch_size``, ``--data.data_base_path``, ...). Entry-point knobs live
under ``--evaluate.*``.

Usage:
    python evaluate.py --evaluate.run_dir runs/normal/GIKT_assist09_20260520-232131_fold0_bs128
    python evaluate.py --evaluate.run_dir ... --evaluate.checkpoint last_checkpoint.pth
    python evaluate.py --evaluate.run_dir ... --general.device cpu
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import model  # noqa: F401  — triggers trainer/model-config discovery
from utils.config import RunConfig, parse_run_archive
from utils.core import TRAINERS, add_file_handler, get_logger, seed_everything
from utils.data_process import get_data_source
from utils.experiment_manager import ExperimentManager

logger = get_logger(__name__)


@dataclass
class EvaluateConfig:
    """Entry-point knobs for evaluate.py, exposed as ``--evaluate.*`` flags.

    Args:
        run_dir: Trained run directory; its ``run_config.yaml`` seeds the
            RunConfig and locates the checkpoint.
        checkpoint: Checkpoint filename inside ``run_dir``.
    """

    run_dir: str
    checkpoint: str = "best_model.pth"


def _parse(argv: list[str] | None = None) -> tuple[RunConfig, Any, Path]:
    """Parse the archived RunConfig (plus CLI overrides) + EvaluateConfig."""
    return parse_run_archive(
        argv,
        prog="evaluate.py",
        description="Evaluate a trained KT model on the test set",
        entry_node="evaluate",
        entry_cls=EvaluateConfig,
    )


def main() -> None:
    """Evaluate a trained model checkpoint on the test set."""
    rc, ev_cfg, run_dir = _parse()
    checkpoint_path = run_dir / ev_cfg.checkpoint

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    model_name = rc.experiment.model_name
    dataset_name = rc.data.dataset

    # Override for evaluation mode
    rc.general.cloud_tracking = False
    rc.general.checkpoint_path = None  # weights loaded manually after build
    rc.general.skip_test = True  # prevent TestEvaluationCallback during build

    seed_everything(rc.general.seed, deterministic=rc.general.deterministic)

    exp_manager = ExperimentManager.from_run_dir(run_dir)
    # Route evaluate outputs into run_dir/evaluate/ to keep the training dir
    # untouched (mirrors case_analysis/).
    eval_manager = exp_manager.create_sub_experiment("evaluate")
    add_file_handler(Path(eval_manager.get_log_dir()) / "run.log")

    # Header logs emitted after the file sink is attached so they reach run.log.
    logger.info(f"Model: {model_name}  Dataset: {dataset_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Loading dataset: {dataset_name}...")
    data_src = get_data_source(rc)

    logger.info(f"Initializing trainer for model: {model_name}...")
    trainer = TRAINERS.get(model_name)(
        rc=rc, data_src=data_src, exp_manager=eval_manager
    )
    trainer.load_weights(str(checkpoint_path))

    logger.info("Running evaluation on test set...")
    trainer.evaluate()


if __name__ == "__main__":
    main()
