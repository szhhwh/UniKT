"""Unified knowledge tracing training script."""

from pathlib import Path

import model  # noqa: F401
from utils.config import ConfigParser
from utils.core import TRAINERS, add_file_handler, get_logger, seed_everything
from utils.data_process import get_data_source
from utils.experiment_manager import ExperimentManager, ExperimentType

logger = get_logger(__name__)


def main() -> None:
    """Train a knowledge tracing model."""
    rc = ConfigParser(
        prog="train.py", description="Knowledge Tracing Training Script"
    ).parse_args()
    model_name = rc.experiment.model_name

    # Seed as early as possible so init, data loading, and training RNG are reproducible.
    seed_everything(rc.general.seed, deterministic=rc.general.deterministic)
    exp_manager = ExperimentManager.from_run_config(rc, ExperimentType.NORMAL)
    add_file_handler(Path(exp_manager.get_log_dir()) / "run.log")
    logger.info(f"Experiment directory: {exp_manager.get_log_dir()}")

    logger.info(f"Building dataset: {rc.data.dataset}...")
    data_src = get_data_source(rc)

    logger.info(f"Initializing trainer for model: {model_name}...")
    trainer = TRAINERS.get(model_name)(
        rc=rc, data_src=data_src, exp_manager=exp_manager
    )
    logger.info("Starting training...")
    trainer.run()


if __name__ == "__main__":
    main()
