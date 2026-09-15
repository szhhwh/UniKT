"""Optuna hyperparameter search runner.

Provides a command-line interface for running Optuna-based hyperparameter
searches on KT models. Supports configurable parameter spaces, multiple
optimization metrics, and trial history export. Entry-point knobs live under
``--optuna_search.*`` alongside the reflective RunConfig flags.
"""

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

import model  # noqa: F401
from utils.config import ConfigParser, build_node, config_to_dict
from utils.core import TRAINERS, add_file_handler, get_logger
from utils.data_process import get_data_source
from utils.experiment_manager import ExperimentManager, ExperimentType
from utils.optuna_utils import (
    OptunaTuner,
    TrainerObjectiveWrapper,
    direction_for_metric,
    load_optuna_config,
    param_spaces_from_model_config,
)

logger = get_logger(__name__)

_METRICS = ("auc", "acc", "auprc", "rmse", "loss")


@dataclass
class OptunaSearchConfig:
    """Entry-point knobs for optuna_search.py, exposed as ``--optuna_search.*``.

    Args:
        optuna_config: Path to the Optuna yaml (n_trials/sampler/pruner/...).
        metric: Comma-separated metric(s) to optimize (e.g. ``auc`` or
            ``auc,rmse``); more than one enables multi-objective search.
        resume: Resume an existing search from its run directory (reuses
            study.db).
        keep_trial_artifacts: Keep per-trial SwanLab tracking, checkpoints,
            and test evaluation (disabled by default during search to save
            compute).
        output_dir: Fixed output directory for all search artifacts (study.db,
            trial subdirs, CSV). When set, the timestamped ExperimentManager
            dir is skipped so callers (e.g. the web backend) can locate
            study.db deterministically. Omit to keep the default timestamped
            behaviour.
    """

    optuna_config: str = "./configs/optuna/optuna_config.yaml"
    metric: str = "auc"
    resume: str | None = None
    keep_trial_artifacts: bool = False
    output_dir: str | None = None


def _parse(argv: list[str] | None = None):
    """Parse the reflective RunConfig flags plus the OptunaSearchConfig node.

    Returns ``(rc, opt_cfg, metrics)`` with ``metrics`` split from the
    comma-separated ``--optuna_search.metric`` and validated.
    """
    rc, ns = ConfigParser(
        prog="optuna_search.py",
        description="Unified Optuna Hyperparameter Search",
        extra_nodes={"optuna_search": OptunaSearchConfig},
    ).parse_with_extras(argv)
    opt_cfg = build_node(OptunaSearchConfig, ns["optuna_search"])
    metrics = [m.strip() for m in opt_cfg.metric.split(",") if m.strip()]
    invalid = [m for m in metrics if m not in _METRICS]
    if invalid:
        raise SystemExit(
            f"optuna_search.py: invalid metric(s): {', '.join(invalid)} "
            f"(choose from {', '.join(_METRICS)})"
        )
    return rc, opt_cfg, metrics


def main():
    """Main entry point."""
    rc, optuna_args, metrics = _parse()
    model_name = rc.experiment.model_name

    # Load config first to annotate experiment directory with n_trials
    logger.info(f"Loading Optuna config from: {optuna_args.optuna_config}")
    optuna_config = load_optuna_config(optuna_args.optuna_config)

    if optuna_args.resume:
        # Reuse the existing run dir so study.db and study_name match the prior
        # run, letting create_study(load_if_exists=True) restore trial history.
        exp_manager = ExperimentManager.from_run_dir(optuna_args.resume)
        logger.info(f"Resuming search in: {exp_manager.get_log_dir()}")
    elif optuna_args.output_dir:
        # Web backend supplies a deterministic dir so study.db is locatable;
        # wrap it directly instead of creating a fresh timestamped run dir.
        os.makedirs(optuna_args.output_dir, exist_ok=True)
        exp_manager = ExperimentManager.from_run_dir(optuna_args.output_dir)
        exp_manager.exp_type = ExperimentType.HYPERPARAM_SEARCH
        logger.info(f"Experiment directory: {exp_manager.get_log_dir()}")
    else:
        exp_manager = ExperimentManager(
            exp_type=ExperimentType.HYPERPARAM_SEARCH,
            model_name=model_name,
            dataset_name=rc.data.dataset,
            base_dir="runs",
            tags=[f"n_trials{optuna_config.n_trials}"],
        )
        logger.info(f"Experiment directory: {exp_manager.get_log_dir()}")
    add_file_handler(Path(exp_manager.get_log_dir()) / "run.log")

    logger.info("=" * 60)
    logger.info(f"{model_name} Optuna Hyperparameter Search")
    logger.info("=" * 60)

    optuna_config.save_dir = exp_manager.get_log_dir()
    if optuna_args.resume:
        # study_name + db path must match the original run for load_if_exists.
        # n_trials is the number of NEW trials this invocation (Optuna semantics).
        logger.info(
            f"Resume: study_name='{optuna_config.study_name}' (must match the "
            f"original run), db={optuna_config.save_dir}/study.db"
        )
    # Directions are derived from the metric(s); multiple metrics enable
    # multi-objective search (override any directions in the config file).
    optuna_config.directions = [direction_for_metric(m) for m in metrics]
    if len(metrics) > 1:
        logger.info(
            f"Multi-objective search: metrics={metrics}, "
            f"directions={optuna_config.directions}"
        )
        logger.info("For multi-objective, set 'sampler: nsgaii' in the optuna config")
    else:
        logger.info(f"Optimizing metric '{metrics[0]}' ({optuna_config.directions[0]})")

    # Suppression is search-scoped; apply it to a copy so `rc` stays clean for the
    # reproduction config. --keep-trial-artifacts opts out of suppression entirely.
    search_rc = copy.deepcopy(rc)
    if not optuna_args.keep_trial_artifacts:
        search_rc.general.cloud_tracking = False
        search_rc.general.save_last_checkpoint = False
        search_rc.general.skip_test = True
        logger.info(
            "Per-trial SwanLab tracking, checkpoint saving, and test evaluation "
            "disabled to save compute (pass --optuna_search.keep_trial_artifacts "
            "true to keep them)"
        )

    # Search space is derived solely from the model's ModelConfig field metadata.
    param_spaces = param_spaces_from_model_config(model_name)
    if not param_spaces:
        raise ValueError(
            f"{model_name}Config has no fields with 'optuna' metadata, so there is "
            "nothing to search. To enable hyperparameter search, annotate fields on "
            "its ModelConfig with metadata={'optuna': {...}}.\n"
            "Examples (see model/GIKT/GIKT_trainer.py):\n"
            "    n_hop: int = field(default=3, "
            "metadata={'optuna': {'type': 'int', 'low': 1, 'high': 5}})\n"
            "    embedding_dim: int = field(default=100, "
            "metadata={'optuna': {'type': 'int', 'low': 64, 'high': 256, 'log': True}})\n"
            "    batch_size: int = field(default=32, "
            "metadata={'optuna': {'type': 'categorical', 'choices': [32, 64, 128, 256]}})\n"
            "Supported 'type' values: 'int', 'float', 'categorical'; "
            "keys: low, high, log, step, choices."
        )
    logger.info(
        f"Searchable params from {model_name}Config: {[s.name for s in param_spaces]}"
    )

    def data_src_factory():
        return get_data_source(search_rc)

    trainer_class = TRAINERS.get(model_name)

    objective_wrapper = TrainerObjectiveWrapper(
        trainer_class=trainer_class,
        data_src_fn=data_src_factory,
        base_rc=search_rc,
        metric_name=metrics,
        exp_manager=exp_manager,
    )

    tuner = OptunaTuner(
        config=optuna_config,
        param_space=param_spaces,
        objective_fn=objective_wrapper,
    )

    logger.info(
        f"Starting hyperparameter search with {optuna_config.n_trials} trials..."
    )
    best_params = tuner.search()

    tuner.print_summary()

    # tuner copied the best trial's (suppressed) archive into best_run_config.yaml;
    # rebuild it from the clean rc so `train.py --config best_run_config.yaml`
    # retrains normally (swanlab/checkpoints/test on), not with search suppression.
    if not optuna_args.keep_trial_artifacts and best_params:
        repro_rc = copy.deepcopy(rc)
        for name, value in best_params.items():
            setattr(repro_rc.model, name, value)
        repro_cfg_path = Path(optuna_config.save_dir) / "best_run_config.yaml"
        repro_cfg_path.write_text(
            yaml.safe_dump(
                config_to_dict(repro_rc), sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )
        logger.info(f"Reproducible best run config saved to {repro_cfg_path}")

    df = tuner.get_dataframe()
    if df is not None:
        log_dir = exp_manager.get_log_dir()
        df_path = os.path.join(log_dir, f"trials_history_{model_name.lower()}.csv")
        df.to_csv(df_path, index=False)
        logger.info(f"Trials history saved to: {df_path}")

    logger.info("=" * 60)
    logger.info("Search completed successfully!")
    logger.info("=" * 60)

    return best_params


if __name__ == "__main__":
    main()
