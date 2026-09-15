#!/usr/bin/env python3
r"""Case Analysis Framework for KT Models.

Plugin-driven CLI for analyzing trained KT models at the per-student
level:

1. ``inference``: restore a run (its ``run_config.yaml`` seeds the
   RunConfig; any reflective flag overrides it), run its registered
   analyzer, hand the output to a sink (default: canonical parquet via
   DataFrameSink)
2. ``select``: pick representative users with a selector plugin
   (default: diverse/extreme/random over per-user metrics)
3. ``plot``: render selected users with a visualizer plugin
   (default: knowledge-state heatmap)

Usage:
    # Step 1: Run inference
    python case_analysis.py inference \
        --case.run_dir runs/normal/HDHKT_assistments09_xxx_fold0

    # Step 2: Select users
    python case_analysis.py select \
        --case.run_dir runs/normal/HDHKT_assistments09_xxx_fold0 \
        --case.selector diverse --case.num_users 10

    # Step 3: Generate visualizations
    python case_analysis.py plot \
        --case.run_dir runs/normal/HDHKT_assistments09_xxx_fold0 \
        --case.selected_users diverse
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from jsonargparse import ArgumentParser

import model  # noqa: F401  (triggers analyzer registry discovery)
from utils.case_analysis import (
    DataFrameSink,
    compute_user_metrics,
    get_user_sequence,
    load_case_results,
)
from utils.config import build_node, parse_run_archive
from utils.core import (
    ANALYZERS,
    CASE_SELECTORS,
    CASE_SINKS,
    CASE_VISUALIZERS,
    get_logger,
    seed_everything,
)
from utils.data_process import get_data_source

logger = get_logger(__name__)


@dataclass
class CaseInferenceConfig:
    """``inference`` subcommand knobs (``--case.*`` flags).

    Args:
        run_dir: Run directory containing ``run_config.yaml`` and the checkpoint.
        checkpoint: Checkpoint filename inside ``run_dir``.
        sink: Case data sink plugin name (CASE_SINKS registry).
    """

    run_dir: str
    checkpoint: str = "best_model.pth"
    sink: str = "dataframe"


@dataclass
class CaseSelectConfig:
    """``select`` subcommand knobs (``--case.*`` flags).

    Defaults mirror the built-in selectors' ``select`` signature but are
    declared here on purpose: the public CLI surface stays stable even if a
    plugin's Python default changes.

    Args:
        run_dir: Run directory whose ``case_analysis/predictions.parquet`` to read.
        selector: User selector plugin name (CASE_SELECTORS registry).
        num_users: Maximum number of users to select.
        min_seq_len: Minimum attempt count required.
        min_error: Error-rate window lower bound (pairs with max_error).
        max_error: Error-rate window upper bound (pairs with min_error).
        min_confidence: Mean-confidence window lower bound.
        max_confidence: Mean-confidence window upper bound.
    """

    run_dir: str
    selector: str = "diverse"
    num_users: int = 20
    min_seq_len: int = 20
    min_error: float = 0.1
    max_error: float = 0.9
    min_confidence: float = 0.3
    max_confidence: float = 0.95


@dataclass
class CasePlotConfig:
    """``plot`` subcommand knobs.

    Args:
        run_dir: Run directory whose ``case_analysis/`` tree to read.
        selected_users: Selector name (e.g. ``diverse``) or path to a
            selected_users.json file.
        visualizer: Visualizer plugin name (CASE_VISUALIZERS registry).
        max_seq_len: Rendering truncation only; unrelated to
            ``--data.max_seq_len`` (the model was trained on).
    """

    run_dir: str
    selected_users: str
    visualizer: str = "heatmap"
    max_seq_len: int | None = None


def cmd_inference(rc, case):
    """Step 1: Run inference and save predictions."""
    run_dir = Path(case.run_dir).resolve()
    checkpoint_path = run_dir / case.checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_name = rc.experiment.model_name
    dataset_name = rc.data.dataset

    seed_everything(rc.general.seed, deterministic=rc.general.deterministic)

    if model_name not in ANALYZERS:
        sys.exit(
            f"Model '{model_name}' has no registered case analyzer. "
            f"Available: {sorted(ANALYZERS.keys())}"
        )
    AnalyzerClass = ANALYZERS.get(model_name)

    logger.info(f"Starting inference for {model_name} on {dataset_name}...")

    data_src = get_data_source(rc)
    if case.sink not in CASE_SINKS:
        sys.exit(f"Unknown sink '{case.sink}'. Available: {sorted(CASE_SINKS.keys())}")
    sink = CASE_SINKS.get(case.sink)()
    # device/batch_size overrides ride on rc (--general.device /
    # --model.batch_size); the analyzer falls back to rc when left None.
    analyzer = AnalyzerClass(
        rc=rc,
        data_src=data_src,
        checkpoint_path=str(checkpoint_path),
        sink=sink,
    )
    result = analyzer.run_inference()

    if not isinstance(result, pd.DataFrame):
        logger.info(
            f"Sink '{case.sink}' produced a non-DataFrame result; "
            "persistence is the sink's own responsibility."
        )
        return

    output_dir = run_dir / "case_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.parquet"
    DataFrameSink.save(result, str(predictions_path))

    user_metrics = compute_user_metrics(result)
    metrics_path = output_dir / "user_summaries.parquet"
    user_metrics.to_parquet(metrics_path, index=False)

    logger.info("✓ Inference complete!")
    logger.info(f"Predictions saved to: '{predictions_path}'")
    logger.info(f"User metrics saved to: '{metrics_path}'")
    logger.info(f"Total predictions: {len(result)}")


def cmd_select(args):
    """Step 2: Select users from existing predictions."""
    logger.info("Selecting users from predictions...")

    run_dir = Path(args.run_dir).resolve()
    predictions_path = run_dir / "case_analysis" / "predictions.parquet"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}\nPlease run 'inference' command first."
        )

    df = load_case_results(str(predictions_path))

    if args.selector not in CASE_SELECTORS:
        sys.exit(
            f"Unknown selector '{args.selector}'. "
            f"Available: {sorted(CASE_SELECTORS.keys())}"
        )
    SelectorClass = CASE_SELECTORS.get(args.selector)

    # Selector kwargs follow the UserSelector interface; the two window
    # options are tuple-valued in the interface, scalar min/max on the CLI.
    selected_users = SelectorClass().select(
        df,
        min_seq_len=args.min_seq_len,
        error_rate_range=(args.min_error, args.max_error),
        confidence_range=(args.min_confidence, args.max_confidence),
        max_users=args.num_users,
    )

    if not selected_users:
        logger.warning("No users selected. Try adjusting the filtering criteria.")
        return

    output_dir = run_dir / "case_analysis" / args.selector
    output_dir.mkdir(parents=True, exist_ok=True)

    user_metrics = compute_user_metrics(df)
    selected_metrics = user_metrics[user_metrics["user_id"].isin(selected_users)]

    selected_users_path = output_dir / "selected_users.json"
    selected_metrics.to_json(selected_users_path, orient="records", indent=2)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Selected users saved to: {selected_users_path}")
    logger.info("Selected users statistics:")
    logger.info(
        f" - Num attempts: {selected_metrics['num_attempts'].min():.0f} - {selected_metrics['num_attempts'].max():.0f}"
    )
    logger.info(
        f" - Error rate: {selected_metrics['error_rate'].min():.3f} - {selected_metrics['error_rate'].max():.3f}"
    )
    logger.info(f" - Accuracy: {selected_metrics['accuracy'].mean():.3f} avg")


def cmd_plot(args):
    """Step 3: Generate visualizations for selected users."""
    logger.info("Generating visualizations...")

    run_dir = Path(args.run_dir).resolve()
    predictions_path = run_dir / "case_analysis" / "predictions.parquet"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}\nPlease run 'inference' command first."
        )

    df = load_case_results(str(predictions_path))

    selected_users_path = args.selected_users
    if selected_users_path in CASE_SELECTORS:
        selected_users_path = (
            run_dir / "case_analysis" / selected_users_path / "selected_users.json"
        )
    else:
        selected_users_path = Path(selected_users_path)

    if not selected_users_path.exists():
        raise FileNotFoundError(
            f"Selected users file not found: {selected_users_path}\nPlease run 'select' command first or provide a valid path."
        )

    selected_data = json.loads(selected_users_path.read_text())
    selected_users = [u["user_id"] for u in selected_data]

    if args.visualizer not in CASE_VISUALIZERS:
        sys.exit(
            f"Unknown visualizer '{args.visualizer}'. "
            f"Available: {sorted(CASE_VISUALIZERS.keys())}"
        )
    VisualizerClass = CASE_VISUALIZERS.get(args.visualizer)

    logger.info(f"Generating plots for {len(selected_users)} users...")

    output_dir = selected_users_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = VisualizerClass()

    import matplotlib.pyplot as plt

    for user_id in selected_users:
        user_data = get_user_sequence(df, user_id)
        if args.max_seq_len and len(user_data) > args.max_seq_len:
            user_data = user_data.head(args.max_seq_len).reset_index(drop=True)
        fig = visualizer.plot_user(
            user_data,
            user_id,
            output_path=str(output_dir / f"user_{user_id}_heatmap.png"),
        )
        plt.close(fig)

    logger.info(f"Generated {len(selected_users)} individual user visualizations")
    logger.info(f"Figures saved to: {output_dir}")


def _run_inference(rest: list[str]) -> None:
    """Parse ``inference`` args (archived RunConfig + reflective overrides)."""
    rc, case, _ = parse_run_archive(
        rest,
        prog="case_analysis.py inference",
        description="Run model inference and save per-user predictions",
        entry_node="case",
        entry_cls=CaseInferenceConfig,
    )
    cmd_inference(rc, case)


def _run_select(rest: list[str]) -> None:
    """Parse ``select`` args via a reflective parser over CaseSelectConfig."""
    parser = ArgumentParser(
        prog="case_analysis.py select",
        description="Select users from predictions via a selector plugin",
    )
    parser.add_class_arguments(CaseSelectConfig, "case")
    ns = parser.parse_args(rest)
    cmd_select(build_node(CaseSelectConfig, ns["case"]))


def _run_plot(rest: list[str]) -> None:
    """Parse ``plot`` args via a reflective parser over CasePlotConfig."""
    parser = ArgumentParser(
        prog="case_analysis.py plot",
        description="Generate visualizations for selected users",
    )
    parser.add_class_arguments(CasePlotConfig, "case")
    ns = parser.parse_args(rest)
    cmd_plot(build_node(CasePlotConfig, ns["case"]))


def main():
    """Run the case analysis workflow (inference, selection, plotting)."""
    parser = _build_cli()
    ns, rest = parser.parse_known_args()
    ns.handler(rest)


def _build_cli() -> argparse.ArgumentParser:
    """Top-level subcommand router.

    Each subcommand registers a shell parser: argument parsing stays with the
    handler's reflective parser, so its flags (and its own ``-h``) pass through
    untouched, while argparse owns top-level ``--help``, unknown-command
    errors, and usage formatting.
    """
    parser = argparse.ArgumentParser(
        prog="case_analysis.py",
        description="Per-student case analysis for trained KT models.",
    )
    subparsers = parser.add_subparsers(
        dest="command", metavar="{inference,select,plot}", required=True
    )
    for name, handler, help_text in (
        (
            "inference",
            _run_inference,
            "Run model inference and save per-user predictions",
        ),
        ("select", _run_select, "Select representative users from saved predictions"),
        ("plot", _run_plot, "Render heatmaps for selected users"),
    ):
        sub = subparsers.add_parser(name, add_help=False, help=help_text)
        sub.set_defaults(handler=handler)
    return parser


if __name__ == "__main__":
    main()
