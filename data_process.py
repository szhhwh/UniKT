"""Command-line tool for downloading and processing datasets."""

from dataclasses import fields as dataclass_fields

from jsonargparse import ActionConfigFile, ArgumentParser

from utils.config import DownloadConfig, GeneralConfig, ProcessConfig, RunDataConfig
from utils.core import get_logger, seed_everything
from utils.data_process import get_data_source

logger = get_logger(__name__)


def _add_common_nodes(parser: ArgumentParser) -> None:
    """Add the ``data`` and ``general`` config nodes shared by both subcommands."""
    parser.add_class_arguments(RunDataConfig, "data")
    parser.add_class_arguments(GeneralConfig, "general")
    parser.add_argument("--config", action=ActionConfigFile)


def build_parser() -> ArgumentParser:
    """Build the CLI with ``download`` / ``process`` subcommands."""
    parser = ArgumentParser(prog="data_process.py", description="Data Processing CLI")
    subs = parser.add_subcommands(required=True)

    download = ArgumentParser(prog="data_process.py download")
    _add_common_nodes(download)
    # Flat flags via dataclass: --data_url / --force / --max_retries / --num_threads
    download.add_class_arguments(DownloadConfig)
    subs.add_subcommand("download", download)

    process = ArgumentParser(prog="data_process.py process")
    _add_common_nodes(process)
    # Flat flag via dataclass: --extra
    process.add_class_arguments(ProcessConfig)
    subs.add_subcommand("process", process)

    return parser


class _PartialRC:
    """Lightweight rc view exposing only the ``data`` and ``general`` nodes."""

    def __init__(self, sub_ns):
        self.data = RunDataConfig(
            **{f.name: sub_ns.data[f.name] for f in dataclass_fields(RunDataConfig)}
        )
        self.general = GeneralConfig(
            **{f.name: sub_ns.general[f.name] for f in dataclass_fields(GeneralConfig)}
        )


def cmd_download(rc, ns):
    """Handle `download` subcommand."""
    dp = get_data_source(rc)
    sub_ns = ns[ns.subcommand]
    if getattr(sub_ns, "data_url", None):
        dp.data_url = sub_ns.data_url

    if not dp.data_url:
        raise ValueError(
            "No data_url available for this dataset. Provide --data_url explicitly."
        )

    logger.info(f"Downloading dataset {rc.data.dataset} to {dp.data_folder}")
    dp.fetch_data(
        force_download=getattr(sub_ns, "force", False),
        max_retries=getattr(sub_ns, "max_retries", 3),
        num_threads=getattr(sub_ns, "num_threads", 4),
    )
    dp.save_metadata()
    logger.info("Download complete.")


def cmd_process(rc, ns):
    """Handle `process` subcommand."""
    seed_everything(rc.general.seed, deterministic=False)
    dp = get_data_source(rc)
    sub_ns = ns[ns.subcommand]
    dp.clean_raw_data()
    # The raw interaction frame is no longer needed after cleaning; release it so
    # it doesn't pile up against the (much larger) split-stage intermediates.
    # NOTE: question_data_raw is still consumed by transform_data, so keep it.
    for attr in ("sequence_data_raw",):
        if hasattr(dp, attr):
            setattr(dp, attr, None)

    dp.transform_data()
    # Cleaned data and question metadata are fully consumed by transform_data;
    # release before the memory-heavy split stages.
    for attr in ("cleaned_raw_data", "question_data_raw"):
        if hasattr(dp, attr):
            setattr(dp, attr, None)

    if rc.data.sample_size is not None or rc.data.sample_ratio is not None:
        dp.sample(
            sample_size=rc.data.sample_size,
            sample_ratio=rc.data.sample_ratio,
            sample_strategy=rc.data.sample_strategy,
            attempts_bins=rc.data.sample_attempts_bins,
            correct_bins=rc.data.sample_correct_bins,
        )

    if rc.data.kfold and rc.data.kfold > 1:
        dp.add_kfold_labels(n_splits=rc.data.kfold, test_ratio=rc.data.test_ratio)

    dp.build_split_question_sequence_data()
    dp.build_split_skill_sequence_data()
    if "windowlate" in (sub_ns.extra or []):
        dp.build_windowlate_data()

    dp.save_data()


if __name__ == "__main__":
    import sys

    from utils.config import expand_short_flags, require_dataset

    parser = build_parser()
    ns = parser.parse_args(expand_short_flags(sys.argv[1:]))
    rc = _PartialRC(ns[ns.subcommand])
    require_dataset(rc)  # fail fast; this entry point skips ConfigParser's check

    if ns.subcommand == "download":
        cmd_download(rc, ns)
    else:  # required subcommand, only "process" remains
        cmd_process(rc, ns)
