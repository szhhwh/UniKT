"""Unified experiment management module.

Provides consistent experiment log directory management supporting
standard training and hyperparameter search workflows.
"""

import argparse
from datetime import datetime
from enum import Enum
from pathlib import Path

from utils.config import RunConfig
from utils.core import get_logger

logger = get_logger(__name__)


class ExperimentType(Enum):
    """Enumeration of supported experiment types."""

    NORMAL = "normal"
    HYPERPARAM_SEARCH = "hyperparam_search"
    EFFICIENCY = "efficiency"


class ExperimentManager:
    """Unified experiment manager.

    Responsibilities:
    1. Create standardised experiment directory structures
    2. Generate consistent naming conventions
    3. Manage experiment subdirectories
    4. Provide factory methods for command-line argument creation

    Example:
        >>> # Method 1: Direct creation
        >>> manager = ExperimentManager(
        ...     exp_type=ExperimentType.NORMAL,
        ...     model_name="GIKT",
        ...     dataset_name="assist09",
        ...     tags=["fold0"]
        ... )
        >>> log_dir = manager.get_log_dir()
        >>> # runs/normal/GIKT_assist09_20241201-120000_fold0/

        >>> # Method 2: From command-line args
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument("--model", type=str, default="GIKT")
        >>> parser.add_argument("--dataset", type=str, default="assist09")
        >>> parser.add_argument("--fold", type=int, default=0)
        >>> args = parser.parse_args()
        >>> manager = ExperimentManager.from_args(args, ExperimentType.NORMAL)
    """

    def __init__(
        self,
        exp_type: ExperimentType,
        model_name: str,
        dataset_name: str,
        base_dir: str = "runs",
        tags: list[str] | None = None,
    ):
        """Initialise the experiment manager.

        Args:
            exp_type: Experiment type (NORMAL / HYPERPARAM_SEARCH).
            model_name: Model name (e.g. GIKT, HDHKT, SQGKT).
            dataset_name: Dataset name (e.g. assist09, assist12, assist17, ednet).
            base_dir: Base directory (default: "runs").
            tags: Optional list of tags (e.g. fold0, bs64).
        """
        self.exp_type = exp_type
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.base_dir = Path(base_dir)
        self.tags = tags or []
        # True when wrapping an already-existing run dir (evaluate/case_analysis);
        # trainers skip re-archiving run_config.yaml in that case to preserve the
        # original training archive.
        self.is_existing_run = False

        # Create experiment directory; disambiguate same-second launches.
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"{model_name}_{dataset_name}_{timestamp}"
        if self.tags:
            exp_name += "_" + "_".join(self.tags)

        self.exp_dir = self._create_unique_dir(self.base_dir / exp_type.value, exp_name)

        logger.debug(f"Experiment directory created: {self.exp_dir}")

    @staticmethod
    def _create_unique_dir(parent: Path, name: str) -> Path:
        """Create ``parent/name`` atomically, appending ``_2``, ``_3``, ... on collision.

        ``mkdir(exist_ok=False)`` is atomic, so concurrent processes colliding
        on the second-resolution timestamp each land in distinct directories.
        """
        candidate = parent / name
        suffix = 1
        while True:
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                suffix += 1
                candidate = parent / f"{name}_{suffix}"
                logger.warning(
                    f"Run directory '{parent / name}' already exists; "
                    f"using '{candidate}' instead"
                )

    def get_log_dir(self) -> str:
        """Return the absolute path to the experiment log directory.

        Returns:
            Absolute path string of the log directory.
        """
        return str(self.exp_dir)

    def create_sub_experiment(self, sub_name: str) -> "ExperimentManager":
        """Create a sub-experiment manager sharing the same timestamp.

        Used for scenarios requiring multiple sub-experiments (e.g. hyperparameter
        search). All sub-experiments share the same parent timestamp.

        Args:
            sub_name: Sub-experiment name (e.g. "trial_0", "full_model", "no_gnn").

        Returns:
            A new ExperimentManager instance pointing to the subdirectory.

        Example:
            >>> parent_manager = ExperimentManager(...)
            >>> child_manager = parent_manager.create_sub_experiment("trial_0")
            >>> # parent: runs/hyperparam_search/GIKT_assist09_20241201-120000/
            >>> # child:  runs/hyperparam_search/GIKT_assist09_20241201-120000/trial_0/
        """
        # Create sub-experiment directory
        sub_dir = self.exp_dir / sub_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Bypass __init__: reuse the parent's attributes and point at a sub-directory.
        sub_manager = ExperimentManager.__new__(ExperimentManager)
        sub_manager.exp_type = self.exp_type
        sub_manager.model_name = self.model_name
        sub_manager.dataset_name = self.dataset_name
        sub_manager.base_dir = self.base_dir
        sub_manager.tags = [*self.tags, sub_name]
        sub_manager.exp_dir = sub_dir
        sub_manager.is_existing_run = self.is_existing_run

        logger.debug(f"Sub-experiment created: {sub_dir}")
        return sub_manager

    def create_subdir(self, name: str) -> Path:
        """Create a subdirectory within the experiment directory.

        Args:
            name: Subdirectory name.

        Returns:
            Path object for the subdirectory.

        Example:
            >>> manager = ExperimentManager(...)
            >>> full_model_dir = manager.create_subdir("full_model")
            >>> # runs/normal/GIKT_assist09_20241201-120000/full_model/
        """
        subdir = self.exp_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Subdirectory created: {subdir}")
        return subdir

    @staticmethod
    def from_args(
        args: argparse.Namespace, exp_type: ExperimentType
    ) -> "ExperimentManager":
        """Create a manager from command-line arguments.

        Automatically extracts model name, dataset name, and common tags
        from the args namespace.

        Args:
            args: Command-line argument object (argparse.Namespace).
            exp_type: Experiment type.

        Returns:
            An ExperimentManager instance.
        """
        model = getattr(args, "model", "unknown")
        dataset = getattr(args, "dataset", "unknown")

        tags = []
        # Extract fold tag
        if hasattr(args, "fold") and args.fold is not None:
            tags.append(f"fold{args.fold}")
        # Extract batch_size tag
        if hasattr(args, "batch_size"):
            tags.append(f"bs{args.batch_size}")

        return ExperimentManager(
            exp_type=exp_type,
            model_name=model,
            dataset_name=dataset,
            base_dir=getattr(args, "base_dir", "runs"),
            tags=tags,
        )

    @staticmethod
    def from_run_config(rc: RunConfig, exp_type: ExperimentType) -> "ExperimentManager":
        """Create a manager from a RunConfig instance.

        Reads model/dataset identity, fold, and batch_size tags from the
        typed config tree.
        """
        tags = []
        if rc.data.fold is not None:
            tags.append(f"fold{rc.data.fold}")
        if hasattr(rc.model, "batch_size"):
            tags.append(f"bs{rc.model.batch_size}")
        return ExperimentManager(
            exp_type=exp_type,
            model_name=rc.experiment.model_name,
            dataset_name=rc.data.dataset,
            base_dir="runs",
            tags=tags,
        )

    @staticmethod
    def from_run_dir(run_dir: str | Path) -> "ExperimentManager":
        """Create an ExperimentManager wrapping an existing run directory.

        Unlike the normal constructor, this does NOT create a new timestamped
        directory. Used for evaluation/inference on already-trained models.

        Args:
            run_dir: Path to an existing run directory.

        Returns:
            ExperimentManager pointing to the existing directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        run_path = Path(run_dir).resolve()
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        manager = ExperimentManager.__new__(ExperimentManager)
        manager.exp_type = ExperimentType.NORMAL
        manager.model_name = ""
        manager.dataset_name = ""
        manager.base_dir = run_path.parent.parent
        manager.tags = []
        manager.exp_dir = run_path
        manager.is_existing_run = True

        logger.debug(f"ExperimentManager bound to existing dir: {run_path}")
        return manager

    def get_experiment_info(self) -> dict:
        """Return a dictionary of experiment metadata.

        Returns:
            Dictionary containing experiment type, model name, dataset name,
            base directory, experiment directory, and tags.
        """
        return {
            "experiment_type": self.exp_type.value,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "base_dir": str(self.base_dir),
            "experiment_dir": str(self.exp_dir),
            "tags": self.tags,
        }


__all__ = ["ExperimentManager", "ExperimentType"]
