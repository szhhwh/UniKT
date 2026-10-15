"""Run the normal training entry point with a fixed, isolated job directory."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train
from utils.experiment_manager import ExperimentManager

run_dir = Path(sys.argv.pop(1))


def manager_from_config(rc, exp_type):
    """Bind normal training to an isolated ablation job directory."""
    if rc.data.dataset == "ednet_kt1" and rc.data.sample_size == 5000:
        path = Path(rc.data.data_base_path) / "ednet_kt1/metadata.json"
        metadata = json.loads(path.read_text())
        if not (
            metadata.get("sampled")
            and metadata.get("num_users") == 5000
            and metadata.get("random_seed") == rc.general.seed
            and metadata.get("sampling_config", {}).get("n_samples_actual") == 5000
        ):
            raise ValueError("EdNet ablation requires the validated 5000-user subset.")
    manager = ExperimentManager.from_run_dir(run_dir)
    manager.model_name = rc.experiment.model_name
    manager.dataset_name = rc.data.dataset
    manager.exp_type = exp_type
    manager.is_existing_run = False
    return manager


ExperimentManager.from_run_config = staticmethod(manager_from_config)
train.main()
