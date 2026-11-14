"""Keep shared infrastructure and configuration usable without training dependencies."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "utils.core",
        "utils.core.progress",
        "utils.core.experiment_manager",
        "utils.core.dataset_status",
        "utils.config",
    ],
)
def test_lightweight_imports(module: str, tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        module_name, data_dir = sys.argv[1:]
        blocked = {
            "torch", "numpy", "polars", "pyarrow",
            "utils.data_process", "utils.model_data", "utils.training",
        }
        if module_name.startswith("utils.core"):
            blocked.add("utils.config")

        class ImportGuard(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".")
                       for name in blocked):
                    raise AssertionError(f"Unexpected dependency: {fullname}")

        sys.meta_path.insert(0, ImportGuard())
        module = importlib.import_module(module_name)
        if module_name == "utils.core.dataset_status":
            assert module.list_dataset_statuses(data_dir)
        """
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, module, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
