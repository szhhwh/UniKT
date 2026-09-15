<div align="center">

# UniKT

### A unified framework for reproducible knowledge-tracing research

<p>
  From raw interactions to reliable experiments, evaluation, and insight.
</p>

<p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.10"></a>
  <a href="https://pixi.prefix.dev/"><img src="https://img.shields.io/badge/managed%20with-pixi-6C3BF5?style=flat-square" alt="Managed with pixi"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-16a34a?style=flat-square" alt="MIT License"></a>
</p>

<p>
  <a href="https://unikt.lionhao.top/"><img src="https://img.shields.io/badge/Read_the_documentation-2563EB?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Read the documentation"></a>
</p>

<p>
  <a href="#quick-start"><b>Quick start</b></a>
  &nbsp;·&nbsp;
  <a href="#common-commands"><b>Commands</b></a>
  &nbsp;·&nbsp;
  <a href="#web-manager"><b>Web manager</b></a>
</p>

</div>

---

UniKT brings data preparation, training, hyperparameter search, evaluation, efficiency benchmarking, and case analysis into one consistent workflow. Its extensible design lets researchers incorporate new models, data sources, and experiment configurations without disrupting established workflows.

## Highlights

- **Reproducible by default** — Pixi environments, a committed lockfile, archived run configurations, and deterministic execution where supported.
- **One workflow, end to end** — download and process data, run cross-validation, track metrics, evaluate checkpoints, and inspect predictions.
- **Built for research** — Optuna search, SwanLab integration, efficiency benchmarks, and prediction-level case analysis are included.
- **Easy to extend** — add a trainer, configuration, or data source through the framework registry without changing the entry points.

## Quick start

### 1 · Create the environment

```bash
git clone https://github.com/szhhwh/UniKT.git
cd UniKT
pixi install
```

The default environment uses **CUDA 12.8 · Python 3.12 · PyTorch 2.10**. Enter it with `pixi shell`, or run each command through `pixi run` as shown below.

<details>
<summary><b>Need another environment?</b></summary>

```bash
pixi shell -e cpu      # CPU-only
pixi shell -e mamba    # Mamba-based models
pixi shell -e xlstm    # xLSTM-based models
```

</details>

### 2 · Prepare data

```bash
pixi run python data_process.py download -d assistments09
pixi run python data_process.py process -d assistments09
```

Built-in processors cover ASSISTments, EdNet-KT1, Junyi 2015, KDD Cup 2010, MOOC-Radar, NIPS 2020, Practice Anatomy, Slepemapy, and XES3G5M. Check the active options with `pixi run python data_process.py download --help`.

### 3 · Train

```bash
# Train one validation fold.
pixi run python train.py -m GIKT -d assistments09 --fold 0

# Train several folds with the included helper.
PYTHON="pixi run python" bash scripts/run_kfold.sh GIKT "0 1 2 3 4" -d assistments09
```

Each experiment is saved under `runs/` with its resolved configuration, checkpoints, and metrics.

## Common commands

| Task | Command |
| :-- | :-- |
| **Train a model** | `pixi run python train.py -m <model> -d <dataset>` |
| **Search hyperparameters** | `pixi run python optuna_search.py -m <model> -d <dataset>` |
| **Evaluate a saved run** | `pixi run python evaluate.py --evaluate.run_dir runs/normal/<run_id>` |
| **Benchmark efficiency** | `pixi run python efficiency.py -m <model> -d <dataset>` |
| **Inspect model predictions** | `pixi run python case_analysis.py inference --case.run_dir runs/normal/<run_id>` |

> [!TIP]
> Every command exposes its current, code-generated reference through `--help`. For model-specific options, run `pixi run python train.py -m <model> --help`. Commands that restore an archived run (`evaluate.py`, `case_analysis.py inference`) locate the model from `--evaluate.run_dir`/`--case.run_dir`, so pass the run directory for the model-bound reference, e.g. `pixi run python evaluate.py --evaluate.run_dir <run_id> --help`.

### Tracking experiments

Metrics are always written locally. Cloud tracking is enabled by default (`--general.cloud_tracking`, on by default); the backend is chosen by the `KT_TRACKING_BACKEND` env var (`swanlab` | `wandb`, default `swanlab`), and the two are mutually exclusive.

For SwanLab, sign in once before logging runs:

```bash
pixi run swanlab login
```

For W&B, set `WANDB_API_KEY` (or `pixi run wandb login`) and `export KT_TRACKING_BACKEND=wandb`.

For a local-only run, pass `--general.cloud_tracking false`. Backend settings (`KT_SWANLAB_*`, `KT_WANDB_*`, `SWANLAB_MODE`, `LARK_WEBHOOK_URL`) belong in `.env`.

## Web manager

Launch, monitor, and manage experiments in a browser with the optional web interface.

```bash
pixi install -e web
cd web/frontend && npm install && cd ../..
pixi run web-serve
```

Open [localhost:5173](http://localhost:5173), then see the [web manager guide](web/README.md) for development details.

## Acknowledgments

UniKT builds upon the excellent work of [pyKT](https://github.com/pykt-team/pykt-toolkit), which provided valuable references for several knowledge-tracing implementations.

## License

Released under the [MIT License](LICENSE).
