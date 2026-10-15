"""Run the paired five-fold difficulty ablations, preserving training settings."""

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["algebra2005", "assistments09", "bridge2006", "ednet_kt1"]
VARIANTS = {
    "wo_rasch_embedding": (False, True),
    "wo_irt_head": (True, False),
    "wo_both": (False, False),
    "full": (True, True),
}


def atomic_json(path, value):
    """Atomically publish the latest experiment state."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def summarize(out):
    """Collect completed folds and publish final five-fold groups."""
    manifest_path = out / "manifest.json"
    planned_groups = (
        {
            (job["dataset"], job["variant"])
            for job in json.loads(manifest_path.read_text())["jobs"]
        }
        if manifest_path.exists()
        else {(ds, variant) for ds in DATASETS for variant in VARIANTS}
    )
    rows = []
    for ds in DATASETS:
        for variant in VARIANTS:
            if (ds, variant) not in planned_groups:
                continue
            for fold in range(5):
                p = out / ds / variant / f"fold{fold}" / "completed.json"
                if p.exists():
                    rows.append(json.loads(p.read_text()))
    fields = [
        "dataset",
        "variant",
        "fold",
        "auc",
        "acc",
        "rmse",
        "auprc",
        "best_epoch",
        "seconds",
        "epochs",
        "training_seconds",
        "validation_seconds",
        "epoch_total_seconds",
    ]
    with (out / "results_per_fold.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "# AxisKT difficulty ablations",
        "",
        "Test metrics use each fold's best validation-AUC checkpoint.",
        "Only groups with all five folds are reported as final results.",
        "",
        "| Dataset | Ablation | Folds | Test AUC (mean ± sample SD) | ACC | RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    aggregates = []
    for ds in DATASETS:
        for variant in VARIANTS:
            if (ds, variant) not in planned_groups:
                continue
            group = [r for r in rows if r["dataset"] == ds and r["variant"] == variant]
            if len(group) != 5:
                table.append(
                    f"| {ds} | {variant} | {len(group)}/5 | pending | pending | pending |"
                )
                continue
            aggregate = {"dataset": ds, "variant": variant, "n_folds": len(group)}
            for metric in ["auc", "acc", "rmse", "auprc"]:
                vals = [r[metric] for r in group if metric in r]
                if len(vals) == 5:
                    aggregate[f"{metric}_mean"] = statistics.mean(vals)
                    aggregate[f"{metric}_std"] = statistics.stdev(vals)
            aggregates.append(aggregate)
            table.append(
                f"| {ds} | {variant} | 5/5 | {aggregate['auc_mean']:.4f} ± {aggregate['auc_std']:.4f} | {aggregate['acc_mean']:.4f} ± {aggregate['acc_std']:.4f} | {aggregate['rmse_mean']:.4f} ± {aggregate['rmse_std']:.4f} |"
            )
    atomic_json(out / "results_summary.json", aggregates)
    (out / "RESULTS.md").write_text("\n".join(table) + "\n")
    return len(rows)


def prepare(out):
    """Archive source hyperparameters and generate paired job configs."""
    out.mkdir(parents=True, exist_ok=True)
    configs = out / "configs"
    configs.mkdir(exist_ok=True)
    sources = out / "source_configs"
    sources.mkdir(exist_ok=True)
    plans = []
    source_manifest = {}
    for ds in DATASETS:
        source = ROOT / "configs/AxisKT/ablation" / f"ReKTP_{ds}_best.yaml"
        text = source.read_text()
        (sources / source.name).write_text(text)
        base = yaml.safe_load(text)
        removed = base["model"].pop("residual_scale", None)
        base["experiment"]["model_name"] = "AxisKT"
        base["general"] = {
            "seed": 42,
            "device": "cuda",
            "cloud_tracking": False,
            "progress": "none",
        }
        base["data"].update(data_base_path=str(ROOT / "data"), kfold=5)
        base["model"].update(use_global=True, use_local=True, use_forgetting=True)
        source_manifest[ds] = {
            "source": source.name,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "unsupported_residual_scale": removed,
        }
        for fold in range(5):
            for variant, (rasch, irt) in VARIANTS.items():
                config = json.loads(json.dumps(base))
                config["data"]["fold"] = fold
                config["model"].update(use_rasch_embedding=rasch, use_irt_head=irt)
                path = configs / f"{ds}_{variant}_fold{fold}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                plans.append(
                    {
                        "dataset": ds,
                        "variant": variant,
                        "fold": fold,
                        "config": str(path),
                    }
                )
    atomic_json(
        out / "manifest.json",
        {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "sources": source_manifest,
            "jobs": plans,
            "method": "No hyperparameter tuning. Identical model construction and initial RNG draws; only designated mechanism disabled. Rasch removal covers local writes, event embeddings and fused inference. IRT removal predicts theta directly, retaining embedding beta. Validation AUC selects best checkpoint, patience=10; maximum epochs from source configs. Five validation folds share a fixed held-out test set (fold=-1); fold SD is not independent-test-set uncertainty. Includes the full model and three ablations: embedding difficulty, IRT head, and both.",
        },
    )
    (out / "implementation.patch").write_text(
        subprocess.check_output(
            ["git", "diff", "--", "model/AxisKT"], cwd=ROOT, text=True
        )
    )
    return plans


def main():
    """Execute the durable sequential experiment queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "axiskt_difficulty_ablation_20261013",
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.summarize:
        summarize(out)
        return
    if (out / "manifest.json").exists():
        plans = json.loads((out / "manifest.json").read_text())["jobs"]
    else:
        plans = prepare(out)
    env = dict(
        os.environ,
        POLARS_MAX_THREADS="8",
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONUNBUFFERED="1",
    )
    for job in plans:
        run = out / job["dataset"] / job["variant"] / f"fold{job['fold']}"
        run.mkdir(parents=True, exist_ok=True)
        if (run / "completed.json").exists():
            continue
        if (run / "metrics_train.csv").exists():
            raise RuntimeError(
                f"Incomplete existing run: {run}; resume explicitly from its checkpoint or move it before retrying."
            )
        started = time.time()
        completed = summarize(out)
        atomic_json(
            out / "status.json",
            {
                "state": "running",
                "completed": completed,
                "total": len(plans),
                "current": job,
                "started": started,
                "runner_pid": os.getpid(),
            },
        )
        print("START", job, flush=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_axiskt_ablation_job.py"),
            str(run),
            "--config",
            job["config"],
        ]
        with (run / "console.log").open("w") as log:
            process = subprocess.Popen(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            atomic_json(
                out / "current_process.json",
                {
                    "runner_pid": os.getpid(),
                    "training_pid": process.pid,
                    "run_dir": str(run),
                },
            )
            result = process.wait()
        if result:
            atomic_json(
                out / "status.json",
                {
                    "state": "failed",
                    "completed": completed,
                    "total": len(plans),
                    "current": job,
                    "exit_code": result,
                    "console": str(run / "console.log"),
                },
            )
            raise SystemExit(result)
        with (run / "metrics_test.csv").open() as f:
            metrics = list(csv.DictReader(f))[-1]
        record = {
            "dataset": job["dataset"],
            "variant": job["variant"],
            "fold": job["fold"],
            "seconds": time.time() - started,
            "run_dir": str(run),
        }
        record.update({k: float(v) for k, v in metrics.items() if k != "epoch" and v})
        with (run / "metrics_val.csv").open() as f:
            vals = list(csv.DictReader(f))
        best = max(vals, key=lambda r: float(r["auc"]))
        record.update(best_epoch=int(best["epoch"]), best_val_auc=float(best["auc"]))
        atomic_json(run / "completed.json", record)
        print("DONE", record, flush=True)
        summarize(out)
    atomic_json(
        out / "status.json",
        {"state": "complete", "completed": summarize(out), "total": len(plans)},
    )


if __name__ == "__main__":
    main()
