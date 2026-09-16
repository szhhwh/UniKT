"""Metric accumulator: collects batch outputs, computes epoch-level metrics.

Public interface (``reset``/``update``/``compute``) is unchanged from the
pre-refactor module. Internally, ``compute`` builds a :class:`MetricContext`
mirroring the old two-branch logic (train/val vs test/group) and fans out to
every auto-discovered metric registered via ``@register_metric``.
"""

import numpy as np
import torch

from utils.core import METRICS

from .base import MetricContext
from .grouping import _group_scores

# Deterministic output ordering — keeps log lines and CSV headers stable across
# runs and bit-aligned with the pre-refactor implementation. Newly added
# metrics fall through to alphabetical order after the known ones.
_TRAIN_ORDER = ["acc", "auc", "auprc", "rmse", "r2"]
_GROUP_METRIC_ORDER = ["acc", "rmse", "r2", "auc", "auprc"]
_FUSION_ORDER = {"mean": 0, "vote": 1, "all": 2}


def _order_keys(out: dict[str, float], phase: str) -> dict[str, float]:
    """Return ``out`` reordered deterministically (no keys added/removed)."""
    if phase == "test" and any(k.startswith(("mean_", "vote_", "all_")) for k in out):
        metric_rank = {m: i for i, m in enumerate(_GROUP_METRIC_ORDER)}

        def group_key(key_name: str) -> tuple[int, int, str]:
            fusion, _, metric = key_name.partition("_")
            return (
                _FUSION_ORDER.get(fusion, 99),
                metric_rank.get(metric, 99),
                key_name,
            )

        return {k: out[k] for k in sorted(out, key=group_key)}

    rank = {m: i for i, m in enumerate(_TRAIN_ORDER)}
    return {k: out[k] for k in sorted(out, key=lambda k: (rank.get(k, 99), k))}


class MetricsAccumulator:
    """Accumulator for batch-level predictions and epoch-level metrics.

    Collects batch outputs across an epoch, then computes all registered
    metrics. Test phase with ``group_id`` produces per-fusion metrics.

    Example:
        >>> accum = MetricsAccumulator()
        >>> accum.reset("train")
        >>> for batch in dataloader:
        ...     outputs = model(batch)
        ...     accum.update("train", outputs)
        >>> metrics = accum.compute("train")
    """

    def __init__(self) -> None:
        """Initialize the accumulator with an empty internal store."""
        self._accumulators: dict[str, dict[str, list[torch.Tensor]]] = {}

    def reset(self, phase: str) -> None:
        """Reset accumulators for a given phase.

        Args:
            phase: Phase name, e.g. ``"train"`` or ``"val"``.
        """
        self._accumulators[phase] = {
            "y_label": [],
            "y_pred": [],
            "y_score": [],
            "y_prob": [],
            "group_id": [],
        }

    def update(self, phase: str, outputs: dict[str, torch.Tensor]) -> None:
        """Update accumulators with a batch of model outputs.

        Args:
            phase: Phase name, e.g. ``"train"`` or ``"val"``.
            outputs: Dict containing ``"y_label"``, ``"y_predict"``,
                ``"y_score"``, ``"y_prob"`` tensors and optional ``"group_id"``.
        """
        if phase not in self._accumulators:
            self.reset(phase)

        accum = self._accumulators[phase]
        accum["y_label"].append(outputs["y_label"].detach().cpu())
        accum["y_pred"].append(outputs["y_predict"].detach().cpu())
        accum["y_score"].append(outputs["y_score"].detach().cpu())
        accum["y_prob"].append(outputs["y_prob"].detach().cpu())
        group_id = outputs.get("group_id")
        if group_id is not None:
            accum["group_id"].append(group_id.detach().cpu())

    def compute(self, phase: str) -> dict[str, float]:
        """Compute epoch-level metrics for the given phase.

        Args:
            phase: Phase name, e.g. ``"train"``, ``"val"``, or ``"test"``.

        Returns:
            Dictionary mapping metric names to values. Empty dict if no data
            has been accumulated for the given phase.
        """
        if phase not in self._accumulators:
            return {}

        accum = self._accumulators[phase]
        if not accum["y_label"]:
            return {}

        ctx = self._build_context(phase, accum)
        if ctx is None:
            return {}

        metrics: dict[str, float] = {}
        for name in METRICS:
            metrics.update(METRICS.get(name)().compute(ctx))
        return _order_keys(metrics, phase)

    @staticmethod
    def _build_context(phase: str, accum: dict[str, list]) -> MetricContext | None:
        """Build the :class:`MetricContext` for ``phase`` from accumulated batches.

        Mirrors the pre-refactor two-branch logic exactly: test phase with
        ``group_id`` runs group-fusion and label-consistency validation
        (raising before any metric runs); otherwise raw per-instance fields
        are used with their native dtype (no extra casting).
        """
        if phase == "test" and accum["group_id"]:
            group_id = torch.cat(accum["group_id"]).numpy()
            y_label_raw = torch.cat(accum["y_label"]).numpy()
            y_score_raw = torch.cat(accum["y_score"]).numpy()

            uniq_groups, inverse = np.unique(group_id, return_inverse=True)
            num_groups = uniq_groups.shape[0]

            # Label: take the first value per group; verify consistency
            first_idx = np.full(num_groups, inverse.shape[0], dtype=np.int64)
            np.minimum.at(first_idx, inverse, np.arange(inverse.shape[0]))
            group_label = y_label_raw[first_idx].astype(np.float64)

            if np.any(y_label_raw != group_label[inverse]):
                raise ValueError(
                    "Inconsistent labels within the same group_id in test evaluation."
                )

            groups = {
                fusion: (
                    group_label,
                    _group_scores(y_score_raw, inverse, num_groups, fusion, 0.5),
                )
                for fusion in ("mean", "vote", "all")
            }
            return MetricContext(phase=phase, y_label=group_label, groups=groups)

        y_label = torch.cat(accum["y_label"]).numpy()
        y_pred = torch.cat(accum["y_pred"]).numpy()
        y_score = torch.cat(accum["y_score"]).numpy()
        y_prob = torch.cat(accum["y_prob"]).numpy()
        return MetricContext(
            phase=phase,
            y_label=y_label,
            y_pred=y_pred,
            y_score=y_score,
            y_prob=y_prob,
        )
