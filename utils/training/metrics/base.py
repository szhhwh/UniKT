"""Metric plug-in interface: ``Metric`` ABC + ``MetricContext``.

Each metric subclasses :class:`Metric`, registers via
``@register_metric("name")`` and is auto-discovered (see package ``__init__``).
A metric only declares its data requirements (``source`` /
``requires_two_classes`` / ``threshold``) and implements the pure
:meth:`Metric.score`; the base class handles train/val vs test/group
branching, data-sufficiency gating, and key naming uniformly — so every
metric gets identical, cannot-be-forgotten protection against undefined
inputs (empty / single-class / non-finite → the key is omitted, never a
fake 0.0 or nan).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class MetricContext:
    """Inputs for metric computation, abstracting train/val vs test/group.

    In train/val mode the per-instance raw fields (``y_pred``/``y_score``/
    ``y_prob``) are populated and ``groups`` is ``None``. In test/group mode
    ``groups`` maps each fusion name to ``(group_label, group_score)`` and
    ``y_label`` holds the per-group label (float64); the raw per-instance
    fields are ``None``.

    Attributes:
        phase: Phase name (``"train"`` / ``"val"`` / ``"test"``).
        y_label: Ground-truth labels — per-group (float64) in test/group mode,
            otherwise raw per-instance.
        y_pred: Binary predictions (train/val only).
        y_score: Ranking scores (train/val only).
        y_prob: Predicted probabilities (train/val only).
        groups: ``{fusion: (group_label, group_score)}`` (test/group only).
    """

    phase: str
    y_label: np.ndarray
    y_pred: np.ndarray | None = None
    y_score: np.ndarray | None = None
    y_prob: np.ndarray | None = None
    groups: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


class Metric(ABC):
    """A pluggable evaluation metric (template-method style).

    Subclasses declare their data needs as class attributes and implement
    :meth:`score` — a pure function over already-validated data. The base
    :meth:`compute` then uniformly handles:

    - train/val vs test/group branching (``ctx.groups``),
    - data-sufficiency gating: empty input, single-class ``y_label`` (when
      ``requires_two_classes``), sklearn ``ValueError``, or a non-finite
      result all cause the key to be **omitted** (never a fake 0.0 / nan),
    - optional binarisation of the prediction via ``threshold``,
    - key naming (``name`` in train/val, ``{fusion}_{name}`` in test/group).

    Attributes:
        name: Metric key in train/val mode (and suffix in test/group mode).
        source: ``MetricContext`` field used as the prediction in train/val
            mode (``"y_pred"`` / ``"y_score"`` / ``"y_prob"``).
        requires_two_classes: Skip when ``y_label`` has <2 unique values
            (ranking/agreement metrics are undefined on a single class).
        threshold: If set, binarise the prediction as ``(pred >= threshold)``
            before scoring (classification metrics).
    """

    name: str
    source: str
    requires_two_classes: bool = False
    threshold: float | None = None

    @abstractmethod
    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        """Compute the metric over already-validated data.

        Args:
            y_true: Ground-truth labels (non-empty; >=2 classes when required).
            y_value: Prediction (``ctx.{source}`` in train/val, fused group
                score in test/group; already binarised when ``threshold`` set).

        Returns:
            The metric value. Non-finite results are dropped by the base class.
        """

    def compute(self, ctx: MetricContext) -> dict[str, float]:
        """Compute metric values for the context (template method)."""
        if ctx.groups:
            return {
                f"{fusion}_{self.name}": value
                for fusion, (label, group_score) in ctx.groups.items()
                if (value := self._eval(label, group_score)) is not None
            }
        value = self._eval(ctx.y_label, getattr(ctx, self.source))
        return {self.name: value} if value is not None else {}

    def _eval(self, y_true: np.ndarray, y_value: np.ndarray) -> float | None:
        """Gate on data sufficiency, then delegate to :meth:`score`.

        Returns None (-> key omitted) when the metric is undefined: empty
        input, single-class labels (if required), sklearn ValueError, or a
        non-finite result.
        """
        y_true = np.asarray(y_true)
        if y_true.size == 0:
            return None
        if self.requires_two_classes and np.unique(y_true).size < 2:
            return None
        if self.threshold is not None:
            y_value = (np.asarray(y_value) >= self.threshold).astype(np.float64)
        try:
            value = float(self.score(y_true, y_value))
        except ValueError:
            return None
        return value if math.isfinite(value) else None
