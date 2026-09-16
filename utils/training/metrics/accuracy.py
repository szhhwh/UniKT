"""Accuracy metric."""

import numpy as np
from sklearn.metrics import accuracy_score

from utils.core import register_metric

from .base import Metric


@register_metric("acc")
class AccuracyMetric(Metric):
    """Classification accuracy.

    Train/val: model binary predictions (``y_pred``). Test/group: fused group
    scores binarised at 0.5. Defined for a single class (1.0 when all
    correct); omitted only on empty input.
    """

    name = "acc"
    source = "y_pred"
    threshold = 0.5

    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(accuracy_score(y_true, y_value))
