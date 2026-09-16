"""AUPRC (average precision) metric."""

import numpy as np
from sklearn.metrics import average_precision_score

from utils.core import register_metric

from .base import Metric


@register_metric("auprc")
class AUPRCMetric(Metric):
    """Area under the precision-recall curve (average precision).

    Train/val: ranking scores (``y_score``). Test/group: fused group scores.
    Omitted when ``y_label`` has fewer than two classes (undefined).
    """

    name = "auprc"
    source = "y_score"
    requires_two_classes = True

    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        """Return average precision (AUPRC)."""
        return float(average_precision_score(y_true, y_value))
