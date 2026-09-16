"""ROC AUC metric."""

import numpy as np
from sklearn.metrics import roc_auc_score

from utils.core import register_metric

from .base import Metric


@register_metric("auc")
class AUCMetric(Metric):
    """Area under the ROC curve.

    Train/val: ranking scores (``y_score``). Test/group: fused group scores.
    Omitted when ``y_label`` has fewer than two classes (AUC undefined).
    """

    name = "auc"
    source = "y_score"
    requires_two_classes = True

    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        """Return ROC AUC."""
        return float(roc_auc_score(y_true, y_value))
