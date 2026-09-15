"""Cohen's Kappa metric."""

import numpy as np
from sklearn.metrics import cohen_kappa_score

from utils.core import register_metric

from .base import Metric


@register_metric("kappa")
class KappaMetric(Metric):
    """Cohen's Kappa (agreement corrected for chance).

    Train/val: model binary predictions (``y_pred``). Test/group: fused group
    scores binarised at 0.5. Omitted when ``y_label`` has fewer than two
    classes (Kappa undefined on a single class).
    """

    name = "kappa"
    source = "y_pred"
    requires_two_classes = True
    threshold = 0.5

    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        """Return Cohen's Kappa."""
        return float(cohen_kappa_score(y_true, y_value))
