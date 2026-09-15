"""MAE (mean absolute error) metric."""

import numpy as np
from sklearn.metrics import mean_absolute_error

from utils.core import register_metric

from .base import Metric


@register_metric("mae")
class MAEMetric(Metric):
    """Mean absolute error.

    Train/val: predicted probabilities (``y_prob``). Test/group: fused group
    scores.
    """

    name = "mae"
    source = "y_prob"

    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        """Return mean absolute error."""
        return float(mean_absolute_error(y_true, y_value))
