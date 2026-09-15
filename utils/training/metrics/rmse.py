"""RMSE metric."""

import numpy as np
from sklearn.metrics import root_mean_squared_error

from utils.core import register_metric

from .base import Metric


@register_metric("rmse")
class RMSEMetric(Metric):
    """Root mean squared error.

    Train/val: predicted probabilities (``y_prob``). Test/group: fused group
    scores.
    """

    name = "rmse"
    source = "y_prob"

    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        """Return root mean squared error."""
        return float(root_mean_squared_error(y_true, y_value))
