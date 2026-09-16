"""R² (squared Pearson correlation) metric."""

import numpy as np

from utils.core import register_metric

from .base import Metric
from .grouping import _pearson_r2


@register_metric("r2")
class R2Metric(Metric):
    """Squared Pearson correlation between truth and prediction.

    Train/val: predicted probabilities (``y_prob``). Test/group: fused group
    scores. Returns 0.0 when either input has zero variance (correlation
    undefined); omitted only on empty input.
    """

    name = "r2"
    source = "y_prob"

    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        """Return squared Pearson correlation."""
        return float(_pearson_r2(y_true, y_value))
