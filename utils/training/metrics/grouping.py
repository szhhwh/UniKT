"""Group-fusion helpers shared across metric computation.

``_group_scores`` aggregates per-instance scores into per-group scores under
mean/vote/all fusion (used by the accumulator to build the test/group context);
``_pearson_r2`` is the squared Pearson correlation used by the R² metric.
Both are migrated verbatim from the pre-refactor metrics module to preserve
numerical behaviour exactly.
"""

import numpy as np


def _group_scores(
    y_score: np.ndarray,
    inverse: np.ndarray,
    num_groups: int,
    fusion_type: str,
    threshold: float,
) -> np.ndarray:
    """Compute aggregated scores per group according to fusion_type.

    Args:
        y_score: Per-instance scores.
        inverse: Inverse mapping from instance to group index.
        num_groups: Total number of groups.
        fusion_type: Aggregation strategy — ``"mean"`` (group mean),
            ``"vote"`` (majority direction subset mean), or ``"all"``
            (whole-group mean for unanimous groups, otherwise majority
            subset mean).
        threshold: Classification threshold.

    Returns:
        Array of aggregated scores, one per group.

    Raises:
        ValueError: If fusion_type is unsupported.
    """
    group_count = np.bincount(inverse, minlength=num_groups).astype(np.float64)
    if fusion_type == "mean":
        group_sum = np.bincount(inverse, weights=y_score, minlength=num_groups).astype(
            np.float64
        )
        return group_sum / np.maximum(group_count, 1.0)

    correct_sum = np.bincount(
        inverse, weights=(y_score >= threshold), minlength=num_groups
    ).astype(np.float64)
    majority = (correct_sum / np.maximum(group_count, 1.0)) >= 0.5

    if fusion_type == "vote":
        selected = np.where(
            majority[inverse], y_score >= threshold, y_score < threshold
        )
        selected_count = np.bincount(
            inverse, weights=selected, minlength=num_groups
        ).astype(np.float64)
        mask = selected | (selected_count == 0)[inverse]
    elif fusion_type == "all":
        uniform = (correct_sum == group_count) | (correct_sum == 0)
        base = np.where(majority[inverse], y_score >= threshold, y_score < threshold)
        mask = base | uniform[inverse]
    else:
        raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    weights = mask.astype(np.float64)
    numerator = np.bincount(
        inverse, weights=y_score * weights, minlength=num_groups
    ).astype(np.float64)
    denominator = np.bincount(inverse, weights=weights, minlength=num_groups).astype(
        np.float64
    )
    return numerator / np.maximum(denominator, 1.0)


def _pearson_r2(y_true: np.typing.ArrayLike, y_pred: np.typing.ArrayLike) -> float:
    """Squared Pearson correlation between truth and prediction.

    R² here is the squared Pearson coefficient (per the paper's definition),
    not the regression coefficient of determination. Returns 0.0 when either
    input has zero variance (correlation is undefined).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    dc_true = y_true - y_true.mean()
    dc_pred = y_pred - y_pred.mean()
    denom = np.sqrt((dc_true**2).sum() * (dc_pred**2).sum())
    if denom == 0:
        return 0.0
    r = (dc_true * dc_pred).sum() / denom
    return float(r * r)
