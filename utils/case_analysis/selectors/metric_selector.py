"""Default user selectors filtering on per-user metric profiles.

All three strategies share the same candidate filter (minimum attempt
count, error-rate and confidence windows) and differ only in how they
sample from the filtered pool.
"""

from typing import Any

import pandas as pd

from ...core import get_logger, register_case_selector
from ..interfaces import UserSelector
from ..user_metrics import compute_user_metrics

logger = get_logger(__name__)


def _filter_candidates(
    user_metrics: pd.DataFrame,
    *,
    min_seq_len: int,
    error_rate_range: tuple[float, float],
    confidence_range: tuple[float, float],
) -> pd.DataFrame:
    """Apply the shared candidate filter to per-user metrics."""
    return user_metrics[
        (user_metrics["num_attempts"] >= min_seq_len)
        & (user_metrics["error_rate"] >= error_rate_range[0])
        & (user_metrics["error_rate"] <= error_rate_range[1])
        & (user_metrics["avg_confidence"] >= confidence_range[0])
        & (user_metrics["avg_confidence"] <= confidence_range[1])
    ].copy()


class _MetricSelectorBase(UserSelector):
    """Shared filter pipeline for metric-based selectors."""

    def _candidates(
        self,
        results: pd.DataFrame,
        *,
        min_seq_len: int,
        error_rate_range: tuple[float, float],
        confidence_range: tuple[float, float],
    ) -> pd.DataFrame:
        """Compute user metrics and apply the shared filter."""
        return _filter_candidates(
            compute_user_metrics(results),
            min_seq_len=min_seq_len,
            error_rate_range=error_rate_range,
            confidence_range=confidence_range,
        )


@register_case_selector("diverse")
class DiverseSelector(_MetricSelectorBase):
    """Sample users from different error-rate bins for coverage."""

    def select(
        self,
        results: pd.DataFrame,
        *,
        min_seq_len: int = 20,
        error_rate_range: tuple[float, float] = (0.1, 0.9),
        confidence_range: tuple[float, float] = (0.3, 0.95),
        max_users: int = 20,
        **options: Any,
    ) -> list:
        """Select users spread across five error-rate bins.

        Args:
            results: Canonical case results DataFrame.
            min_seq_len: Minimum attempt count required.
            error_rate_range: Valid error-rate window.
            confidence_range: Valid average-confidence window.
            max_users: Maximum number of users to select.
            **options: Extra selector options; accepted and ignored.

        Returns:
            List of selected user IDs.
        """
        filtered = self._candidates(
            results,
            min_seq_len=min_seq_len,
            error_rate_range=error_rate_range,
            confidence_range=confidence_range,
        )
        if len(filtered) == 0:
            logger.warning(
                "No users match the filtering criteria. Returning empty list."
            )
            return []
        if len(filtered) <= max_users:
            logger.info(
                f"Only {len(filtered)} users match criteria, returning all of them."
            )
            return filtered["user_id"].tolist()

        filtered["error_bin"] = pd.cut(
            filtered["error_rate"],
            bins=5,
            labels=["very_low", "low", "medium", "high", "very_high"],
        )

        selected = []
        per_bin = max(1, max_users // filtered["error_bin"].nunique())

        for bin_label in filtered["error_bin"].unique():
            bin_users = filtered[filtered["error_bin"] == bin_label]
            n_take = min(per_bin, len(bin_users))
            sampled = (
                bin_users.sample(n=n_take, random_state=42)["user_id"].tolist()
                if len(bin_users) > n_take
                else bin_users["user_id"].tolist()
            )
            selected.extend(sampled)

        # Top up from the remainder when bins did not fill the quota
        if len(selected) < max_users:
            remaining = filtered[~filtered["user_id"].isin(selected)]
            additional = remaining.sample(
                n=min(max_users - len(selected), len(remaining)), random_state=42
            )["user_id"].tolist()
            selected.extend(additional)

        selected = selected[:max_users]
        logger.info(
            f"Selected {len(selected)} users using 'diverse' strategy "
            f"(from {len(filtered)} filtered users)"
        )
        return selected


@register_case_selector("extreme")
class ExtremeSelector(_MetricSelectorBase):
    """Select users with the highest error rates (debugging aid)."""

    def select(
        self,
        results: pd.DataFrame,
        *,
        min_seq_len: int = 20,
        error_rate_range: tuple[float, float] = (0.1, 0.9),
        confidence_range: tuple[float, float] = (0.3, 0.95),
        max_users: int = 20,
        **options: Any,
    ) -> list:
        """Select the ``max_users`` users with the highest error rates.

        Args:
            results: Canonical case results DataFrame.
            min_seq_len: Minimum attempt count required.
            error_rate_range: Valid error-rate window.
            confidence_range: Valid average-confidence window.
            max_users: Maximum number of users to select.
            **options: Extra selector options; accepted and ignored.

        Returns:
            List of selected user IDs.
        """
        filtered = self._candidates(
            results,
            min_seq_len=min_seq_len,
            error_rate_range=error_rate_range,
            confidence_range=confidence_range,
        )
        if len(filtered) == 0:
            logger.warning(
                "No users match the filtering criteria. Returning empty list."
            )
            return []
        selected = filtered.nlargest(max_users, "error_rate")["user_id"].tolist()
        logger.info(
            f"Selected {len(selected)} users using 'extreme' strategy "
            f"(from {len(filtered)} filtered users)"
        )
        return selected


@register_case_selector("random")
class RandomSelector(_MetricSelectorBase):
    """Randomly sample users from the filtered pool."""

    def select(
        self,
        results: pd.DataFrame,
        *,
        min_seq_len: int = 20,
        error_rate_range: tuple[float, float] = (0.1, 0.9),
        confidence_range: tuple[float, float] = (0.3, 0.95),
        max_users: int = 20,
        **options: Any,
    ) -> list:
        """Randomly sample ``max_users`` users from the filtered pool.

        Args:
            results: Canonical case results DataFrame.
            min_seq_len: Minimum attempt count required.
            error_rate_range: Valid error-rate window.
            confidence_range: Valid average-confidence window.
            max_users: Maximum number of users to select.
            **options: Extra selector options; accepted and ignored.

        Returns:
            List of selected user IDs.
        """
        filtered = self._candidates(
            results,
            min_seq_len=min_seq_len,
            error_rate_range=error_rate_range,
            confidence_range=confidence_range,
        )
        if len(filtered) == 0:
            logger.warning(
                "No users match the filtering criteria. Returning empty list."
            )
            return []
        selected = filtered.sample(n=min(max_users, len(filtered)), random_state=42)[
            "user_id"
        ].tolist()
        logger.info(
            f"Selected {len(selected)} users using 'random' strategy "
            f"(from {len(filtered)} filtered users)"
        )
        return selected


__all__ = ["DiverseSelector", "ExtremeSelector", "RandomSelector"]
