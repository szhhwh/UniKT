"""Tests for ``utils.training.metrics``: base-class gating + per-metric behaviour.

Covers the template-method refactor: every metric inherits identical
data-sufficiency gating from the base (empty / single-class / non-finite
-> key omitted), and ``score`` is a pure function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
import torch

from utils.training.metrics.accumulator import MetricsAccumulator
from utils.training.metrics.accuracy import AccuracyMetric
from utils.training.metrics.auc import AUCMetric
from utils.training.metrics.auprc import AUPRCMetric
from utils.training.metrics.base import Metric, MetricContext
from utils.training.metrics.kappa import KappaMetric
from utils.training.metrics.mae import MAEMetric
from utils.training.metrics.r2 import R2Metric
from utils.training.metrics.rmse import RMSEMetric


def _ctx(
    y_label: np.typing.ArrayLike,
    y_pred: np.typing.ArrayLike | None = None,
    y_score: np.typing.ArrayLike | None = None,
    y_prob: np.typing.ArrayLike | None = None,
    groups: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> MetricContext:
    def a(x: np.typing.ArrayLike | None) -> np.ndarray | None:
        return None if x is None else np.asarray(x)

    return MetricContext(
        phase="val",
        y_label=np.asarray(y_label),
        y_pred=a(y_pred),
        y_score=a(y_score),
        y_prob=a(y_prob),
        groups=groups,
    )


# ---------------------------------------------------------------------------
# base-class gating (_eval)
# ---------------------------------------------------------------------------


class _ConstMetric(Metric):
    """Test double returning a fixed value, to probe ``_eval`` gating."""

    name = "const"
    source = "y_pred"
    value = 1.0

    def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
        return self.value


class TestBaseGating:
    def test_empty_input_omits_key(self) -> None:
        assert _ConstMetric().compute(_ctx(np.array([]), y_pred=np.array([]))) == {}

    def test_single_class_omitted_when_required(self) -> None:
        class M(_ConstMetric):
            requires_two_classes = True

        assert M().compute(_ctx([1, 1, 1], y_pred=[1, 1, 1])) == {}

    def test_two_classes_kept_when_required(self) -> None:
        class M(_ConstMetric):
            requires_two_classes = True

        assert M().compute(_ctx([1, 0, 1], y_pred=[1, 0, 1])) == {"const": 1.0}

    def test_nan_result_omitted(self) -> None:
        class M(_ConstMetric):
            value = float("nan")

        assert M().compute(_ctx([1, 0], y_pred=[1, 0])) == {}

    def test_inf_result_omitted(self) -> None:
        class M(_ConstMetric):
            value = float("inf")

        assert M().compute(_ctx([1, 0], y_pred=[1, 0])) == {}

    def test_score_value_error_omitted(self) -> None:
        class M(Metric):
            name = "boom"
            source = "y_pred"

            def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
                raise ValueError("boom")

        assert M().compute(_ctx([1, 0], y_pred=[1, 0])) == {}

    def test_threshold_binarises_prediction(self) -> None:
        seen: dict[str, np.ndarray] = {}

        class M(Metric):
            name = "t"
            source = "y_pred"
            threshold = 0.5

            def score(self, y_true: np.ndarray, y_value: np.ndarray) -> float:
                seen["v"] = y_value
                return 1.0

        M().compute(_ctx([1, 0], y_pred=[0.2, 0.7]))
        assert np.array_equal(seen["v"], [0.0, 1.0])


# ---------------------------------------------------------------------------
# per-metric values (known input -> known output)
# ---------------------------------------------------------------------------


class TestMetricValues:
    def test_auc_perfect_ranking(self) -> None:
        assert AUCMetric().compute(_ctx([1, 0], y_score=[0.9, 0.1])) == {"auc": 1.0}

    def test_auc_reversed_ranking(self) -> None:
        assert AUCMetric().compute(_ctx([1, 0], y_score=[0.1, 0.9])) == {"auc": 0.0}

    def test_auprc_perfect(self) -> None:
        out = AUPRCMetric().compute(_ctx([1, 0, 1, 0], y_score=[0.9, 0.1, 0.8, 0.2]))
        assert out["auprc"] == 1.0

    def test_acc_partial(self) -> None:
        out = AccuracyMetric().compute(_ctx([1, 0, 1], y_pred=[1, 0, 0]))
        assert out["acc"] == pytest.approx(2 / 3)

    def test_acc_binary_pred_unchanged_by_threshold(self) -> None:
        # y_pred already 0/1 -> binarisation at 0.5 is a no-op
        assert AccuracyMetric().compute(_ctx([1, 0, 1], y_pred=[1, 0, 1])) == {
            "acc": 1.0
        }

    def test_mae(self) -> None:
        out = MAEMetric().compute(_ctx([0, 1], y_prob=[0.1, 0.9]))
        assert out["mae"] == pytest.approx(0.1)

    def test_rmse_perfect(self) -> None:
        assert RMSEMetric().compute(_ctx([0, 1], y_prob=[0.0, 1.0])) == {"rmse": 0.0}

    def test_kappa_perfect_agreement(self) -> None:
        assert KappaMetric().compute(_ctx([1, 0, 1, 0], y_pred=[1, 0, 1, 0])) == {
            "kappa": 1.0
        }

    def test_r2_perfect_correlation(self) -> None:
        out = R2Metric().compute(_ctx([1, 2, 3], y_prob=[0.1, 0.2, 0.3]))
        assert out["r2"] == pytest.approx(1.0)

    def test_r2_zero_variance_returns_zero(self) -> None:
        # constant y_label -> zero variance -> r2 = 0.0 (kept, not omitted)
        out = R2Metric().compute(_ctx([1, 1, 1], y_prob=[0.2, 0.5, 0.8]))
        assert out == {"r2": 0.0}


# ---------------------------------------------------------------------------
# the bugs being fixed: single-class & empty input
# ---------------------------------------------------------------------------


class TestSingleClassAndEmpty:
    def test_auc_single_class_omitted(self) -> None:
        out = AUCMetric().compute(_ctx([1, 1, 1], y_score=[0.5, 0.6, 0.7]))
        assert out == {}

    def test_auprc_single_class_omitted(self) -> None:
        out = AUPRCMetric().compute(_ctx([1, 1, 1], y_score=[0.5, 0.6, 0.7]))
        assert out == {}

    def test_kappa_single_class_omitted(self) -> None:
        out = KappaMetric().compute(_ctx([1, 1, 1], y_pred=[1, 1, 1]))
        assert out == {}

    def test_acc_single_class_defined(self) -> None:
        # acc is well-defined on a single class (all-correct => 1.0)
        out = AccuracyMetric().compute(_ctx([1, 1, 1], y_pred=[1, 1, 1]))
        assert out == {"acc": 1.0}

    def test_mae_single_class_defined(self) -> None:
        out = MAEMetric().compute(_ctx([1, 1, 1], y_prob=[0.9, 0.8, 0.7]))
        assert out["mae"] == pytest.approx(0.2)

    @pytest.mark.parametrize(
        "metric",
        [
            AUCMetric,
            AUPRCMetric,
            AccuracyMetric,
            MAEMetric,
            RMSEMetric,
            KappaMetric,
            R2Metric,
        ],
    )
    def test_empty_input_omits_key(self, metric: type[Metric]) -> None:
        out = metric().compute(
            _ctx(
                np.array([]),
                y_pred=np.array([]),
                y_score=np.array([]),
                y_prob=np.array([]),
            )
        )
        assert out == {}


# ---------------------------------------------------------------------------
# test/group mode (ctx.groups)
# ---------------------------------------------------------------------------


def _group_ctx() -> MetricContext:
    groups = {
        "mean": (np.array([1.0, 0.0]), np.array([0.9, 0.1])),
        "vote": (np.array([1.0, 0.0]), np.array([0.8, 0.2])),
    }
    return MetricContext(phase="test", y_label=np.array([1.0, 0.0]), groups=groups)


class TestGroupMode:
    def test_auc_emits_per_fusion(self) -> None:
        out = AUCMetric().compute(_group_ctx())
        assert set(out) == {"mean_auc", "vote_auc"}
        assert out["mean_auc"] == 1.0

    def test_acc_binarises_group_score(self) -> None:
        out = AccuracyMetric().compute(_group_ctx())
        # scores 0.9 / 0.1 binarise to 1 / 0, matching labels 1 / 0
        assert out["mean_acc"] == 1.0

    def test_single_class_group_omits_auc(self) -> None:
        groups = {"mean": (np.array([1.0, 1.0]), np.array([0.9, 0.8]))}
        ctx = MetricContext(phase="test", y_label=np.array([1.0, 1.0]), groups=groups)
        assert AUCMetric().compute(ctx) == {}


# ---------------------------------------------------------------------------
# end-to-end: MetricsAccumulator over a single-class epoch
# ---------------------------------------------------------------------------


def _batch(
    y_label: Sequence[float],
    y_pred: Sequence[float],
    y_score: Sequence[float],
    y_prob: Sequence[float],
) -> dict[str, torch.Tensor]:
    def t(x: Sequence[float]) -> torch.Tensor:
        return torch.tensor(x, dtype=torch.float32)

    return {
        "y_label": t(y_label),
        "y_predict": t(y_pred),
        "y_score": t(y_score),
        "y_prob": t(y_prob),
    }


class TestAccumulator:
    def test_single_class_epoch_skips_undefined_metrics(self) -> None:
        # all labels 1 -> AUC / AUPRC / Kappa undefined, must be absent while
        # acc / mae / rmse remain present and valid
        accum = MetricsAccumulator()
        accum.reset("val")
        accum.update(
            "val",
            _batch(
                [1, 1, 1, 1], [1, 1, 0, 1], [0.6, 0.7, 0.4, 0.8], [0.6, 0.7, 0.4, 0.8]
            ),
        )
        m = accum.compute("val")
        assert "auc" not in m
        assert "auprc" not in m
        assert "kappa" not in m
        assert {"acc", "mae", "rmse"} <= set(m)

    def test_normal_epoch_emits_all_metrics(self) -> None:
        accum = MetricsAccumulator()
        accum.reset("val")
        accum.update(
            "val",
            _batch(
                [1, 0, 1, 0], [1, 0, 0, 0], [0.9, 0.1, 0.8, 0.2], [0.9, 0.1, 0.8, 0.2]
            ),
        )
        m = accum.compute("val")
        assert {"acc", "auc", "auprc", "mae", "rmse", "r2", "kappa"} <= set(m)


# ---------------------------------------------------------------------------
# fusion helpers + accumulator state machine (direct)
# ---------------------------------------------------------------------------

from utils.training.metrics.accumulator import _order_keys  # noqa: E402
from utils.training.metrics.grouping import _group_scores, _pearson_r2  # noqa: E402


class TestGroupScores:
    def test_mean_averages_each_group_and_safe_divides_unseen(self) -> None:
        out = _group_scores(np.array([0.6, 0.2]), np.array([0, 1]), 3, "mean", 0.5)
        # group 2 has no members: 0/1 instead of 0/0 -> nan
        assert out.tolist() == pytest.approx([0.6, 0.2, 0.0])

    def test_vote_uses_majority_direction_subset(self) -> None:
        y = np.array([0.9, 0.8, 0.1, 0.1, 0.2, 0.9, 0.9, 0.9])
        inverse = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        out = _group_scores(y, inverse, 3, "vote", 0.5)
        # group 0: majority correct -> mean of the two >= threshold members;
        # group 1: majority incorrect -> mean of the two < threshold members.
        assert out.tolist() == pytest.approx([0.85, 0.15, 0.9])

    def test_vote_group_fully_on_majority_side_equals_mean(self) -> None:
        # When every member sits on the majority side, the selected subset is
        # the whole group, so vote degenerates to the plain mean.
        out = _group_scores(np.array([0.9, 0.9]), np.array([0, 0]), 1, "vote", 0.5)
        assert out.tolist() == pytest.approx([0.9])

    def test_all_unanimous_group_includes_every_member(self) -> None:
        y = np.array([0.9, 0.8, 0.1, 0.1, 0.2])
        inverse = np.array([0, 0, 0, 1, 1])
        out = _group_scores(y, inverse, 2, "all", 0.5)
        # group 1 is unanimous (both below) -> whole-group mean, not majority
        # subset (which would also be the whole group here) — and group 0 is
        # mixed -> majority subset, matching vote.
        assert out.tolist() == pytest.approx([0.85, 0.15])

    def test_unknown_fusion_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported fusion_type"):
            _group_scores(np.array([0.6]), np.array([0]), 1, "median", 0.5)

    def test_exact_half_tie_counts_as_majority(self) -> None:
        # 1 of 2 members >= threshold: ratio exactly 0.5 -> majority is True
        # (>=), so vote selects the correct-side member.
        out = _group_scores(np.array([0.9, 0.1]), np.array([0, 0]), 1, "vote", 0.5)
        assert out.tolist() == pytest.approx([0.9])


class TestPearsonR2:
    def test_perfect_linear_correlation(self) -> None:
        assert _pearson_r2([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
        assert _pearson_r2([1, 2, 3], [3, 2, 1]) == pytest.approx(1.0)  # sign lost

    def test_zero_variance_returns_zero(self) -> None:
        assert _pearson_r2([1, 1, 1], [1, 2, 3]) == 0.0
        assert _pearson_r2([1, 2, 3], [1, 1, 1]) == 0.0


def _group_batch(
    y_label: Sequence[float],
    y_score: Sequence[float],
    group_id: Sequence[int],
) -> dict[str, torch.Tensor]:
    def t(values: Sequence[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32)

    y_pred = [1.0 if s >= 0.5 else 0.0 for s in y_score]
    return {
        "y_label": t(y_label),
        "y_predict": t(y_pred),
        "y_score": t(y_score),
        "y_prob": t(y_score),
        "group_id": torch.tensor(group_id),
    }


class TestAccumulatorGroupPath:
    def test_group_id_in_test_phase_produces_fusion_keys(self) -> None:
        accum = MetricsAccumulator()
        accum.reset("test")
        accum.update(
            "test", _group_batch([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], [7, 7, 3, 3])
        )

        metrics = accum.compute("test")
        expected = {
            f"{fusion}_{metric}"
            for fusion in ("mean", "vote", "all")
            for metric in ("acc", "rmse", "r2", "auc", "auprc")
        }
        assert expected <= set(metrics)
        assert metrics["mean_acc"] == 1.0

    def test_inconsistent_group_labels_raise(self) -> None:
        accum = MetricsAccumulator()
        accum.reset("test")
        accum.update("test", _group_batch([1, 0], [0.9, 0.8], [7, 7]))
        with pytest.raises(ValueError, match="Inconsistent labels"):
            accum.compute("test")

    def test_group_id_ignored_in_val_phase(self) -> None:
        accum = MetricsAccumulator()
        accum.reset("val")
        accum.update(
            "val", _group_batch([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], [7, 7, 3, 3])
        )

        metrics = accum.compute("val")
        assert "acc" in metrics  # plain per-instance metrics
        assert not any(key.startswith(("mean_", "vote_", "all_")) for key in metrics)


class TestOrderingAndReset:
    def test_order_keys_train_phase_known_first_then_alpha(self) -> None:
        out = _order_keys({"rmse": 1, "acc": 2, "zz": 3, "auc": 4, "auprc": 5}, "val")
        assert list(out) == ["acc", "auc", "auprc", "rmse", "zz"]

    def test_order_keys_group_phase_fusion_then_metric(self) -> None:
        out = _order_keys(
            {"all_rmse": 3, "vote_acc": 2, "mean_auc": 1, "mean_acc": 4}, "test"
        )
        assert list(out) == ["mean_acc", "mean_auc", "vote_acc", "all_rmse"]

    def test_multi_batch_concatenation_preserves_order(self) -> None:
        accum = MetricsAccumulator()
        accum.reset("val")
        accum.update("val", _batch([1, 0], [1, 0], [0.9, 0.1], [0.9, 0.1]))
        accum.update("val", _batch([1], [1], [0.8], [0.8]))

        ctx = cast(
            MetricContext,
            MetricsAccumulator._build_context("val", accum._accumulators["val"]),
        )
        assert ctx.y_label.tolist() == [1.0, 0.0, 1.0]  # batch order kept

    def test_compute_on_unreset_phase_is_empty(self) -> None:
        assert MetricsAccumulator().compute("train") == {}

    def test_compute_after_reset_without_updates_is_empty(self) -> None:
        accum = MetricsAccumulator()
        accum.reset("train")
        assert accum.compute("train") == {}

    def test_update_on_unknown_phase_auto_resets(self) -> None:
        accum = MetricsAccumulator()
        accum.update("banana", _batch([1, 0], [1, 0], [0.9, 0.1], [0.9, 0.1]))
        assert accum.compute("banana")["acc"] == 1.0
