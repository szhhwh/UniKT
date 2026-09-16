"""Tests for InferenceOpsMixin: equivalence with BaseTrainer and helper math."""

import pytest
import torch

from utils.training import BaseTrainer, InferenceOpsMixin


class _Host(InferenceOpsMixin):
    def __init__(self) -> None:
        self.device_ = torch.device("cpu")


def test_mixin_methods_shared_with_base_trainer() -> None:
    for name in (
        "_try_gpu",
        "_move_tensor_to_device",
        "_extract_valid_predictions",
        "_pad_to_full_sequence",
        "_handle_empty_batch",
        "_generate_binary_predictions",
    ):
        assert getattr(InferenceOpsMixin, name) is getattr(BaseTrainer, name), name


def test_mixin_usable_without_trainer() -> None:
    host = _Host()
    assert host._move_tensor_to_device(torch.zeros(1)).device.type == "cpu"
    assert torch.equal(
        host._generate_binary_predictions(torch.tensor([0.5, -0.3])),
        torch.tensor([1, 0]),
    )


def test_extract_valid_predictions_next_item_alignment() -> None:
    host = _Host()
    y_hat_full = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    response = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)

    y_hat, y_label, valid = host._extract_valid_predictions(y_hat_full, response, mask)

    assert y_hat.tolist() == [1.0, 2.0, 5.0, 6.0, 7.0]
    assert y_label.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert valid.tolist() == [[True, True, False], [True, True, True]]


def test_extract_valid_predictions_same_position_normalizes() -> None:
    host = _Host()
    # same-position output: out[t] predicts response[t]; first column is padding
    same_pos = torch.tensor([[9.0, 1.0, 2.0, 3.0], [9.0, 5.0, 6.0, 7.0]])
    response = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)

    y_hat, y_label, _ = host._extract_valid_predictions(
        same_pos, response, mask, same_position=True
    )

    assert y_hat.tolist() == [1.0, 2.0, 5.0, 6.0, 7.0]
    assert y_label.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_pad_to_full_sequence_shape_and_zeros() -> None:
    host = _Host()
    out = host._pad_to_full_sequence(torch.ones(2, 3))
    assert out.shape == (2, 4)
    assert torch.equal(out[:, -1], torch.zeros(2))


def test_handle_empty_batch_raises() -> None:
    host = _Host()
    with pytest.raises(ValueError, match="Empty valid targets"):
        host._handle_empty_batch(torch.zeros(0), torch.zeros(0))


# ---------------------------------------------------------------------------
# dtype/device resolution and extraction edges
# ---------------------------------------------------------------------------


class TestMoveTensorToDevice:
    def test_casts_dtype_when_given(self) -> None:
        host = _Host()
        moved = host._move_tensor_to_device(torch.tensor([0, 1, 2]), dtype=torch.bool)
        assert moved.dtype == torch.bool
        assert moved.tolist() == [False, True, True]

    def test_preserves_dtype_when_none(self) -> None:
        host = _Host()
        moved = host._move_tensor_to_device(
            torch.tensor([1.5, 2.5], dtype=torch.float32)
        )
        assert moved.dtype == torch.float32


class TestTryGpu:
    def test_cpu_branch_when_cuda_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert InferenceOpsMixin._try_gpu() == torch.device("cpu")


class TestBinaryPredictions:
    def test_threshold_boundary_is_inclusive(self) -> None:
        host = _Host()
        # A value exactly equal to the threshold is predicted 1 (>=).
        out = host._generate_binary_predictions(
            torch.tensor([0.5, 0.49]), threshold=0.5
        )
        assert out.tolist() == [1, 0]


class TestExtractValidPredictionsEdges:
    def test_length_one_sequence_yields_empty_valid_predictions(self) -> None:
        host = _Host()
        y_hat_full = torch.tensor([[2.0], [3.0]])
        response = torch.tensor([[1], [0]])
        mask = torch.tensor([[1], [1]], dtype=torch.bool)

        y_hat, y_label, valid = host._extract_valid_predictions(
            y_hat_full, response, mask
        )

        assert y_hat.numel() == 0  # no adjacent pair exists in a length-1 sequence
        assert y_label.numel() == 0
        assert valid.shape == (2, 0)

    def test_integer_mask_normalized_to_bool(self) -> None:
        # Data files often store int8 masks; the helper casts them internally
        # so callers need no manual conversion.
        host = _Host()
        y_hat_full = torch.tensor([[1.0, 2.0, 3.0]])
        response = torch.tensor([[0, 1, 0]])
        int_mask = torch.tensor([[1, 1, 0]], dtype=torch.int8)
        bool_mask = torch.tensor([[True, True, False]])

        out_int = host._extract_valid_predictions(y_hat_full, response, int_mask)
        out_bool = host._extract_valid_predictions(y_hat_full, response, bool_mask)
        assert torch.equal(out_int[0], out_bool[0])
        assert torch.equal(out_int[1], out_bool[1])
        assert out_int[0].tolist() == [1.0]
