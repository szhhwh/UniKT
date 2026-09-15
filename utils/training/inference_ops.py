"""Device and tensor helpers shared by trainers and inference-only consumers.

These methods depend only on a ``device_`` attribute and their inputs, so
any class that manages ``self.device_`` can mix them in without dragging in
the training lifecycle (e.g. case analysis analyzers).
"""

import torch


class InferenceOpsMixin:
    """Mixin providing device resolution and next-item prediction helpers.

    Host classes must set ``self.device_`` before calling the instance
    methods; :meth:`_try_gpu` is static and usable anytime.
    """

    device_: torch.device | None

    @staticmethod
    def _try_gpu() -> torch.device:
        """Get the best available GPU device, falling back to CPU.

        Returns:
            A torch.device, ``"cuda"`` if available else ``"cpu"``.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_tensor_to_device(
        self, tensor: torch.Tensor, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Move a tensor to the trainer's device, optionally casting dtype.

        Args:
            tensor: Input tensor.
            dtype: Target dtype (e.g. ``torch.bool``), optional.

        Returns:
            Tensor moved to device and optionally cast.
        """
        assert self.device_ is not None, "host class must set device_ first"
        result = tensor.to(self.device_)
        if dtype is not None:
            result = result.to(dtype)
        return result

    def _extract_valid_predictions(
        self,
        y_hat_full: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        same_position: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract predictions and labels at valid positions.

        Convention: ``y_hat_full[t]`` predicts ``response[t+1]``
        (next-item). Extraction always follows next-item alignment:
        ``y_hat_full[:, :-1]`` paired with ``response[:, 1:]``, with
        valid mask ``mask[:, :-1] & mask[:, 1:]``.

        When ``same_position=True``, the input uses same-position
        convention (``out[t]`` predicts ``response[t]``). The output
        is left-shifted by one and padded with a placeholder column
        to normalize into next-item view before extraction — no second
        alignment is introduced.

        Args:
            y_hat_full: Model output tensor ``[B, S]``.
            response: Response label tensor ``[B, S]``.
            mask: Valid position mask ``[B, S]`` (bool; int masks are cast).
            same_position: Whether input uses same-position convention
                (``out[t]`` predicts ``response[t]``).

        Returns:
            Tuple ``(y_hat, y_label, valid_mask)`` where:
                y_hat: Predictions at valid positions.
                y_label: Labels at valid positions.
                valid_mask: Mask of valid adjacent pairs.
        """
        # masked_select below requires a bool mask; data files often store
        # int8 masks, so normalize any integer mask on entry.
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        # Normalize same-position input to next-item view
        if same_position:
            y_hat_full = self._pad_to_full_sequence(y_hat_full[:, 1:])

        # Next-item alignment: t-th prediction corresponds to (t+1)-th label
        y_hat_seq = y_hat_full[:, :-1]
        y_label_seq = response.float()[:, 1:]
        mask_curr = mask[:, :-1]
        mask_next = mask[:, 1:]
        valid_mask = mask_curr & mask_next

        # Select valid positions with masking
        y_hat = torch.masked_select(y_hat_seq, valid_mask)
        y_label = torch.masked_select(y_label_seq, valid_mask)

        return y_hat, y_label, valid_mask

    def _pad_to_full_sequence(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Pad a tensor with a trailing zero column, extending ``[B, L]`` to ``[B, L+1]``.

        Used for models (GKT, SAKT, SGKT, MIKT, KQN) whose output
        length is ``S-1`` under next-item convention. The trailing
        placeholder is discarded by ``_extract_valid_predictions``'s
        ``[:, :-1]`` slice.

        Args:
            y_hat: Model output ``[B, L]``.

        Returns:
            Tensor ``[B, L+1]`` with a zero placeholder at the last column.
        """
        dummy = torch.zeros(y_hat.size(0), 1, device=y_hat.device)
        return torch.cat([y_hat, dummy], dim=1)

    def _handle_empty_batch(
        self, y_hat: torch.Tensor, y_label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle an empty batch by raising a descriptive error.

        Args:
            y_hat: Prediction tensor.
            y_label: Label tensor.

        Returns:
            The input ``(y_hat, y_label)`` tuple unchanged.

        Raises:
            ValueError: If the label tensor is empty.
        """
        if y_label.numel() == 0:
            raise ValueError(
                "Empty valid targets in current batch: no positions satisfy "
                "the training mask alignment. Please check data preprocessing/sampling "
                "settings (e.g., min_seq_len, sample_users, batch_size)."
            )
        return y_hat, y_label

    def _generate_binary_predictions(
        self, y_hat: torch.Tensor, threshold: float = 0.0
    ) -> torch.Tensor:
        """Generate binary predictions from logits using a threshold.

        Args:
            y_hat: Prediction logits.
            threshold: Classification threshold (default 0.0).

        Returns:
            Binary prediction tensor (0 or 1).
        """
        assert self.device_ is not None, "host class must set device_ first"
        return torch.ge(y_hat, torch.tensor(threshold).to(self.device_)).to(torch.int)


__all__ = ["InferenceOpsMixin"]
