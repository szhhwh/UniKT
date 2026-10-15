"""Trainer and configuration for AxisKT."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("AxisKT")
class AxisKTConfig(ModelConfig):
    """AxisKT configuration.

    Args:
        hidden_dim: Shared event, local-state, and encoder dimension.
        n_blocks: Number of global conv encoder blocks.
        conv_kernel_size: Causal-conv kernel size; each block mixes the
            last ``kernel_size`` positions.
        conv_dilation_base: Dilation base per block; block ``i`` uses
            ``base**i``.
        question_embed_dim: Intrinsic width of the per-question embedding;
            -1 (default) means ``hidden_dim``, 0 removes the pathway, and
            smaller widths are lifted back to ``hidden_dim`` by a shared
            projection. Values above ``hidden_dim`` are capped.
        use_global: Ablate the global causal dilated-conv branch. ``False``
            skips the stacked conv encoder and feeds zero global features to
            the readout; the branch parameters then stay inert.
        use_local: Ablate the local per-KC affine recursion branch. ``False``
            skips the segmented scan and feeds zero local features to the
            readout; the branch parameters then stay inert.
        use_forgetting: Enable the learned gap-conditioned temporal decay in
            the local branch. ``False`` keeps local writes and KC isolation but
            uses an identity transition for a forgetting ablation.
        use_rasch_embedding: Include beta_q modulation in event embeddings and
            local KC writes. False retains the output difficulty head.
        use_irt_head: Use a*(theta-beta_q) logits. False uses the same linear
            ability head directly, retaining Rasch embeddings.
        dropout: Dropout probability.
        epochs: Maximum training epochs.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size: Training batch size.
        max_clip_grad_norm: Maximum gradient clipping norm.
        amp: Enable bf16 autocast for the forward pass; the Triton scan
            keeps fp32 internally. Off by default.
    """

    hidden_dim: int = field(
        default=256,
        metadata={"optuna": {"type": "categorical", "choices": [64, 128, 256]}},
    )
    n_blocks: int = field(
        default=4, metadata={"optuna": {"type": "int", "low": 1, "high": 4}}
    )
    conv_kernel_size: int = field(
        default=3,
        metadata={"optuna": {"type": "categorical", "choices": [3, 5, 7]}},
    )
    conv_dilation_base: int = field(
        default=2,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2]}},
    )
    amp: bool = False
    dropout: float = field(
        default=0.4, metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}}
    )
    question_embed_dim: int = field(
        default=-1,
        metadata={
            "optuna": {"type": "categorical", "choices": [0, 8, 16, 32, 64, 128, 256]}
        },
    )
    use_global: bool = True
    use_local: bool = True
    use_forgetting: bool = True
    use_rasch_embedding: bool = True
    use_irt_head: bool = True
    epochs: int = 100
    learning_rate: float = field(
        default=2.7e-3,
        metadata={"optuna": {"type": "float", "low": 1e-4, "high": 5e-3, "log": True}},
    )
    weight_decay: float = field(
        default=1.4e-4,
        metadata={"optuna": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}},
    )
    batch_size: int = field(
        default=32,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 80, 128]}},
    )
    max_clip_grad_norm: float = field(
        default=5.0,
        metadata={"optuna": {"type": "float", "low": 0.5, "high": 10, "log": True}},
    )


@register_trainer("AxisKT")
class AxisKTTrainer(BaseTrainer):
    """Train AxisKT with one next-item objective per original question."""

    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.AxisKT.AxisKT_data import (
            AxisKTModelData,
            axiskt_packed_collate_fn,
            build_axiskt_model,
        )

        model_data = AxisKTModelData(data_src)
        train_data, val_data, test_data, extra = model_data.prepare_data(rc)

        m = rc.model
        logger.info("Initializing AxisKT model...")
        model = build_axiskt_model(rc, data_src, extra)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            lr_scheduler=None,
            max_clip_grad_norm=m.max_clip_grad_norm,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=axiskt_packed_collate_fn,
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        from model.AxisKT.AxisKT_data import use_compact_global

        questions, responses, times, mask, kc_order, kc_inverse, valid_idx = batch_data
        compact_global = not self.run_config.compile.compile and use_compact_global(
            mask
        )
        questions = self._move_tensor_to_device(questions)
        responses = self._move_tensor_to_device(responses)
        times = self._move_tensor_to_device(times)
        mask = self._move_tensor_to_device(mask)
        kc_order = self._move_tensor_to_device(kc_order)
        kc_inverse = self._move_tensor_to_device(kc_inverse)
        valid_idx = self._move_tensor_to_device(valid_idx)

        use_amp = bool(self.run_config.model.amp)
        with torch.autocast(
            device_type=self.device_.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits_full = self.model(
                questions,
                responses,
                times,
                mask,
                kc_order=kc_order,
                kc_inverse=kc_inverse,
                compact_global=compact_global,
            )
        if use_amp:
            logits_full = logits_full.float()
        logits = logits_full[:, :-1].flatten()[valid_idx]
        labels = responses.float()[:, 1:].flatten()[valid_idx]
        logits, labels = self._handle_empty_batch(logits, labels)
        probabilities = torch.sigmoid(logits)
        return {
            "y_hat": logits,
            "y_label": labels,
            "y_predict": self._generate_binary_predictions(logits, threshold=0.0),
            "y_score": logits,
            "y_prob": probabilities,
        }


__all__ = ["AxisKTConfig", "AxisKTTrainer"]
