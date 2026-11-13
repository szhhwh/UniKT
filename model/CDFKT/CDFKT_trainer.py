"""CDFKT configuration and UniKT trainer."""

from dataclasses import field

import torch

from utils.config import ModelConfig
from utils.core import get_logger, register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents

logger = get_logger(__name__)


@register_model_config("CDFKT")
class CDFKTConfig(ModelConfig):
    """CDFKT defaults from the source training entry point.

    Args:
        emb_size: Question, response, and knowledge state dimension.
        dropout: Prediction head dropout probability.
        base_intensity: Initial baseline situational intensity.
        alpha_init_scale: Standard deviation of the excitation matrix.
        min_half_life: Minimum initial decay half-life in sequence steps.
        max_half_life: Maximum initial decay half-life in sequence steps.
        sa_layerscale_init: Initial situational residual scale.
        sa_dropout: Situational encoder dropout probability.
        sa_gate_bias: Initial excitation gate bias.
        fusion_dropout: Fusion gate dropout probability.
        d_state: Mamba state dimension.
        d_conv: Mamba causal convolution width.
        expand: Mamba hidden expansion factor.
        max_clip_grad_norm: Maximum gradient norm.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        lr_decay: Optional exponential learning rate decay.
        weight_decay: Adam weight decay.
        batch_size: Training batch size.
    """

    emb_size: int = field(
        default=512,
        metadata={
            "optuna": {
                "type": "categorical",
                "choices": [64, 128, 192, 256, 384, 512],
            }
        },
    )
    dropout: float = field(
        default=0.02479219844695262,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.8}},
    )
    base_intensity: float = field(
        default=0.0035787294084155263,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.5}},
    )
    alpha_init_scale: float = field(
        default=0.01237861633441871,
        metadata={"optuna": {"type": "float", "low": 1e-3, "high": 0.2, "log": True}},
    )
    min_half_life: float = 1.0
    max_half_life: float = 16.0
    sa_layerscale_init: float = field(
        default=0.5243660665573915,
        metadata={"optuna": {"type": "float", "low": 0.01, "high": 2.0, "log": True}},
    )
    sa_dropout: float = field(
        default=0.11921708954769812,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.3}},
    )
    sa_gate_bias: float = field(
        default=3.157777130289173,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 4.0}},
    )
    fusion_dropout: float = field(
        default=0.22306120985336225,
        metadata={"optuna": {"type": "float", "low": 0.0, "high": 0.3}},
    )
    d_state: int = field(
        default=48,
        metadata={
            "optuna": {"type": "categorical", "choices": [8, 12, 16, 24, 32, 48, 64]}
        },
    )
    d_conv: int = field(
        default=2,
        metadata={"optuna": {"type": "categorical", "choices": [2, 3, 4]}},
    )
    expand: int = field(
        default=1,
        metadata={"optuna": {"type": "categorical", "choices": [1, 2, 4]}},
    )
    max_clip_grad_norm: float = 10.0
    epochs: int = 200
    learning_rate: float = field(
        default=0.0004233643821974351,
        metadata={"optuna": {"type": "float", "low": 5e-5, "high": 3e-3, "log": True}},
    )
    lr_decay: float | None = None
    weight_decay: float = field(
        default=4.995497603743331e-6,
        metadata={"optuna": {"type": "float", "low": 1e-8, "high": 3e-4, "log": True}},
    )
    batch_size: int = field(
        default=32,
        metadata={"optuna": {"type": "categorical", "choices": [32, 64, 128]}},
    )


@register_trainer("CDFKT")
class CDFKTTrainer(BaseTrainer):
    def build_components(self, rc, data_src):
        from model.CDFKT.CDFKT_data import CDFKTModelData
        from model.CDFKT.CDFKT_model import CDFKT

        model_data = CDFKTModelData(data_src)
        train, val, test = model_data.prepare_data(rc)
        metadata = data_src.get_metadata()
        m = rc.model
        logger.info("Initializing CDFKT model...")
        model = CDFKT(
            num_q=metadata["num_questions"],
            num_c=metadata["num_skills"],
            question_concept_map=model_data.question_concept_map,
            emb_size=m.emb_size,
            dropout=m.dropout,
            base_intensity=m.base_intensity,
            alpha_init_scale=m.alpha_init_scale,
            min_half_life=m.min_half_life,
            max_half_life=m.max_half_life,
            sa_layerscale_init=m.sa_layerscale_init,
            sa_dropout=m.sa_dropout,
            sa_gate_bias=m.sa_gate_bias,
            fusion_dropout=m.fusion_dropout,
            d_state=m.d_state,
            d_conv=m.d_conv,
            expand=m.expand,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        lr_scheduler = None
        if m.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=m.lr_decay
            )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            lr_scheduler=lr_scheduler,
            train_data=train,
            val_data=val,
            test_data=test,
            max_clip_grad_norm=m.max_clip_grad_norm,
        )

    def forward_pass(self, batch_data):
        question, response, mask = (
            self._move_tensor_to_device(tensor) for tensor in batch_data
        )
        full = self._pad_to_full_sequence(self.model(question, response))
        prediction, label, _ = self._extract_valid_predictions(full, response, mask)
        prediction, label = self._handle_empty_batch(prediction, label)
        return {
            "y_hat": prediction,
            "y_label": label,
            "y_predict": self._generate_binary_predictions(prediction, threshold=0.5),
            "y_score": prediction,
            "y_prob": prediction,
        }
