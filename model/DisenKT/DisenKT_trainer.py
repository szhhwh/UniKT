"""UniKT training integration for DisenKT."""

from dataclasses import field
from typing import Literal

import torch
import torch.nn.functional as F

from utils.config import ModelConfig, create_optimized_dataloader
from utils.core import register_model_config, register_trainer
from utils.training import BaseTrainer, RuntimeComponents


@register_model_config("DisenKT")
class DisenKTConfig(ModelConfig):
    """DisenKT model configuration.

    Args:
        hidden_size: Hidden dimension of the model.
        num_heads: Number of attention heads.
        gcn_layers: Number of concept-graph GCN layers.
        dropout: Dropout probability.
        alpha: Reconstruction loss weight; None = 1.0 (course) / 0.5 (student).
        similarity_weight: Similarity loss weight; None = 1.0 (course) / 0.3 (student).
        kl_anneal_cap: KL annealing cap beta; None = 0.2 (course) / 0.03 (student).
        kl_anneal_steps: Steps for beta to reach kl_anneal_cap.
        response_flip_probability: Flip probability for contrastive negatives.
        similarity_margin: Margin of the hinge similarity loss.
        domain_mode: How domains are split (student-level or course-level).
        source_domain: Source course name; "all" = all other courses.
        source_domains: Source course list, overriding source_domain.
        target_domain: Target course name.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization) for optimizer.
    """

    hidden_size: int = 128
    num_heads: int = 4
    gcn_layers: int = 1
    dropout: float = 0.1
    alpha: float | None = None
    similarity_weight: float | None = None
    kl_anneal_cap: float | None = None
    kl_anneal_steps: int = 50_000
    response_flip_probability: float = 0.6
    similarity_margin: float = 0.3
    domain_mode: Literal["student", "course"] = "student"
    source_domain: str | None = None
    source_domains: list[str] | None = None
    target_domain: str | None = None
    epochs: int = field(default=150)
    batch_size: int = field(default=32)
    learning_rate: float = field(default=1e-3)
    weight_decay: float = field(default=0.0)


def _gaussian_kl(
    mean: torch.Tensor, variance: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    per_step = 0.5 * (mean.square() + variance - 1.0 - variance.log()).sum(dim=-1)
    return per_step[valid].mean()


@register_trainer("DisenKT")
class DisenKTTrainer(BaseTrainer):
    def build_components(self, rc, data_src) -> RuntimeComponents:
        from model.DisenKT.DisenKT_data import DisenKTModelData
        from model.DisenKT.DisenKT_model import DisenKT

        train_domains, valid, test, graphs, num_skills, max_len = DisenKTModelData(
            data_src
        ).prepare_data(rc)
        self._train_datasets = train_domains
        self._active_domain = None
        m = rc.model
        model = DisenKT(
            num_skills=num_skills,
            max_seq_len=max_len,
            batch_size=m.batch_size,
            hidden_size=m.hidden_size,
            num_heads=m.num_heads,
            gcn_layers=m.gcn_layers,
            dropout=m.dropout,
            graphs=graphs,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=m.learning_rate, weight_decay=m.weight_decay
        )
        return RuntimeComponents(
            model=model,
            optimizer=optimizer,
            loss_fn=torch.nn.BCELoss(),
            train_data=train_domains["target"],
            val_data=valid,
            test_data=test,
        )

    def _setup_data_loaders(self) -> None:
        super()._setup_data_loaders()
        self.domain_loaders = {
            name: (
                self.train_data
                if name == "target"
                else create_optimized_dataloader(
                    dataset,
                    batch_size=self.run_config.model.batch_size,
                    shuffle=True,
                    device=self.device_,
                    pin_memory=self.run_config.general.pin_memory,
                )
            )
            for name, dataset in self._train_datasets.items()
        }

    def _process_epoch(self, epoch: int, is_train: bool) -> float:
        if not is_train:
            return super()._process_epoch(epoch, is_train=False)

        self.model.load_global_shared_encoder()
        self.model.train()
        self.metrics_accumulator.reset("train")
        self.callback_manager.on_phase_begin(epoch, "train", trainer=self)

        weighted_loss_sum = 0.0
        total_samples = 0
        last_representations = {}
        batch_idx = 0
        for domain, loader in self.domain_loaders.items():
            self._active_domain = domain
            for batch_data in loader:
                self.callback_manager.on_batch_begin(
                    epoch, batch_idx, "train", trainer=self
                )
                outputs, loss_tensor = self.compute_train_step(batch_data)
                self.metrics_accumulator.update("train", outputs)
                loss = loss_tensor.item()
                n_samples = outputs["y_label"].numel()
                weighted_loss_sum += loss * n_samples
                total_samples += n_samples
                last_representations[domain] = (
                    outputs["latent"]["z_s"] * outputs["mask"].unsqueeze(-1)
                ).detach()
                if self.run_config.general.log_batch_metrics:
                    self.metric_logger.log_batch(
                        phase="train",
                        global_step=self._global_step,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        loss=loss,
                        stage=self._current_stage,
                    )
                self._global_step += 1
                getattr(self.model, f"{domain}_step").add_(1)
                self.callback_manager.on_batch_end(
                    epoch, batch_idx, "train", loss, trainer=self
                )
                batch_idx += 1

        self._active_domain = None
        self.model.aggregate_shared_state(
            last_representations,
            {name: len(dataset) for name, dataset in self._train_datasets.items()},
        )
        metrics = self.metrics_accumulator.compute("train")
        mean_loss = weighted_loss_sum / total_samples if total_samples else 0.0
        metrics["loss"] = mean_loss
        self.metric_logger.log_metrics(
            phase="train",
            metrics=metrics,
            step=epoch + self._metric_step_offset,
            epoch=epoch,
            stage=self._current_stage,
        )
        self.callback_manager.on_phase_end(
            epoch, "train", mean_loss, metrics, trainer=self
        )
        return mean_loss

    def forward_pass(self, batch_data):
        batch = tuple(self._move_tensor_to_device(item) for item in batch_data)
        skill, response, mask = batch
        domain = self._active_domain or "target"
        if self.model.training:
            latent = self.model.forward_domain(
                domain,
                skill,
                response,
                mask,
                self.run_config.model.response_flip_probability,
            )
        else:
            latent = self.model.forward_target(skill, response, mask)
        y_hat, y_label, _ = self._extract_valid_predictions(
            latent["prediction"], response, mask, same_position=True
        )
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)
        output = {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": self._generate_binary_predictions(y_hat, threshold=0.5),
            "y_score": y_hat,
            "y_prob": y_hat,
        }
        if self.model.training:
            output.update(latent=latent, mask=mask, domain=domain)
        return output

    def _compute_loss(self, outputs: dict) -> torch.Tensor:
        m = self.run_config.model
        latent = outputs["latent"]
        mask = outputs["mask"].clone()
        # UniKT's same-position KT convention evaluates only after one interaction.
        mask[:, 0] = False

        reconstruction = self.loss(outputs["y_hat"], outputs["y_label"])
        kl = sum(
            _gaussian_kl(latent[f"mean_{kind}"], latent[f"variance_{kind}"], mask)
            for kind in ("s", "e")
        )
        course_mode = m.domain_mode == "course"
        alpha = m.alpha if m.alpha is not None else (1.0 if course_mode else 0.5)
        similarity_weight = (
            m.similarity_weight
            if m.similarity_weight is not None
            else (1.0 if course_mode else 0.3)
        )
        kl_cap = (
            m.kl_anneal_cap
            if m.kl_anneal_cap is not None
            else (0.2 if course_mode else 0.03)
        )
        domain = outputs["domain"]
        local_step = getattr(self.model, f"{domain}_step").item()
        beta = min(kl_cap, local_step / m.kl_anneal_steps)

        similarity = reconstruction.new_zeros(())
        if self.model.has_global_shared_rep.item():
            global_rep = self.model.global_shared_rep[: latent["z_s"].size(0)]
            discriminator = self.model.discriminators[domain]
            positive = discriminator(latent["z_s"], global_rep)
            negative = discriminator(latent["negative_z_s"], global_rep)
            similarity = F.relu(m.similarity_margin - positive + negative)[mask].mean()
        return alpha * reconstruction + beta * kl + similarity_weight * similarity
