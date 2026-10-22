"""HDHKT Case Analyzer.

Provides inference-only capabilities for HDHKT model case analysis.
"""

from typing import Any

import numpy as np
import torch

from utils.case_analysis.base_analyzer import BaseCaseAnalyzer
from utils.core import register_analyzer
from utils.data_process import DataSource
from utils.training.runtime_components import RuntimeComponents

from .HDHKT_data import HDHKTModelData
from .HDHKT_model import HDHKT


@register_analyzer("HDHKT")
class HDHKTAnalyzer(BaseCaseAnalyzer):
    """HDHKT-specific case analyzer for inference and visualization."""

    def __init__(self, rc, data_src: DataSource, checkpoint_path: str, **kwargs):
        """Initialize HDHKT analyzer.

        Args:
            rc: RunConfig (OmegaConf DictConfig)
            data_src: Data source instance
            checkpoint_path: Path to model checkpoint
            **kwargs: Forwarded to ``BaseCaseAnalyzer`` (sink, device,
                batch_size).
        """
        super().__init__(rc, data_src, checkpoint_path, **kwargs)

    def build_components(self, rc, data_src) -> RuntimeComponents:
        """Assemble the HDHKT model and user-annotated val dataset."""
        model_data = HDHKTModelData(data_src)
        data_dict = model_data.prepare_data(rc, with_user_ids=True)

        val_dataset = data_dict["val_dataset"]
        hypergraph = data_dict["skill_hypergraph"]
        hetero_graph = data_dict["hetero_graph"]
        question_skill_matrix = data_dict["question_skill_matrix"]

        self.num_questions = data_src.get_metadata("num_questions")
        self.num_skills = data_src.get_metadata("num_skills")

        # Store full question-skill matrix for multi-skill extraction
        self.question_skill_matrix_np = (
            question_skill_matrix.numpy()
        )  # [num_questions, num_skills]

        # Precompute skill->question mapping mask for skill-level probability aggregation.
        # Shape: [num_skills, num_questions]
        self.skill_question_mask = question_skill_matrix.T.float()

        # Cache question -> skill IDs for fast per-question prediction.
        self.question_to_skill_ids = [
            np.where(self.question_skill_matrix_np[q_id] == 1)[0].tolist()
            for q_id in range(self.num_questions)
        ]

        m = rc.model
        model = HDHKT(
            data_metadata=data_src.get_metadata(),
            hetero_metadata=hetero_graph.metadata(),
            hidden_dim=m.hidden_dim,
            n_hop=m.n_hop,
            heads=m.heads,
            lstm_layers=m.lstm_layers,
            dropout=m.dropout,
            history_neighbour=m.history_neighbour,
            att_bound=m.att_bound,
        )

        self.hetero_graph = hetero_graph
        self.hypergraph = hypergraph
        self.question_skill_matrix = question_skill_matrix

        return RuntimeComponents(model=model, val_data=val_dataset)

    def on_device(self, device: torch.device) -> None:
        """Move graph structures and matrices to the inference device."""
        self.hetero_graph = self.hetero_graph.to(device)
        self.hypergraph = self.hypergraph.to(device)
        self.question_skill_matrix = self.question_skill_matrix.to(device)

    def forward_pass(
        self,
        batch_data: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> dict[str, torch.Tensor]:
        """HDHKT forward pass for inference.

        Args:
            batch_data: Tuple of (user_ids, sequence, response, mask, skills);
                user ids are ignored here and consumed by
                ``extract_case_data``.

        Returns:
            Dictionary with y_hat (flattened), y_label (flattened), y_predict,
            full_y_hat (unflattened), and knowledge_states (for visualization)
        """
        _, sequence, response, mask, skills = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        skills = self._move_tensor_to_device(skills)
        mask = self._move_tensor_to_device(mask)

        y_hat_full, skill_hetero_conv, lstm_output = self.model(
            sequence,
            response,
            mask,
            self.hetero_graph,
            self.hypergraph,
            skills,
            return_states=True,
        )

        y_hat, y_label, _ = self._extract_valid_predictions(y_hat_full, response, mask)

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        # Knowledge state definition for heatmap:
        # For each timestep and skill, average model-predicted correctness probability
        # over all questions linked to that skill.
        knowledge_states = self._compute_skill_average_probabilities(
            sequence,
            response,
            mask,
            skill_hetero_conv,
            lstm_output,
        )

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_hat_full": y_hat_full,
            "knowledge_states": knowledge_states,
        }

    def _compute_skill_average_probabilities(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        skill_hetero_conv: torch.Tensor,
        lstm_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-skill mean predicted correctness probabilities.

        For each timestep and skill, average model-predicted correctness probability
        over all questions linked to that skill.

        Returns:
            Tensor of shape [B, S, num_skills] with values in [0, 1].
        """
        B, S = sequence.shape

        all_question_ids = torch.arange(
            self.num_questions, device=sequence.device, dtype=torch.long
        )

        question_probs = self._predict_probabilities_for_questions(
            sequence,
            response,
            mask,
            skill_hetero_conv,
            lstm_output,
            all_question_ids,
        )  # [B, S, Q]

        # Average probabilities by skill using question-skill mapping
        skill_question_mask = self.skill_question_mask.to(sequence.device)  # [K, Q]
        weighted_sum = torch.einsum("bsq,kq->bsk", question_probs, skill_question_mask)
        question_count = skill_question_mask.sum(dim=-1).clamp_min(1.0)
        knowledge_states = weighted_sum / question_count.view(1, 1, -1)

        return knowledge_states

    def _predict_probabilities_for_questions(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        skill_hetero_conv: torch.Tensor,
        lstm_output: torch.Tensor,
        question_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict correctness probabilities for selected questions at each timestep.

        Returns:
            Tensor [B, S, len(question_ids)].
        """
        B, S = sequence.shape
        Q = question_ids.numel()

        if Q == 0:
            return torch.zeros(B, S, 0, device=sequence.device, dtype=lstm_output.dtype)

        H = self.model.hidden_dim

        # Build question representations once
        question_conv_fused, question_embedding_sequence, exercise_emb = (
            self._build_question_representations(sequence, response)
        )

        # Pre-allocate output tensor to avoid list.append + torch.cat overhead
        output = torch.empty(B, S, Q, device=sequence.device, dtype=lstm_output.dtype)

        # Process each question sequentially to avoid memory explosion
        for i, q_id in enumerate(question_ids.tolist()):
            question_embed = question_conv_fused[q_id].view(1, 1, -1).expand(B, S, -1)

            history_question_neighbors = self.model.history_review(
                question_embedding_sequence,
                question_embed,
                exercise_emb,
                mask,
            )

            # [B, S, 1+M, H]
            student_status = torch.cat(
                [lstm_output.unsqueeze(2), history_question_neighbors], dim=2
            )

            skill_ids_list = self.question_to_skill_ids[q_id]
            if not skill_ids_list:
                related_skill_embs = torch.zeros(
                    B, S, 0, H, device=sequence.device, dtype=lstm_output.dtype
                )
            else:
                skill_ids = torch.tensor(
                    skill_ids_list, device=sequence.device, dtype=torch.long
                )
                related_skill_embs = (
                    skill_hetero_conv[skill_ids]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(B, S, -1, -1)
                )

            # [B, S, 1+num_skills, H]
            knowledge_status = torch.cat(
                [question_embed.unsqueeze(2), related_skill_embs], dim=2
            )

            # [B, S]
            logits_q = self.model.general_interaction(
                student_status, knowledge_status, mask
            )
            output[:, :, i] = torch.sigmoid(logits_q)

        return output

    def _build_question_representations(
        self, sequence: torch.Tensor, response: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build fused question representations and exercise embeddings."""
        answers_embedding = self.model.answer_embedding(response)

        question_hyper_conv = self.model.hgnn_conv(
            self.model.question_embedding_hyper.weight, self.hypergraph
        )

        conv = self.model.hetero_conv(
            {
                "question": self.model.question_embedding.weight,
                "skill": self.model.skill_embedding.weight,
                "assignment": self.model.assignment_embedding.weight,
                "template": self.model.template_embedding.weight,
            },
            self.hetero_graph.edge_index_dict,
        )
        question_hetero_conv = conv["question"]

        question_conv_fused = self.model.fuse(question_hetero_conv, question_hyper_conv)
        question_embedding_sequence = question_conv_fused[sequence]

        exercise_emb = torch.cat(
            [question_embedding_sequence, answers_embedding], dim=-1
        )
        exercise_emb = torch.relu(self.model.fc_exercise(exercise_emb))
        exercise_emb = self.model.embedding_dropout(exercise_emb)

        return question_conv_fused, question_embedding_sequence, exercise_emb

    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        """Extract case data from batch outputs.

        Args:
            batch_data: (user_ids, sequences, responses, masks, skills)
            outputs: Output dict from forward_pass

        Returns:
            Dictionary with extracted data (all flattened to valid positions)
        """
        users, sequences, responses, masks, skills = batch_data

        y_label = outputs["y_label"]
        y_predict = outputs["y_predict"]
        y_hat = outputs["y_hat"]
        knowledge_states = outputs["knowledge_states"]  # [B, S, num_skills]

        masks_bool = torch.zeros_like(masks, dtype=torch.bool)
        masks_bool[:, 1:] = masks[:, :-1].bool() & masks[:, 1:].bool()

        valid_indices = masks_bool.view(-1).nonzero(as_tuple=True)[0]

        question_ids_flat = sequences.view(-1)[valid_indices].cpu().numpy()
        user_ids_flat = users[valid_indices // masks.shape[1]].cpu().numpy()

        num_skills = knowledge_states.shape[-1]
        knowledge_states_flat = (
            knowledge_states.view(-1, num_skills)[valid_indices].cpu().numpy()
        )

        selected_skills = (
            skills.reshape(-1, skills.size(-1))[valid_indices].cpu().numpy()
        )
        skill_ids_list = [
            ids[ids != self.num_skills].tolist() for ids in selected_skills
        ]

        return {
            "user_ids": user_ids_flat,
            "question_ids": question_ids_flat,
            "skills": skill_ids_list,
            "labels": y_label.cpu().numpy(),
            "predictions": y_predict.cpu().numpy(),
            "logits": y_hat.cpu().numpy(),
            "mask": masks_bool.view(-1)[valid_indices].cpu().numpy(),
            "knowledge_states": knowledge_states_flat,
        }
