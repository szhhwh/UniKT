"""AxisKT: private KC scans with a stacked causal-conv global encoder."""

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from model.AxisKT.triton_scan import segmented_scalar_affine_exclusive_scan

_EventFeatures = namedtuple(
    "_EventFeatures",
    [
        "skill_ids",
        "skill_mask",
        "question_vector",
    ],
)


class GlobalConvBlock(nn.Module):
    """Causal depthwise-separable conv block with a residual shell."""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("conv kernel_size must be >= 1")
        # Causal: left-pad only, so each position looks back
        # (kernel_size - 1) * dilation steps and never at the future.
        self.pad = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, dilation=dilation, groups=d_model
        )
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)  # [B, d, N]
        y = F.pad(y, (self.pad, 0))
        y = self.depthwise(y)
        y = self.pointwise(y)
        y = F.gelu(y).transpose(1, 2)  # [B, N, d]
        return self.norm(residual + self.dropout(y))


class AxisKT(nn.Module):
    """Question-level KT with decoupled per-KC storage and global interaction.

    The readout concatenates ``[global_state, local_pre_state,
    event_embedding]``. ``use_global`` / ``use_local`` ablate the global
    conv branch / the local scan branch (an ablated branch emits all-zero
    features and receives no gradient); ``use_forgetting=False`` keeps the
    local writes and scan but uses an identity temporal transition.
    """

    def __init__(
        self,
        data_metadata: dict,
        hidden_dim: int = 128,
        n_blocks: int = 2,
        max_gap_bins: int = 8,
        dropout: float = 0.2,
        conv_kernel_size: int = 3,
        conv_dilation_base: int = 2,
        question_embed_dim: int | None = None,
        use_global: bool = True,
        use_local: bool = True,
        use_forgetting: bool = True,
    ):
        super().__init__()
        if max_gap_bins < 1:
            raise ValueError("max_gap_bins must be at least 1")
        self.num_questions = int(data_metadata["num_questions"])
        self.num_skills = int(data_metadata["num_skills"])
        self.hidden_dim = hidden_dim
        self.max_gap_bins = max_gap_bins
        self.use_global = bool(use_global)
        self.use_local = bool(use_local)
        self.use_forgetting = bool(use_forgetting)

        # Cap the intrinsic question-embedding width at ``hidden_dim``.
        if question_embed_dim is None:
            question_embed_dim = hidden_dim
        if question_embed_dim < 0:
            raise ValueError("question_embed_dim must be non-negative")
        question_embed_dim = min(question_embed_dim, hidden_dim)
        if question_embed_dim == 0:
            self.question_embed = None
            self.question_embed_proj = None
        else:
            self.question_embed = nn.Embedding(self.num_questions, question_embed_dim)
            self.question_embed_proj = (
                None
                if question_embed_dim == hidden_dim
                else nn.Linear(question_embed_dim, hidden_dim, bias=False)
            )
        self.skill_embed = nn.Embedding(
            self.num_skills + 1, hidden_dim, padding_idx=self.num_skills
        )
        self.answer_embed = nn.Embedding(2, hidden_dim)
        self.question_diff = nn.Embedding(self.num_questions, 1)
        self.skill_change = nn.Embedding(
            self.num_skills + 1, hidden_dim, padding_idx=self.num_skills
        )

        # Scalar per-dimension transition: gap-modulated decay (forgetting) and
        # event-conditioned write, one private state per KC.
        self.local_write = nn.Linear(hidden_dim, hidden_dim)
        self.local_init = nn.Linear(hidden_dim, hidden_dim)
        # Learned per-gap, per-state-dimension decay logits.
        self.local_decay_logits = nn.Parameter(
            torch.full((max_gap_bins, hidden_dim), -4.0)
        )
        # Per-KC softmax scores use local-state and skill features.
        self.local_readout = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.global_blocks = nn.ModuleList(
            GlobalConvBlock(
                d_model=hidden_dim,
                dropout=dropout,
                kernel_size=conv_kernel_size,
                dilation=conv_dilation_base**i,
            )
            for i in range(n_blocks)
        )
        self.global_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.global_norm = nn.LayerNorm(hidden_dim)
        # IRT head: logit = a·(θ−β) with θ from ``ability_head`` over the
        # readout features and β from ``question_diff`` of the next question;
        # zero-initialized so training starts at chance (logit 0).
        self.ability_head = nn.Linear(3 * hidden_dim, 1)
        nn.init.zeros_(self.ability_head.weight)
        nn.init.zeros_(self.ability_head.bias)
        self.irt_disc = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.question_diff.weight)
        nn.init.zeros_(self.local_write.weight)
        nn.init.zeros_(self.local_write.bias)
        nn.init.zeros_(self.local_readout.weight)

    def _question_vector(self, questions: torch.Tensor) -> torch.Tensor:
        """Return the per-question vector at ``hidden_dim`` width.

        ``question_embed_dim`` sets the intrinsic width: 0 removes the pathway
        (zeros), a value below ``hidden_dim`` is lifted by a shared projection,
        and values at or above ``hidden_dim`` use a full-width embedding
        directly.
        """
        if self.question_embed is None:
            return torch.zeros(
                *questions.shape,
                self.hidden_dim,
                device=questions.device,
                dtype=self.skill_embed.weight.dtype,
            )
        vector = self.question_embed(questions)
        if self.question_embed_proj is not None:
            vector = self.question_embed_proj(vector)
        return vector

    def _resolve_event_features(
        self,
        questions: torch.Tensor,
        skill_ids: torch.Tensor,
        skill_mask: torch.Tensor,
    ) -> _EventFeatures:
        """Combine event annotations with the shared question embedding."""
        return _EventFeatures(
            skill_ids=skill_ids,
            skill_mask=skill_mask,
            question_vector=self._question_vector(questions),
        )

    def _packed_question_conditioned_readout(
        self,
        packed_state: torch.Tensor,
        packed_skill_embedding: torch.Tensor,
        packed_pos: torch.Tensor,
        packed_valid: torch.Tensor,
        question_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Question-conditioned readout directly in packed space.

        Each occurrence is scored from its private KC state and KC embedding;
        a softmax over the occurrences of a position weights the states before
        summing, reduced with ``scatter_reduce`` over the packed position ids.
        ``question_vector`` supplies the output shape.
        """
        h = self.hidden_dim
        weight = self.local_readout.weight
        w_local = weight[:, :h]
        w_skill = weight[:, h:]
        # Keep attention scores in fp32, as in the inference kernel.
        with torch.autocast(device_type=packed_state.device.type, enabled=False):
            scores = F.linear(packed_state.float(), w_local.float()).squeeze(
                -1
            ) + F.linear(packed_skill_embedding.float(), w_skill.float()).squeeze(-1)
        # Padded occurrences are invalid; keep them out of their position's
        # softmax group.
        scores = scores.masked_fill(~packed_valid, torch.finfo(scores.dtype).min)
        seg_max = torch.scatter_reduce(
            torch.zeros_like(scores),
            1,
            packed_pos,
            scores,
            reduce="amax",
            include_self=False,
        )
        weights = torch.exp(scores - seg_max.gather(1, packed_pos))
        seg_sum = torch.scatter_reduce(
            torch.zeros_like(weights),
            1,
            packed_pos,
            weights,
            reduce="sum",
        )
        weights = weights / seg_sum.gather(1, packed_pos)
        weights = torch.where(packed_valid, weights, torch.zeros_like(weights))
        pooled = packed_state * weights.unsqueeze(-1)
        return torch.scatter_reduce(
            pooled.new_zeros(question_vector.size(0), question_vector.size(1), h),
            1,
            packed_pos.unsqueeze(-1).expand(-1, -1, h),
            pooled,
            reduce="sum",
        )

    def _pack_kc_positions(
        self,
        questions: torch.Tensor,
        mask: torch.Tensor,
        skill_ids: torch.Tensor,
        skill_mask: torch.Tensor,
        kc_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Pack valid KC occurrences of each row in skill-major order.

        Returns the packed skill ids, validity mask, and original position of
        every occurrence, plus the permutation mapping the flattened [S, K]
        grid into the packed layout.
        """
        batch_size, seq_len = questions.shape
        occurrence_mask = skill_mask & mask.unsqueeze(-1)
        max_skills = skill_ids.size(-1)
        flat_skill = skill_ids.flatten(1)
        flat_valid = occurrence_mask.flatten(1)
        positions = (
            torch.arange(seq_len, device=questions.device)
            .view(1, seq_len, 1)
            .expand(batch_size, seq_len, max_skills)
        )
        flat_pos = positions.flatten(1)

        if kc_order is None:
            # Direct model calls retain the original dynamic-sort fallback.
            invalid_key = (self.num_skills + 1) * (seq_len + 1)
            sort_key = flat_skill * (seq_len + 1) + flat_pos
            sort_key = torch.where(flat_valid, sort_key, invalid_key + flat_pos)
            order = torch.argsort(sort_key, dim=1, stable=True)
            packed_length = int(flat_valid.sum(dim=1).max().item())
            order = order[:, :packed_length]
        else:
            order = kc_order

        def gather(values: torch.Tensor) -> torch.Tensor:
            return torch.gather(values, 1, order)

        return gather(flat_skill), gather(flat_valid), gather(flat_pos), order

    def _pack_kc_occurrences(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        skill_ids: torch.Tensor,
        skill_mask: torch.Tensor,
        kc_order: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, seq_len = questions.shape
        max_skills = skill_ids.size(-1)
        (
            packed_skill,
            packed_valid,
            packed_pos,
            order,
        ) = self._pack_kc_positions(questions, mask, skill_ids, skill_mask, kc_order)

        # Only within-sequence time differences matter.
        times = times.reshape(batch_size, seq_len, 1).expand(
            batch_size, seq_len, max_skills
        )
        question_occ = questions.unsqueeze(-1).expand_as(skill_ids)
        response_occ = responses.unsqueeze(-1).expand_as(skill_ids)

        def gather(values: torch.Tensor) -> torch.Tensor:
            return torch.gather(values, 1, order)

        return (
            packed_skill,
            gather(times.flatten(1)),
            gather(question_occ.flatten(1)),
            gather(response_occ.flatten(1)),
            packed_valid,
            order,
            packed_pos,
        )

    def _scan_states(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        event_features: _EventFeatures,
        kc_order: torch.Tensor | None = None,
        packed: tuple[torch.Tensor, ...] | None = None,
        skill_embedding: torch.Tensor | None = None,
        skill_change_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the gap/decay/write pipeline and the segmented scan.

        Returns each occurrence's exclusive scan state — the KC memory
        accumulated through that KC's previous occurrences only — together
        with the shared packed tensors. The occurrence's own decay composes
        only into later reads of the same KC, so the timestamp of the
        interaction being predicted never enters its prediction.
        """
        if packed is None:
            packed = self._pack_kc_occurrences(
                questions,
                responses,
                times,
                mask,
                event_features.skill_ids,
                event_features.skill_mask,
                kc_order=kc_order,
            )
        (
            packed_skill,
            packed_time,
            packed_question,
            packed_response,
            packed_valid,
            order,
            packed_pos,
        ) = packed

        previous_time = torch.zeros_like(packed_time)
        previous_time[:, 1:] = packed_time[:, :-1]
        same_segment = torch.zeros_like(packed_valid)
        same_segment[:, 1:] = (
            packed_valid[:, 1:]
            & packed_valid[:, :-1]
            & (packed_skill[:, 1:] == packed_skill[:, :-1])
        )
        gap = torch.where(same_segment, packed_time - previous_time, packed_time + 1)
        gap_bucket = torch.floor(torch.log2(gap.clamp_min(1).float())).long()
        gap_bucket = gap_bucket.clamp_max(self.max_gap_bins - 1)

        if skill_embedding is None:
            skill_embedding = self.skill_embed(packed_skill)
        if skill_change_embedding is None:
            skill_change_embedding = self.skill_change(packed_skill)
        local_input = (
            self._question_vector(packed_question)
            + skill_embedding
            + self.question_diff(packed_question) * skill_change_embedding
            + self.answer_embed(packed_response)
        )
        if self.use_forgetting:
            decay_table = torch.exp(-F.softplus(self.local_decay_logits))
            decay_codes = gap_bucket
        else:
            # Identity transition: keep local writes and per-KC segmentation,
            # drop only the temporal forgetting.
            decay_table = torch.ones(
                1,
                self.hidden_dim,
                device=packed_skill.device,
                dtype=skill_embedding.dtype,
            )
            decay_codes = torch.zeros_like(gap_bucket)
        init_table = torch.tanh(self.local_init(self.skill_embed.weight))
        bias = torch.tanh(self.local_write(local_input)).unsqueeze(-1)

        packed_state_blocks = segmented_scalar_affine_exclusive_scan(
            None,
            bias,
            packed_skill,
            packed_valid,
            None,
            matrix_table=decay_table,
            matrix_codes=decay_codes,
            init_table=init_table,
            init_codes=packed_skill,
            post_multiply=False,
        )
        packed_state = packed_state_blocks.squeeze(-1)
        return (
            packed_state,
            skill_embedding,
            packed_skill,
            packed_valid,
            packed_pos,
            order,
        )

    def _local_pre_states(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        event_features: _EventFeatures,
        kc_order: torch.Tensor | None = None,
        packed: tuple[torch.Tensor, ...] | None = None,
        skill_embedding: torch.Tensor | None = None,
        skill_change_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = questions.shape
        if not self.use_local:
            return torch.zeros(
                batch_size,
                seq_len,
                self.hidden_dim,
                device=questions.device,
                dtype=self.skill_embed.weight.dtype,
            )
        (
            packed_state,
            skill_embedding,
            _packed_skill,
            packed_valid,
            packed_pos,
            _order,
        ) = self._scan_states(
            questions,
            responses,
            times,
            mask,
            event_features,
            kc_order,
            packed=packed,
            skill_embedding=skill_embedding,
            skill_change_embedding=skill_change_embedding,
        )
        return self._packed_question_conditioned_readout(
            packed_state,
            skill_embedding,
            packed_pos,
            packed_valid,
            event_features.question_vector,
        )

    def _event_embeddings(
        self,
        questions: torch.Tensor,
        event_features: _EventFeatures,
        mask: torch.Tensor,
        kc_order: torch.Tensor | None = None,
        packed: tuple[torch.Tensor, ...] | None = None,
        skill_embedding: torch.Tensor | None = None,
        skill_change_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-position event embedding with packed-occurrence KC pooling.

        The per-KC skill and change embeddings are pooled over each
        position's valid occurrences; masked positions pool to zero.
        """
        if packed is not None:
            packed_skill, packed_valid, packed_pos = packed[0], packed[4], packed[6]
        else:
            (packed_skill, packed_valid, packed_pos) = self._pack_kc_positions(
                questions,
                mask,
                event_features.skill_ids,
                event_features.skill_mask,
                kc_order,
            )[:3]
        if skill_embedding is None:
            skill_embedding = self.skill_embed(packed_skill)
        if skill_change_embedding is None:
            skill_change_embedding = self.skill_change(packed_skill)
        counts = torch.scatter_reduce(
            torch.zeros(
                questions.shape[0], questions.shape[1], device=questions.device
            ),
            1,
            packed_pos,
            packed_valid.float(),
            reduce="sum",
        )
        denom = counts.clamp_min(1.0).unsqueeze(-1)
        pos_index = packed_pos.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        pooled_skill = (
            torch.scatter_reduce(
                packed_valid.new_zeros(
                    questions.shape[0], questions.shape[1], self.hidden_dim
                ).float(),
                1,
                pos_index,
                skill_embedding,
                reduce="sum",
            )
            / denom
        )
        pooled_change = (
            torch.scatter_reduce(
                packed_valid.new_zeros(
                    questions.shape[0], questions.shape[1], self.hidden_dim
                ).float(),
                1,
                pos_index,
                skill_change_embedding,
                reduce="sum",
            )
            / denom
        )
        return (
            event_features.question_vector
            + pooled_skill
            + self.question_diff(questions) * pooled_change
        )

    def _global_history_states_dense(
        self,
        event_embedding: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        global_state = event_embedding + self.answer_embed(responses)
        global_state = global_state.masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.global_blocks:
            global_state = block(global_state)
        return self.global_norm(global_state + self.global_ffn(global_state))

    def _global_history_states_compact(
        self,
        event_embedding: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the existing causal blocks only through each row's last event.

        Interior masked positions stay in the compact stream: the dense blocks
        can produce nonzero states there, which later valid positions may read.
        Only trailing positions are discarded, so valid outputs keep the same
        causal dependencies and all convolution weights remain unchanged.
        """
        batch_size, seq_len, hidden = event_embedding.shape
        active = torch.arange(seq_len, device=mask.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        flat_index = active.flatten().nonzero().flatten()
        row = flat_index // seq_len
        pos = flat_index % seq_len
        starts = lengths.cumsum(0) - lengths

        state = event_embedding.flatten(0, 1).index_select(0, flat_index)
        answers = responses.flatten().index_select(0, flat_index)
        state = state + self.answer_embed(answers)
        state = state.masked_fill(
            ~mask.flatten().index_select(0, flat_index).unsqueeze(1), 0.0
        )

        neighbor_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for block in self.global_blocks:
            depthwise_weight = block.depthwise.weight[:, 0, :]
            mixed = state * depthwise_weight[:, -1]
            for kernel_index in range(depthwise_weight.size(1) - 1):
                offset = (
                    depthwise_weight.size(1) - 1 - kernel_index
                ) * block.depthwise.dilation[0]
                if offset not in neighbor_cache:
                    neighbor_index = starts[row] + (pos - offset).clamp_min(0)
                    neighbor_cache[offset] = (
                        neighbor_index,
                        (pos >= offset).unsqueeze(1),
                    )
                neighbor_index, valid_neighbor = neighbor_cache[offset]
                previous = state.index_select(0, neighbor_index)
                mixed = (
                    mixed
                    + previous * valid_neighbor * depthwise_weight[:, kernel_index]
                )
            if block.depthwise.bias is not None:
                mixed = mixed + block.depthwise.bias
            mixed = F.linear(
                mixed, block.pointwise.weight[:, :, 0], block.pointwise.bias
            )
            state = block.norm(state + block.dropout(F.gelu(mixed)))

        state = self.global_norm(state + self.global_ffn(state))
        dense = event_embedding.new_zeros(batch_size * seq_len, hidden)
        dense.index_copy_(0, flat_index, state)
        return dense.view(batch_size, seq_len, hidden)

    def _global_history_states(
        self,
        event_embedding: torch.Tensor,
        responses: torch.Tensor,
        mask: torch.Tensor,
        compact: bool = False,
    ) -> torch.Tensor:
        if not self.use_global:
            return torch.zeros_like(event_embedding)
        # The CPU batch mask makes the density decision, avoiding a GPU-to-CPU
        # synchronization on every forward. Small inputs keep cuDNN's dense path.
        if compact and event_embedding.is_cuda and event_embedding.numel() >= 250_000:
            seq_len = mask.size(1)
            positions = torch.arange(1, seq_len + 1, device=mask.device)
            lengths = torch.where(mask, positions, 0).amax(dim=1)
            return self._global_history_states_compact(
                event_embedding, responses, mask, lengths
            )
        return self._global_history_states_dense(event_embedding, responses, mask)

    def _irt_term(
        self, features: torch.Tensor, next_questions: torch.Tensor
    ) -> torch.Tensor:
        """IRT logit term ``a·(θ−β)`` — the sole prediction head.

        ``θ`` comes from ``ability_head`` over the readout features and ``β``
        from the shared ``question_diff`` embedding of the predicted question.
        """
        theta = self.ability_head(features).squeeze(-1)
        beta = self.question_diff(next_questions).squeeze(-1)
        return self.irt_disc * (theta - beta)

    def forward(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        skill_ids: torch.Tensor,
        skill_mask: torch.Tensor,
        kc_order: torch.Tensor | None = None,
        kc_inverse: torch.Tensor | None = None,
        compact_global: bool = False,
    ) -> torch.Tensor:
        """Return next-item logits where output[t] predicts response[t+1].

        ``times`` holds per-position interaction times in seconds; only
        within-sequence differences are used, so any consistent offset works.
        The timestamp of a position never enters the prediction of that
        position's response: gap decays compose only into later reads of the
        same KC. ``kc_inverse`` is the dataset-precomputed inverse of
        ``kc_order`` over the full flat slot domain; when omitted the fused
        inference path rebuilds it on the fly. ``compact_global`` selects
        equivalent causal computation over active sequence prefixes; the
        caller determines whether the batch has enough trailing padding.
        """
        if times.shape != questions.shape:
            raise ValueError(
                "times must have shape [batch_size, seq_len] matching questions"
            )
        if (
            skill_ids.ndim != 3
            or skill_ids.shape[:2] != questions.shape
            or skill_ids.size(-1) < 1
        ):
            raise ValueError("skill_ids must have shape [batch_size, seq_len, K]")
        if skill_mask.shape != skill_ids.shape:
            raise ValueError("skill_mask must match skill_ids")
        if kc_order is not None:
            if kc_order.ndim != 2 or kc_order.size(0) != questions.size(0):
                raise ValueError("kc_order must have shape [batch_size, packed_length]")
            if kc_order.size(1) > questions.size(1) * skill_ids.size(-1):
                raise ValueError("kc_order width exceeds the flattened KC width")
            if kc_order.device != questions.device:
                raise ValueError("kc_order must be on the same device as questions")
        if kc_inverse is not None:
            if kc_inverse.shape != (
                questions.size(0),
                questions.size(1) * skill_ids.size(-1),
            ):
                raise ValueError("kc_inverse must have shape [batch_size, S*K]")
            if kc_inverse.device != questions.device:
                raise ValueError("kc_inverse must be on the same device as questions")
        # Interactions are measured relative to the first valid interaction;
        # padding is pinned to zero so log-bucketing never sees negatives.
        first_time = times.masked_fill(~mask, torch.inf).min(dim=1, keepdim=True).values
        times = times - first_time
        times = times.masked_fill(~mask, 0.0)
        event_features = self._resolve_event_features(questions, skill_ids, skill_mask)
        if not torch.is_grad_enabled() and self.use_local and questions.is_cuda:
            # Fused inference path: one Triton kernel produces both the local
            # readout and the event-embedding pooling; training keeps the aten
            # scatter chain so gradient semantics stay untouched.
            from model.AxisKT.packed_readout import fused_readout_event

            (
                packed_state,
                skill_embedding,
                packed_skill,
                packed_valid,
                _packed_pos,
                order,
            ) = self._scan_states(
                questions, responses, times, mask, event_features, kc_order
            )
            skill_change_embedding = self.skill_change(packed_skill)
            question_vector = event_features.question_vector
            if kc_inverse is None:
                kc_inverse = torch.zeros(
                    questions.shape[0],
                    questions.shape[1] * skill_ids.size(2),
                    dtype=torch.long,
                    device=questions.device,
                )
                kc_inverse.scatter_(
                    1,
                    order,
                    torch.arange(order.size(1), device=order.device).expand_as(order),
                )
            slot_valid = (event_features.skill_mask & mask.unsqueeze(-1)).flatten(1)
            local_pre_state, event_embedding = fused_readout_event(
                packed_state,
                skill_embedding,
                skill_change_embedding,
                question_vector,
                questions,
                self.question_diff.weight,
                slot_valid,
                kc_inverse,
                skill_ids.size(2),
                self.local_readout.weight,
            )
        else:
            packed = self._pack_kc_occurrences(
                questions,
                responses,
                times,
                mask,
                event_features.skill_ids,
                event_features.skill_mask,
                kc_order=kc_order,
            )
            skill_embedding = self.skill_embed(packed[0])
            skill_change_embedding = self.skill_change(packed[0])
            event_embedding = self._event_embeddings(
                questions,
                event_features,
                mask=mask,
                kc_order=kc_order,
                packed=packed,
                skill_embedding=skill_embedding,
                skill_change_embedding=skill_change_embedding,
            )
            local_pre_state = self._local_pre_states(
                questions,
                responses,
                times,
                mask,
                event_features,
                kc_order=kc_order,
                packed=packed,
                skill_embedding=skill_embedding,
                skill_change_embedding=skill_change_embedding,
            )
        global_state = self._global_history_states(
            event_embedding, responses, mask, compact=compact_global
        )

        features = torch.cat(
            [
                global_state[:, :-1],
                local_pre_state[:, 1:],
                event_embedding[:, 1:],
            ],
            dim=-1,
        )
        next_questions = questions[:, 1:]
        logits = self._irt_term(features, next_questions)
        logits = logits.masked_fill(~mask[:, 1:], 0.0)
        return torch.cat([logits, logits.new_zeros(logits.size(0), 1)], dim=1)


__all__ = ["AxisKT"]
