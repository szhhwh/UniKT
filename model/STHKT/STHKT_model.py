"""STHKT (Spatiotemporal Knowledge Tracing with Topological Hawkes Process) 模型实现

原始论文: Li et al., "STHKT: Spatiotemporal Knowledge Tracing with Topological
Hawkes Process", Expert Systems With Applications 259 (2025) 125248
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

# Ablation variants of the Hawkes attention term (paper Table 4):
#   full            - both similarity terms modulated by the time-decay kernel
#   without_hawkes  - similarity terms added directly, no decay (-nh)
#   nn              - decay term only, no similarity terms (-nn)
#   qkc             - concept-cluster similarity only (-qkc)
#   qs              - student-group similarity only (-qs)
#   none            - no graph, no Hawkes term; plain cross-attention
_HAWKES_MODES = ("full", "without_hawkes", "nn", "qkc", "qs", "none")

# Random-walk metapath over the joint node space: question -> concept ->
# question -> student -> question, twice; hop weights mix the five question
# nodes of the chain by their distance from the start.
_RW_METAPATH = ("contain", "contain_ed", "finish_ed", "finish") * 2
_RW_HOP_WEIGHTS = (1.0, 0.6, 0.4, 0.25, 0.1)

_DIST_EPS = 1e-5


def _build_causal_constants(
    seqlen: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the seqlen-dependent attention constants.

    Returns:
        (loose_mask, strict_mask, position_effect, pair_dist):
            loose_mask includes the diagonal (encoders may see the current
            step); strict_mask excludes it so the Hawkes boost never mixes
            the current step's interaction value into its own prediction;
            position_effect is |i - j|; pair_dist is |i - j| + eps.
    """
    ones = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device)
    idx = torch.arange(seqlen, device=device, dtype=torch.float32)
    diff = (idx.view(-1, 1) - idx.view(1, -1)).abs()
    return (
        ones.tril().view(1, 1, seqlen, seqlen),
        ones.tril(diagonal=-1).view(1, 1, seqlen, seqlen),
        diff.view(1, 1, seqlen, seqlen),
        (diff + _DIST_EPS).view(1, 1, seqlen, seqlen),
    )


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax restricted to positions where ``mask`` is True.

    The trailing mask multiply hard-zeros residual leakage through the
    -1e32 tail.
    """
    scores = scores.masked_fill(~mask, -1e32)
    scores = F.softmax(scores, dim=-1)
    return scores * mask.float()


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma, position_effect):
    """Monotonic attention with a monotonicity-inducing distance penalty.

    Args:
        q/k/v: [BS, n_heads, seq_len, d_k] tensors.
        d_k: Per-head dimension.
        mask: Additive-attention mask (True = attend).
        dropout: Dropout module applied to the attention distribution.
        zero_pad: Shift attention rows down by one so position ``t`` attends
            with the distribution of ``t-1`` (keeps the answer at ``t`` out).
        gamma: Per-head monotonicity parameter.
        position_effect: Cached |i - j| matrix [1, 1, S, S].

    Returns:
        scores: The (shifted, dropout-ed) attention distribution
            [BS, n_heads, seq_len, seq_len]; the caller aggregates values.
    """
    device = q.device
    seqlen = q.size(2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Monotonic decay: expected attention mass still ahead of each position,
    # scaled by positional distance and a learnable per-head rate. Detached:
    # the penalty shapes the distribution but receives no gradient itself.
    with torch.no_grad():
        scores_ = _masked_softmax(scores, mask)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    gamma = -1.0 * F.softplus(gamma).unsqueeze(0)
    total_effect = torch.clamp(
        torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
    )
    scores = scores * total_effect

    scores = _masked_softmax(scores, mask)

    if zero_pad:
        pad_zero = torch.zeros(scores.size(0), scores.size(1), 1, seqlen, device=device)
        scores = torch.cat([pad_zero, scores[:, :, :-1, :]], dim=2)

    return dropout(scores)


def _sequence_similarity(h, n_heads, causal_mask):
    """Row-softmaxed self-similarity of a sequence representation.

    Args:
        h: [BS, seq_len, D] structural features from a graph module.
        n_heads: Broadcast the single shared similarity matrix to all heads.
        causal_mask: [seq_len, seq_len] strictly-lower-triangular mask.

    Returns:
        similarity: [BS, n_heads, seq_len, seq_len].
    """
    sim = torch.bmm(h, h.transpose(1, 2))
    sim = _masked_softmax(sim, causal_mask)
    return sim.unsqueeze(1).expand(-1, n_heads, -1, -1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with an optional Hawkes structural term
    (paper Eq. 9/13/14/15)."""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        hawkes_mode: str = "none",
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        self.hawkes_mode = hawkes_mode

        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

        if hawkes_mode != "none":
            # alpha/alpha2 act on the (qs, qkc) similarity matrices
            # respectively in decayed modes; without_hawkes swaps the pair.
            self.alpha = nn.Linear(max_seq_len, max_seq_len)
            self.alpha2 = nn.Linear(max_seq_len, max_seq_len)
            self.beta = nn.Linear(max_seq_len, max_seq_len)

        self._reset_parameters()

        # Seqlen-dependent constants, rebuilt on first forward.
        self.register_buffer("_causal_mask_strict", None, persistent=False)
        self.register_buffer("_pair_dist", None, persistent=False)
        self.register_buffer("_position_effect", None, persistent=False)
        self._const_seqlen = 0

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        constant_(self.k_linear.bias, 0.0)
        constant_(self.v_linear.bias, 0.0)
        constant_(self.out_proj.bias, 0.0)
        if self.hawkes_mode != "none":
            xavier_uniform_(self.alpha.weight)
            xavier_uniform_(self.alpha2.weight)
            xavier_uniform_(self.beta.weight)

    def _build_constants(self, seqlen: int, device: torch.device) -> None:
        _, strict, position_effect, pair_dist = _build_causal_constants(seqlen, device)
        self._causal_mask_strict = strict
        self._position_effect = position_effect
        self._pair_dist = pair_dist
        self._const_seqlen = seqlen

    def _hawkes_scores(
        self, scores: torch.Tensor, h_qkc: torch.Tensor, h_qs: torch.Tensor
    ) -> torch.Tensor:
        bs = scores.shape[0]
        causal = self._causal_mask_strict.expand(bs, self.n_heads, -1, -1)
        log_dist = torch.log(self._pair_dist).expand(bs, self.n_heads, -1, -1)
        strict = self._causal_mask_strict[0, 0]

        sim_qkc = _sequence_similarity(h_qkc, self.n_heads, strict)
        sim_qs = (
            sim_qkc
            if h_qkc is h_qs
            else _sequence_similarity(h_qs, self.n_heads, strict)
        )

        if self.hawkes_mode == "without_hawkes":
            boost = self.alpha(sim_qkc) + self.alpha2(sim_qs)
        else:
            decay = torch.exp(-log_dist - self.beta(log_dist))
            if self.hawkes_mode == "nn":
                boost = -self.beta(log_dist)
            elif self.hawkes_mode == "qkc":
                boost = self.alpha2(sim_qkc) * decay
            elif self.hawkes_mode == "qs":
                boost = self.alpha(sim_qs) * decay
            else:  # full
                boost = (self.alpha(sim_qs) + self.alpha2(sim_qkc)) * decay
        return scores + boost * causal.float()

    def forward(self, q, k, v, mask, zero_pad, h_qkc=None, h_qs=None):
        bs = q.size(0)
        seqlen = q.size(1)

        if (
            self._causal_mask_strict is None
            or seqlen != self._const_seqlen
            or self._causal_mask_strict.device != q.device
        ):
            self._build_constants(seqlen, q.device)

        # q is k at every call site; capture the identity before
        # k_linear overwrites it.
        same_qk = q is k
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.d_feature)
        q = (
            k
            if same_qk
            else self.k_linear(q).view(bs, -1, self.n_heads, self.d_feature)
        )
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.d_feature)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(
            q,
            k,
            v,
            self.d_feature,
            mask,
            self.dropout,
            zero_pad,
            self.gammas,
            self._position_effect,
        )
        if self.hawkes_mode != "none":
            scores = self._hawkes_scores(scores, h_qkc, h_qs)

        output = torch.matmul(scores, v)
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)


class TransformerLayer(nn.Module):
    """Single transformer block wrapping :class:`MultiHeadAttention`.

    ``see_current`` selects the causal mask variant: True allows each
    position to attend to itself (question / interaction encoders), False
    drops the diagonal and shifts the attention rows so the cross stream at
    position ``t`` never sees the answer at ``t``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        hawkes_mode: str = "none",
        max_seq_len: int = 200,
    ):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model,
            d_model // n_heads,
            n_heads,
            dropout,
            hawkes_mode=hawkes_mode,
            max_seq_len=max_seq_len,
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.register_buffer("_causal_mask_loose", None, persistent=False)
        self.register_buffer("_causal_mask_strict", None, persistent=False)
        self._const_seqlen = 0

    def _build_constants(self, seqlen: int, device: torch.device) -> None:
        loose, strict, _, _ = _build_causal_constants(seqlen, device)
        self._causal_mask_loose = loose
        self._causal_mask_strict = strict
        self._const_seqlen = seqlen

    def forward(
        self,
        see_current: bool,
        query,
        key,
        values,
        apply_pos=True,
        h_qkc=None,
        h_qs=None,
    ):
        seqlen = query.size(1)
        if (
            self._causal_mask_strict is None
            or seqlen != self._const_seqlen
            or self._causal_mask_strict.device != query.device
        ):
            self._build_constants(seqlen, query.device)
        src_mask = self._causal_mask_loose if see_current else self._causal_mask_strict

        query2 = self.masked_attn_head(
            query,
            key,
            values,
            mask=src_mask,
            zero_pad=not see_current,
            h_qkc=h_qkc,
            h_qs=h_qs,
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class HeteroGCN(nn.Module):
    """Two-kernel single-layer graph convolution (paper Eq. 6/11/12).

    One kernel per subgraph (question–concept, question–student); only
    question-node outputs are consumed downstream.
    """

    def __init__(
        self,
        d_model: int,
        emb_kc: nn.Parameter,
        emb_s: nn.Parameter,
        gcn_qkc: dict,
        gcn_qs: dict,
        scatter: str = "mean",
    ):
        super().__init__()
        self.w_qkc = nn.Linear(d_model, 2 * d_model, bias=False)
        self.w_qs = nn.Linear(d_model, 2 * d_model, bias=False)
        # Plain list so the shared Parameters stay registered once, on the
        # host model's embedding layers.
        self._emb_refs = [emb_kc, emb_s]
        self.scatter = scatter

        for name, g in (("qkc", gcn_qkc), ("qs", gcn_qs)):
            nq = g["ptr"].shape[0] - 1
            adj = torch.sparse_csr_tensor(
                crow_indices=torch.from_numpy(g["ptr"]).long(),
                col_indices=torch.from_numpy(g["idx"]).long(),
                values=torch.from_numpy(g["w"]),
                size=(nq, g["no"]),
            )
            self.register_buffer(f"_{name}_adj", adj, persistent=False)
            self.register_buffer(
                f"_{name}_self_w", torch.from_numpy(g["self_w"]), persistent=False
            )

    def _static_agg(
        self, emb: nn.Parameter, proj: nn.Linear, name: str
    ) -> torch.Tensor:
        """Aggregate ``emb @ W^T`` through the normalized adjacency."""
        return torch.sparse.mm(getattr(self, f"_{name}_adj"), proj(emb))

    def _scatter_to_groups(self, x: torch.Tensor, seq_q: torch.Tensor) -> torch.Tensor:
        """Group sequence features by (batch, question) and re-expand.

        ``mean`` divides each group by its occurrence count within the
        sequence.
        """
        bs, seqlen, d_model = x.shape
        keys = seq_q.reshape(-1) + torch.arange(bs, device=x.device).mul(
            seq_q.max() + 1
        ).view(-1, 1).expand(bs, seqlen).reshape(-1)
        uniq, inverse = torch.unique(keys, return_inverse=True)
        grouped = x.reshape(-1, d_model).new_zeros(len(uniq), d_model)
        grouped.index_add_(0, inverse, x.reshape(-1, d_model))
        if self.scatter == "mean":
            counts = torch.bincount(inverse, minlength=len(uniq)).clamp(min=1)
            grouped = grouped / counts.unsqueeze(-1)
        return grouped[inverse].view(bs, seqlen, d_model)

    def forward(self, x: torch.Tensor, seq_q: torch.Tensor):
        """Run both convolution kernels.

        Args:
            x: Question embeddings [BS, S, D] (layer-normalized).
            seq_q: Question ids [BS, S] in graph question space (0 = padding).

        Returns:
            (h_qkc, h_qs): structural features [BS, S, 2D] per subgraph.
        """
        gX = self._scatter_to_groups(x, seq_q)

        agg_kc = self._static_agg(self._emb_refs[0], self.w_qkc, "qkc")
        agg_s = self._static_agg(self._emb_refs[1], self.w_qs, "qs")

        h_qkc = F.relu(
            self._qkc_self_w[seq_q].unsqueeze(-1) * self.w_qkc(gX) + agg_kc[seq_q]
        )
        h_qs = F.relu(
            self._qs_self_w[seq_q].unsqueeze(-1) * self.w_qs(gX) + agg_s[seq_q]
        )
        return h_qkc, h_qs


class GCNGraph(nn.Module):
    """Graph module producing the two structural feature streams via GCN."""

    def __init__(
        self,
        d_model: int,
        emb_kc: nn.Parameter,
        emb_s: nn.Parameter,
        gcn_qkc: dict,
        gcn_qs: dict,
        scatter: str = "mean",
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.gcn = HeteroGCN(d_model, emb_kc, emb_s, gcn_qkc, gcn_qs, scatter)

    def forward(self, x_question: torch.Tensor, seq_q: torch.Tensor):
        """Temporal-to-spatial projection followed by the two convolutions."""
        return self.gcn(self.layer_norm(x_question), seq_q)


class RWGraph(nn.Module):
    """Graph module producing structural features via metapath random walks.

    Walks are resampled every forward call, so evaluation predictions depend
    on the global RNG state (not stable across repeated evaluates in one
    process). Zero-degree nodes stay in place for the rest of the walk.
    """

    _HOP_SLOTS = (0, 2, 4, 6, 8)

    def __init__(self, emb_q: nn.Parameter, rw_csr: dict):
        super().__init__()
        # Plain list (not a registered attribute) so the shared Parameter
        # stays registered once, on the host model's embedding layer.
        self._emb_refs = [emb_q]
        for etype, (ptr, idx) in rw_csr.items():
            self.register_buffer(
                f"_{etype}_ptr", torch.from_numpy(ptr), persistent=False
            )
            self.register_buffer(
                f"_{etype}_idx", torch.from_numpy(idx), persistent=False
            )

    def forward(self, x_question: torch.Tensor, seq_q: torch.Tensor):
        bs, seqlen = seq_q.shape

        nodes = seq_q.reshape(-1)
        traces = [nodes]
        for etype in _RW_METAPATH:
            ptr = getattr(self, f"_{etype}_ptr")
            idx = getattr(self, f"_{etype}_idx")
            deg = ptr[nodes + 1] - ptr[nodes]
            safe_deg = deg.clamp(min=1)
            # The clamp keeps zero-degree rows in bounds; torch.where
            # discards their lookup result.
            r = (torch.rand_like(safe_deg.float()) * safe_deg).long()
            nxt = idx[(ptr[nodes] + r).clamp(max=idx.shape[0] - 1)]
            nodes = torch.where(deg > 0, nxt, nodes)
            traces.append(nodes)
        traces = torch.stack(traces, dim=1).view(bs, seqlen, -1)

        hop_emb = self._emb_refs[0][traces[:, :, self._HOP_SLOTS]]  # [BS, S, 5, D]

        hq = hop_emb[:, :, 0]
        for i, weight in enumerate(_RW_HOP_WEIGHTS[1:], start=1):
            hq = hq + weight * hop_emb[:, :, i]
        return hq, hq


class STHKT(nn.Module):
    """STHKT 模型

    Args:
        num_questions: 题目总数（嵌入表在此基础上 +1，0 为 padding）。
        num_skills: 概念（技能）总数。
        num_users: 真实学生总数（student embedding 与图节点数）。
        max_seq_len: 序列长度上限（Hawkes 的 seqlen 级线性层依赖）。
        d_model: 隐藏维度。
        num_heads: 注意力头数。
        d_ff: 前馈网络维度。
        dropout: Dropout 概率。
        result_hidden_dim: 输出层隐层维度。
        graph_mode: 图模块类型（"gcn" 或 "random_walk"）。
        hawkes_mode: Hawkes 注意力变体（full/without_hawkes/nn/qkc/qs/none）。
        graph_scatter: GCN 序列特征散落聚合方式（"mean" 或 "sum"）。
        graph_data: STHKTModelData 构建的图结构输入。
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        num_users: int,
        max_seq_len: int,
        d_model: int = 128,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        result_hidden_dim: int = 256,
        graph_mode: str = "gcn",
        hawkes_mode: str = "full",
        graph_scatter: str = "mean",
        graph_data: dict | None = None,
    ):
        super().__init__()
        if hawkes_mode not in _HAWKES_MODES:
            raise ValueError(f"Unknown hawkes_mode: {hawkes_mode}")

        self.hawkes_mode = hawkes_mode

        self.q_emb_layer = nn.Embedding(num_questions + 1, d_model)
        self.a_emb_layer = nn.Embedding(3, d_model)
        self.kc_emb_layer = nn.Embedding(num_skills, d_model)
        self.s_emb_layer = nn.Embedding(num_users, d_model)

        common = {
            "d_model": d_model,
            "d_ff": d_ff,
            "n_heads": num_heads,
            "dropout": dropout,
            "max_seq_len": max_seq_len,
        }
        self.q_Transformer_layer = TransformerLayer(hawkes_mode="none", **common)
        self.qa_Transformer_layer = TransformerLayer(hawkes_mode="none", **common)
        self.qqa_Transformer_layer = TransformerLayer(hawkes_mode=hawkes_mode, **common)

        self.graph = None
        if hawkes_mode != "none":
            if graph_data is None:
                raise ValueError("graph_data is required when hawkes_mode != 'none'.")
            if graph_mode == "gcn":
                self.graph = GCNGraph(
                    d_model,
                    self.kc_emb_layer.weight,
                    self.s_emb_layer.weight,
                    graph_data["gcn"]["qkc"],
                    graph_data["gcn"]["qs"],
                    scatter=graph_scatter,
                )
            elif graph_mode == "random_walk":
                self.graph = RWGraph(self.q_emb_layer.weight, graph_data["rw"])
            else:
                raise ValueError(f"Unknown graph_mode: {graph_mode}")

        self.layernorm_layer_q = nn.LayerNorm(d_model)
        self.layernorm_layer_qa = nn.LayerNorm(d_model)

        self.result_layer = nn.Sequential(
            nn.Linear(d_model * 2, result_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(result_hidden_dim, 1),
        )

    def forward(self, question: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            question: 题目序列 [B, S]（0 = padding）。
            response: 答案序列 [B, S]（0/1，padding 位 2）。

        Returns:
            作答概率 [B, S, 1]；位置 t 由 t 之前的交互与当前题目决定
            （same-position 语义，y_hat[:, t] 直接预测 response[:, t]）。
        """
        q_embed_data = self.q_emb_layer(question)
        qa_embed_data = q_embed_data + self.a_emb_layer(response)

        x = self.q_Transformer_layer(True, q_embed_data, q_embed_data, q_embed_data)
        x = self.layernorm_layer_q(x)
        y = self.qa_Transformer_layer(True, qa_embed_data, qa_embed_data, qa_embed_data)
        y = self.layernorm_layer_qa(y)

        h_qkc = h_qs = None
        if self.graph is not None:
            # Node features stay a pure function of the question id (raw
            # embeddings, layer-normalized inside the graph module).
            h_qkc, h_qs = self.graph(q_embed_data, question)

        h = self.qqa_Transformer_layer(False, x, x, y, h_qkc=h_qkc, h_qs=h_qs)

        result = self.result_layer(torch.cat((h, q_embed_data), dim=-1))
        return torch.sigmoid(result)
