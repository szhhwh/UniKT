import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class GCN(nn.Module):
    """单层图卷积：``Dropout(x) W`` 经稀疏邻接 ``A`` 聚合后加偏置。"""

    def __init__(self, in_dim, out_dim, p):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_dim, out_dim))
        xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        x = torch.sparse.mm(adj, x)
        return x + self.b


class CosinePositionalEmbedding(nn.Module):
    """正余弦位置编码"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("weight", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.weight[:, : x.size(1), :]


def attention(q, k, v, d_k, mask, dropout, zero_pad, boost_focus):
    """带 boost_focus 增强与因果掩码的多头注意力打分。"""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores * (1 + boost_focus)  # boost_focus 广播到 heads

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    if zero_pad:
        output[:, :, 0, :].zero_()
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad, boost_focus):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, boost_focus)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, seq_len):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 因果掩码 [1, 1, L, L]：[0,0,i,j] = (j < i)
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=-1
        ).view(1, 1, seq_len, seq_len)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, mask, query, key, values, boost_focus, apply_pos=True):
        seqlen = query.size(1)
        src_mask = self.causal_mask[:, :, :seqlen, :seqlen]

        zero_pad = mask == 0
        query2 = self.masked_attn_head(
            query,
            key,
            values,
            mask=src_mask,
            zero_pad=zero_pad,
            boost_focus=boost_focus,
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """单流因果 Transformer：query=key=question, values=qa。"""

    def __init__(
        self,
        n_blocks,
        d_model,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        seq_len,
    ):
        super().__init__()
        self.d_model = d_model
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                    seq_len=seq_len,
                )
                for _ in range(n_blocks)
            ]
        )
        self.position_emb = CosinePositionalEmbedding(
            d_model=self.d_model, max_len=seq_len
        )

    def forward(self, q_embed_data, qa_embed_data, boost_focus):
        q_embed_data = q_embed_data + self.position_emb(q_embed_data)
        qa_embed_data = qa_embed_data + self.position_emb(qa_embed_data)

        x = q_embed_data  # query / key
        y = qa_embed_data  # values
        for block in self.blocks_2:
            x = block(
                mask=0,
                query=x,
                key=x,
                values=y,
                boost_focus=boost_focus,
                apply_pos=True,
            )
        return x


class DenoiseKT(nn.Module):
    """DenoiseKT 模型。

    Args:
        num_c: 概念（技能）数量。
        num_q: 题目数量。
        question_concepts: 数据派生查表 ``[num_q, max_concepts]``，每题概念列表，-1 填充。
        question_graph: 数据派生稀疏邻接 ``[num_q, num_q]``（``A = R Rᵀ``），供 GCN 使用。
        d_model: 模型隐藏维度。
        n_blocks: Transformer 块数量。
        dropout / dropout1: Transformer / GCN 的 dropout 概率。
        bf: boost_focus 的距离衰减底数（超参数，例如 0.9）。
        d_ff: 前馈网络维度。
        seq_len: 最大序列长度（位置编码 max_len）。
        kq_same: Key/Query 是否共享线性变换。
        final_fc_dim / final_fc_dim2: 输出 MLP 的两层维度。
        num_attn_heads: 注意力头数。
    """

    def __init__(
        self,
        num_c: int,
        num_q: int,
        question_concepts: torch.Tensor,
        question_graph: torch.Tensor,
        d_model: int = 256,
        n_blocks: int = 1,
        dropout: float = 0.1,
        dropout1: float = 0.1,
        bf: float = 0.9,
        d_ff: int = 64,
        seq_len: int = 200,
        kq_same: int = 1,
        final_fc_dim: int = 256,
        final_fc_dim2: int = 256,
        num_attn_heads: int = 8,
    ):
        super().__init__()
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = d_model
        self.num_attn_heads = num_attn_heads
        self.kq_same = kq_same
        self.bf = bf

        embed_l = d_model

        self.register_buffer(
            "question_concepts", question_concepts.long(), persistent=False
        )
        self.register_buffer("question_graph", question_graph, persistent=False)

        # 相对位置 [L, L]，[i,j] = i-j（对角为 0），供 boost_focus 使用
        idx = torch.arange(seq_len)
        rel_pos = (idx.view(-1, 1) - idx.view(1, -1)).to(torch.int64)
        self.register_buffer("rel_pos", rel_pos, persistent=False)

        # 题目难度（Rasch 风格），初始化为 0（中性）
        self.difficult_param = nn.Embedding(self.num_q + 1, embed_l)
        nn.init.constant_(self.difficult_param.weight, 0.0)

        # 作答嵌入
        self.ans_embed = nn.Embedding(2, embed_l)

        # 技能 / 题目嵌入
        self.skill_embed = nn.Parameter(torch.empty(self.num_c, self.emb_size))
        xavier_uniform_(self.skill_embed)
        self.pro_embed = nn.Parameter(torch.empty(self.num_q, self.emb_size))
        xavier_uniform_(self.pro_embed)

        self.gcn = GCN(self.emb_size, self.emb_size, dropout1)

        self.model = Architecture(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=num_attn_heads,
            dropout=dropout,
            kq_same=self.kq_same,
            seq_len=seq_len,
        )

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim2, 1),
        )

    def get_avg_skill_emb(self, c, emb):
        """对每个位置的概念集合取平均技能嵌入。

        c: ``[B, S, k]``，概念 id，-1 表示填充。
        emb: ``[num_c, D]`` 技能嵌入。
        返回: ``[B, S, D]``。
        """
        # 索引 0 留给填充（-1+1=0），1..num_c 对应技能
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.emb_size, device=emb.device, dtype=emb.dtype), emb],
            dim=0,
        )
        related_concepts = (c + 1).long()  # [B, S, k]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(dim=-2)  # [B, S, D]
        concept_num = (
            torch.where(related_concepts != 0, 1, 0).sum(dim=-1).unsqueeze(-1)
        )  # [B, S, 1]
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        return concept_emb_sum / concept_num

    def boost_focus(self, concept):
        """计算同概念位置对的距离衰减增强矩阵。

        concept: ``[B, S, k]``。
        返回: ``[B, S, S]`` long，``[b,i,j] = |i-j|`` 若 i,j 概念集合相同且 i≠j，否则 0。
        对角线由 ``rel_pos`` 为 0 天然置 0。
        """
        rows = concept.size(1)
        rel_pos = self.rel_pos[:rows, :rows]  # [S, S], i-j

        # 概念集合两两全等比较
        result = torch.all(concept[:, :, None, :] == concept[:, None, :, :], dim=-1)
        bf = torch.abs(result.int() * rel_pos)
        return bf.to(concept.dtype)

    def forward(
        self,
        question: torch.Tensor,
        response: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            question: 题目 id ``[B, S]``。
            response: 作答 ``[B, S]`` (0/1)。

        Returns:
            preds: ``[B, S]``，``preds[:, t]`` 由 ``question[:, t]`` 与历史
            ``qa[:, :t]`` 计算，预测 ``response[:, t]``。
        """
        cq = question
        cr = response
        cc = self.question_concepts[cq]  # [B, S, k]

        # GCN 聚合题目嵌入
        q_embed = self.gcn(self.pro_embed, self.question_graph)  # [num_q, D]
        q_embed_data = F.embedding(cq, q_embed)  # [B, S, D]
        ans_embed_data = self.ans_embed(cr)  # [B, S, D]
        qa_embed_data = q_embed_data + ans_embed_data

        # Rasch 风格题目编码: c_ct + uq * d̄_ct
        q_embed_diff_data = self.get_avg_skill_emb(cc, self.skill_embed)  # [B, S, D]
        pid_embed_data = self.difficult_param(cq)  # [B, S, D]
        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

        # boost_focus: 同概念位置对按距离衰减增强注意力
        boost_focus = self.bf ** self.boost_focus(cc).float()  # [B, S, S]
        boost_focus = boost_focus.masked_fill(boost_focus == 1.0, 0.0)
        boost_focus = boost_focus.unsqueeze(1)  # [B, 1, S, S]

        d_output = self.model(q_embed_data, qa_embed_data, boost_focus)  # [B, S, D]
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)  # [B, S, 2D]
        output = self.out(concat_q).squeeze(-1)  # [B, S]
        return torch.sigmoid(output)
