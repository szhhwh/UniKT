import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.init import constant_, kaiming_normal_, xavier_uniform_

from utils.attention import multihead_attention


class ClusterKT(nn.Module):
    def __init__(
        self,
        n_question,
        n_pid,
        d_model,
        n_blocks,
        kq_same,
        dropout,
        cluster_size,
        final_fc_dim,
        n_heads,
        d_ff,
        n_st,
        n_et,
        separate_qa=False,
    ):
        super().__init__()
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.separate_qa = separate_qa
        self.time = n_st
        self.interval = n_et
        self.d_model = d_model

        # Rasch difficulty model (optional)
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, self.d_model)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, self.d_model)

        self.q_embed = nn.Embedding(self.n_question + 1, self.d_model)
        self.qa_embed = nn.Embedding(2 * self.n_question + 2, self.d_model)
        self.interval_embed = nn.Embedding(
            self.interval + 2, self.d_model, padding_idx=self.interval + 1
        )
        self.time_embed = nn.Embedding(
            self.time + 2, self.d_model, padding_idx=self.time + 1
        )

        self.decoder_map = nn.Linear(self.d_model * 3, self.d_model)
        self.encoder_map = nn.Linear(self.d_model * 3, self.d_model)
        self.lg_unit = LearningGainUnit(dropout, self.d_model)
        self.trans = Architecture(
            n_blocks=n_blocks,
            n_heads=n_heads,
            dropout=dropout,
            d_model=self.d_model,
            d_ff=d_ff,
            kq_same=self.kq_same,
        )

        self.concept_center = Parameter(
            torch.Tensor(cluster_size, d_model), requires_grad=True
        )
        self.state_center = Parameter(
            torch.Tensor(cluster_size, d_model), requires_grad=True
        )
        kaiming_normal_(self.concept_center)
        kaiming_normal_(self.state_center)

        self.add_gate = nn.Linear(d_model, d_model)
        self.erase_gate = nn.Linear(d_model, d_model)
        self.forget_para_map = nn.Linear(2 * d_model, 1)

        self.pre_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model + self.d_model, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                nn.init.constant_(p, 0.0)

    def forward(
        self, q_data, response, mask=None, pid_data=None, dotime=None, lagtime=None
    ):
        """
        Args:
            q_data: skill_group IDs [B, S]
            response: correctness labels [B, S]
            mask: validity mask [B, S] (unused in forward, used by trainer)
            pid_data: question IDs for Rasch model [B, S], optional
            dotime: spent time [B, S], optional (padding if unavailable)
            lagtime: elapsed time intervals [B, S]
        Returns:
            (logits [B, S], cluster_loss scalar)
        """
        # Compute interaction encoding internally
        if self.separate_qa:
            qa_data = q_data + response * self.n_question
        else:
            qa_data = response.clone()
            qa_data[qa_data == 0] = self.n_question
            qa_data = q_data + qa_data

        q_embed_data = self.q_embed(q_data)

        # Embed answers
        if self.separate_qa:
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(qa_data) + q_embed_data

        # Rasch difficulty modeling
        q_embed_diff_data = None
        if self.n_pid > 0 and pid_data is not None:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )

        # Embed temporal features
        st = self.time_embed(dotime)
        et = self.interval_embed(lagtime)
        q_embed_data = self.encoder_map(torch.cat((q_embed_data, st, et), dim=-1))
        qa_embed_data = self.decoder_map(torch.cat((qa_embed_data, st, et), dim=-1))

        batch_size = q_embed_data.size(0)
        seq_len = q_embed_data.size(1)
        # Detach to allow in-place updates during online clustering without breaking autograd
        b_concept_center = self.concept_center.data.unsqueeze(0).repeat(
            batch_size, 1, 1
        )
        b_state_center = self.state_center.data.unsqueeze(0).repeat(batch_size, 1, 1)

        q_ori = q_embed_data.detach().permute(1, 0, 2).unsqueeze(-1)
        qa_ori = qa_embed_data.detach().permute(1, 0, 2).unsqueeze(-1)
        h_add = torch.tanh(self.add_gate(qa_embed_data))
        h_minus = torch.sigmoid(self.erase_gate(qa_embed_data))
        h_list = []
        cluster_loss = torch.tensor(0.0, device=q_embed_data.device)
        sum_center = b_concept_center.clone().detach()
        batch_arange = torch.arange(batch_size, device=q_embed_data.device)
        for i in range(seq_len):
            sim = F.cosine_similarity(
                q_ori[i], b_concept_center.permute(0, 2, 1), dim=1
            )
            idx = torch.argmax(sim, dim=-1)
            b_loss = torch.max(sim, dim=-1).values
            cluster_loss = cluster_loss + (1 - b_loss).sum()
            idx_center = sum_center[batch_arange, idx, :]
            tempt = q_ori[i].squeeze(-1) + idx_center
            cluster_norm = torch.norm(tempt, p=1, dim=-1).unsqueeze(dim=-1)
            b_concept_center[batch_arange, idx, :] = tempt / cluster_norm
            sum_center[batch_arange, idx, :] = tempt

            batch_idx_state = b_state_center[batch_arange, idx, :]
            h_list.append(batch_idx_state)
            batch_idx_state = batch_idx_state * (1 - h_minus[:, i, :]) + h_add[
                :, i, :
            ] * qa_ori[i].squeeze(-1)
            b_state_center[batch_arange, idx, :] = batch_idx_state

        cluster_state = torch.stack(h_list, dim=1)
        # Pairwise cosine similarity
        norm_q = F.normalize(cluster_state.detach(), dim=-1)
        norm_k = F.normalize(cluster_state, dim=-1)
        all_sim = torch.bmm(norm_q, norm_k.transpose(1, 2))  # (B, L, L)

        # Forgetting difficulty: use zeros if Rasch is not enabled
        if q_embed_diff_data is not None:
            forget_difficulty = torch.sigmoid(
                self.forget_para_map(
                    torch.cat((cluster_state, q_embed_diff_data), dim=-1)
                )
            )
        else:
            zeros_input = torch.zeros(
                batch_size, seq_len, self.d_model, device=q_embed_data.device
            )
            forget_difficulty = torch.sigmoid(
                self.forget_para_map(torch.cat((cluster_state, zeros_input), dim=-1))
            )

        learning_states = self.lg_unit(q_embed_data, cluster_state, all_sim)
        forgetting_states = self.trans(
            q_embed_data, learning_states, all_sim, forget_difficulty
        )

        # Predict via attention and MLP (no sigmoid — output logits)
        attn_mask = torch.triu(
            torch.ones(
                learning_states.size(1),
                learning_states.size(1),
                device=learning_states.device,
            ),
            diagonal=1,
        ).bool()
        trans_output = multihead_attention(
            self.pre_attn,
            learning_states,
            forgetting_states,
            cluster_state,
            attn_mask=attn_mask,
        )
        concat_q = torch.cat([trans_output, q_embed_data], dim=-1)
        output = self.mlp(concat_q)
        x = output.squeeze(-1)

        return x, cluster_loss


class LearningGainUnit(nn.Module):
    def __init__(self, dropout, d_model):
        super().__init__()
        self.d_model = d_model
        self.exp = Parameter(
            nn.init.xavier_uniform_(torch.empty(1, self.d_model)), requires_grad=True
        )
        self.linear_gate = nn.Linear(2 * self.d_model, self.d_model)
        self.linear_k = nn.Linear(2 * self.d_model, self.d_model)
        self.linear_re = nn.Linear(2 * self.d_model, self.d_model)

    def forward(self, q_embed_data, cluster_state, all_sim):
        B, L, d = cluster_state.shape
        device = cluster_state.device
        padding = torch.zeros(B, 1, d, device=device)
        shifted_cluster = torch.cat((padding, cluster_state[:, :-1, :]), dim=1)
        shifted_q = torch.cat((padding, q_embed_data[:, :-1, :]), dim=1)

        x = torch.cat([shifted_cluster[:, 1:, :], shifted_q[:, 1:, :]], dim=-1)
        gate = torch.sigmoid(self.linear_gate(x))
        gain = gate * torch.tanh(self.linear_k(x))
        reset = torch.sigmoid(self.linear_re(x))

        seqlen = L - 1
        sim_weight = all_sim[:, :seqlen, :].mean(dim=-1, keepdim=True)

        exp = self.exp.repeat(B, 1).to(device)
        result = torch.empty(B, L, d, device=device)
        result[:, 0, :] = exp
        for t in range(seqlen):
            exp = (
                reset[:, t, :] * exp
                + (1 - reset[:, t, :]) * gain[:, t, :] * sim_weight[:, t, :]
            )
            result[:, t + 1, :] = exp

        return result


class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        self.blocks_1 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(n_blocks)
            ]
        )
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(n_blocks * 2)
            ]
        )

    def forward(self, q_embed_data, qa_embed_data, state_sim, decay):
        y = qa_embed_data
        x = q_embed_data
        for block in self.blocks_1:
            y = block(
                mask=1, query=y, key=y, values=y, state_sim=state_sim, decay=decay
            )
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(
                    mask=1,
                    query=x,
                    key=x,
                    values=x,
                    state_sim=state_sim,
                    decay=decay,
                    apply_pos=False,
                )
                flag_first = False
            else:
                x = block(
                    mask=0,
                    query=x,
                    key=x,
                    values=y,
                    state_sim=state_sim,
                    decay=decay,
                    apply_pos=True,
                )
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
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

    def forward(self, mask, query, key, values, state_sim, decay, apply_pos=True):
        seqlen = query.size(1)
        nopeek_mask = torch.triu(
            torch.ones(1, 1, seqlen, seqlen, device=query.device), diagonal=mask
        ).bool()
        src_mask = nopeek_mask == 0
        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, state_sim, decay, mask=src_mask, zero_pad=True
            )
        else:
            query2 = self.masked_attn_head(
                query, key, values, state_sim, decay, mask=src_mask, zero_pad=False
            )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if not self.kq_same:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, state_sim, decay, mask, zero_pad):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = forgetting_attention(
            q, k, v, self.d_k, mask, self.dropout, zero_pad, state_sim, decay
        )

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)


def forgetting_attention(q, k, v, d_k, mask, dropout, zero_pad, state_sim, difficulty):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_.masked_fill_(mask == 0, 0.0)
        dist_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        del scores_
        torch.sub(disttotal_scores, dist_scores, out=dist_scores)
        del disttotal_scores
        dist_scores.mul_(torch.unsqueeze(state_sim, 1)).clamp_min_(0.0).sqrt_()
        gamma = -1.0 * F.softplus(difficulty.unsqueeze(1))

        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)
        scores = scores * total_effect
        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)
        if zero_pad:
            pad_zero = torch.zeros(bs, head, 1, seqlen, device=scores.device)
            scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        scores = dropout(scores)

    return torch.matmul(scores, v)
