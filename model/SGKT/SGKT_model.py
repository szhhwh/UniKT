"""SGKT model aligned with TensorFlow implementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import GeneralInteraction


class SumAggregator(nn.Module):
    def __init__(self, dim, dropout=0.0, act=torch.tanh):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.weights = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, self_vectors, neighbor_vectors, batch_size, seq_len):
        neighbors_agg = torch.mean(neighbor_vectors, dim=-2)
        output = (self_vectors + neighbors_agg).reshape(-1, self.dim)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = output @ self.weights + self.bias
        output = output.reshape(batch_size, seq_len, -1, self.dim)
        return self.act(output)


class ConcatAggregator(nn.Module):
    def __init__(self, dim, dropout=0.0, act=torch.tanh):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.act = act
        self.weights = nn.Parameter(torch.empty(dim * 2, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, self_vectors, neighbor_vectors, batch_size, seq_len):
        neighbors_agg = torch.mean(neighbor_vectors, dim=-2)
        output = torch.cat([self_vectors, neighbors_agg], dim=-1)
        output = output.reshape(-1, self.dim * 2)
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = output @ self.weights + self.bias
        output = output.reshape(batch_size, seq_len, -1, self.dim)
        return self.act(output)


class HRGEmbedding(nn.Module):
    def __init__(
        self,
        num_skills,
        num_questions,
        embedding_dim,
        question_neighbor_num,
        skill_neighbor_num,
        n_hop=3,
        keep_prob_gnn=0.8,
        aggregator="sum",
    ):
        super().__init__()
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.embedding_dim = embedding_dim
        self.question_neighbor_num = question_neighbor_num
        self.skill_neighbor_num = skill_neighbor_num
        self.n_hop = n_hop
        self.keep_prob_gnn = keep_prob_gnn

        if aggregator not in {"sum", "concat"}:
            raise ValueError("aggregator must be 'sum' or 'concat'")
        aggregator_cls = SumAggregator if aggregator == "sum" else ConcatAggregator

        # The source creates independent aggregators for the two question branches.
        self.aggregators, self.next_aggregators = [
            nn.ModuleList(
                [
                    aggregator_cls(
                        dim=embedding_dim,
                        dropout=1.0 - self.keep_prob_gnn,
                        act=torch.tanh,
                    )
                    for _ in range(n_hop)
                ]
            )
            for _ in range(2)
        ]

    def get_neighbors(self, n_hop, question_index, question_neighbors):
        seeds = [question_index]
        for i in range(n_hop):
            if i % 2 == 0:
                neighbor = question_neighbors[seeds[i].reshape(-1)].reshape(
                    -1, self.max_step, self.question_neighbor_num
                )
            else:
                neighbor = question_neighbors[seeds[i].reshape(-1)].reshape(
                    -1, self.max_step, self.skill_neighbor_num
                )
            seeds.append(neighbor)
        return seeds

    def aggregate(self, input_neighbors, embedding_table, aggregators):
        sq_neighbor_vectors = []
        for neighbors in input_neighbors:
            temp_neighbors = embedding_table[neighbors.reshape(-1)].reshape(
                self.batch_size, self.max_step, -1, self.embedding_dim
            )
            sq_neighbor_vectors.append(temp_neighbors)

        for i, aggregator in enumerate(aggregators):
            for hop in range(self.n_hop - i):
                if hop % 2 == 0:
                    shape = [
                        self.batch_size,
                        self.max_step,
                        -1,
                        self.question_neighbor_num,
                        self.embedding_dim,
                    ]
                else:
                    shape = [
                        self.batch_size,
                        self.max_step,
                        -1,
                        self.skill_neighbor_num,
                        self.embedding_dim,
                    ]
                vector = aggregator(
                    self_vectors=sq_neighbor_vectors[hop],
                    neighbor_vectors=sq_neighbor_vectors[hop + 1].reshape(shape),
                    batch_size=self.batch_size,
                    seq_len=self.max_step,
                )
                sq_neighbor_vectors[hop] = vector
        return sq_neighbor_vectors

    def forward(self, hrg_data, question_indices, next_question_indices):
        self.batch_size, self.max_step = question_indices.shape

        input_neighbors = self.get_neighbors(
            self.n_hop, question_indices, hrg_data["question_neighbors"]
        )
        aggregate_embedding = self.aggregate(
            input_neighbors, hrg_data["feature_embedding"], self.aggregators
        )

        next_input_neighbors = self.get_neighbors(
            self.n_hop, next_question_indices, hrg_data["question_neighbors"]
        )
        next_aggregate_embedding = self.aggregate(
            next_input_neighbors, hrg_data["feature_embedding"], self.next_aggregators
        )

        question_features = aggregate_embedding[0]
        if question_features.dim() == 4:
            question_features = question_features.squeeze(2)

        next_question_features = next_aggregate_embedding[0]
        if next_question_features.dim() == 4:
            next_question_features = next_question_features.squeeze(2)

        return question_features, {
            "next_aggregate_embedding": next_aggregate_embedding,
            "next_question_features": next_question_features,
        }

    def sample_next_neighbors(self, next_aggregate_embedding, num_samples):
        temp_emb = next_aggregate_embedding[1]
        batch_size, seq_len, _, emb_dim = temp_emb.shape
        temp_emb = temp_emb.reshape(-1, self.question_neighbor_num, emb_dim)
        if self.training:
            temp_emb = temp_emb.transpose(0, 1)
            perm = torch.randperm(temp_emb.shape[0], device=temp_emb.device)
            temp_emb = temp_emb[perm].transpose(0, 1)
        if self.question_neighbor_num >= num_samples:
            next_neighbors = temp_emb[:, :num_samples, :].reshape(
                batch_size, seq_len, num_samples, emb_dim
            )
        else:
            repeat_times = -(-num_samples // temp_emb.shape[0])
            tile_neighbor_embedding = temp_emb.repeat(1, repeat_times, 1)
            next_neighbors = tile_neighbor_embedding[:, :num_samples, :].reshape(
                batch_size, seq_len, num_samples, emb_dim
            )
        return next_neighbors


class SGEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gru_gates = nn.Linear(3 * embedding_dim, 2 * embedding_dim)
        self.gru_candidate = nn.Linear(3 * embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(self.gru_gates.weight)
        nn.init.ones_(self.gru_gates.bias)
        nn.init.xavier_uniform_(self.gru_candidate.weight)
        nn.init.zeros_(self.gru_candidate.bias)
        stdv = 1.0 / math.sqrt(embedding_dim)
        self.W_in = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        self.b_in = nn.Parameter(torch.empty(embedding_dim))
        self.W_out = nn.Parameter(torch.empty(embedding_dim, embedding_dim))
        self.b_out = nn.Parameter(torch.empty(embedding_dim))
        nn.init.uniform_(self.W_in, -stdv, stdv)
        nn.init.uniform_(self.b_in, -stdv, stdv)
        nn.init.uniform_(self.W_out, -stdv, stdv)
        nn.init.uniform_(self.b_out, -stdv, stdv)

    def forward(self, question_emb, answer_emb, input_trans_embedding):
        in_states = question_emb + answer_emb + input_trans_embedding
        fin_states_in = in_states @ self.W_in + self.b_in
        fin_states_out = in_states @ self.W_out + self.b_out
        av_all = torch.cat([fin_states_in, fin_states_out], dim=-1) + torch.cat(
            [in_states, in_states], dim=-1
        )

        state = answer_emb[:, 0]
        outputs = []
        for av in av_all.unbind(dim=1):
            reset, update = torch.sigmoid(
                self.gru_gates(torch.cat([av, state], dim=-1))
            ).chunk(2, dim=-1)
            # TF1 applies the reset gate before the recurrent candidate projection.
            candidate = torch.tanh(
                self.gru_candidate(torch.cat([av, reset * state], dim=-1))
            )
            state = update * state + (1 - update) * candidate
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class SelfAttentionHistory(nn.Module):
    def __init__(self, hidden_dim, seq_len, hist_neighbor_num):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.hist_neighbor_num = hist_neighbor_num
        stdv = 1.0 / math.sqrt(hidden_dim)
        self.xita = nn.Parameter(torch.empty(seq_len))
        self.xt1 = nn.Parameter(torch.empty(seq_len, seq_len))
        self.xt2 = nn.Parameter(torch.empty(seq_len, seq_len))
        self.bias = nn.Parameter(torch.empty(seq_len + 1, hidden_dim))
        self.K = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.Q = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.V = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.uniform_(self.xita, -stdv, stdv)
        nn.init.uniform_(self.xt1, -stdv, stdv)
        nn.init.uniform_(self.xt2, -stdv, stdv)
        nn.init.uniform_(self.bias, -stdv, stdv)
        nn.init.uniform_(self.K, -stdv, stdv)
        nn.init.uniform_(self.Q, -stdv, stdv)
        nn.init.uniform_(self.V, -stdv, stdv)

    def forward(self, input_embedding, hist_neighbor_index):
        batch_size, seq_len, hidden_dim = input_embedding.shape
        device = input_embedding.device

        if hist_neighbor_index.dim() == 2:
            hist_neighbor_index = hist_neighbor_index.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        causal_mask_xt = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=self.xt1.dtype)
        )  # [T, T],  mask[u, s] = 1  当  u <= s

        diff = input_embedding - input_embedding[:, :, 0:1]  # [B, T, H]

        transformed1 = torch.einsum(
            "buh,us->bsh", diff, self.xt1 * causal_mask_xt
        )  # [B, T, H]

        exp_transformed = torch.exp(transformed1)  # [B, T, H]

        transformed2 = torch.einsum(
            "bvh,vs->bsh", exp_transformed, self.xt2 * causal_mask_xt
        )  # [B, T, H]

        input_embedding_transformed = transformed2 + self.xita.unsqueeze(0).unsqueeze(
            -1
        )  # [B, T, H]

        zero_padding = torch.zeros(
            batch_size,
            1,
            hidden_dim,
            device=device,
            dtype=input_embedding.dtype,
        )
        input_emb_padded = torch.cat(
            [input_embedding_transformed, zero_padding], dim=1
        )  # [B, T+1, H]

        EK = input_emb_padded @ self.K + self.bias  # [B, T+1, H]
        EQ = input_emb_padded @ self.Q + self.bias  # [B, T+1, H]
        EV = input_emb_padded @ self.V + self.bias  # [B, T+1, H]

        A = torch.matmul(EQ, EK.transpose(1, 2)) / math.sqrt(hidden_dim)
        # [B, T+1, T+1]

        att_len = A.size(1)  # T+1
        causal_mask_att = torch.triu(
            torch.ones(att_len, att_len, device=device), diagonal=1
        ).bool()

        A = A.masked_fill(causal_mask_att.unsqueeze(0), 0.0)
        B = torch.matmul(A, EV)  # [B, T+1, H]

        zero_padding = torch.zeros(
            batch_size,
            1,
            hidden_dim,
            device=device,
            dtype=B.dtype,
        )
        B_padded = torch.cat([B, zero_padding], dim=1)  # [B, T+2, H]
        padding_position = B.size(1)

        hist_neighbor_index = hist_neighbor_index[:, :seq_len, :]
        hist_neighbor_index = torch.where(
            (hist_neighbor_index >= 0) & (hist_neighbor_index < seq_len),
            hist_neighbor_index,
            torch.full_like(hist_neighbor_index, padding_position),
        )
        hist_neighbor_index = hist_neighbor_index.reshape(batch_size, -1)

        hist_neighbors_features = torch.gather(
            B_padded,
            1,
            hist_neighbor_index.unsqueeze(-1).expand(-1, -1, hidden_dim),
        ).reshape(batch_size, seq_len, self.hist_neighbor_num, hidden_dim)

        return hist_neighbors_features


class HistoryRecap(nn.Module):
    def __init__(self, hist_neighbor_num, att_bound):
        super().__init__()
        self.hist_neighbor_num = hist_neighbor_num
        self.att_bound = att_bound

    def forward(self, input_q_emb, next_q_emb, qa_emb, hist_neighbor_index):
        batch_size, seq_len, emb_dim = input_q_emb.shape
        if hist_neighbor_index.dim() == 2:
            hist_neighbor_index = hist_neighbor_index.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        mold_nextq = torch.sqrt(torch.sum(next_q_emb * next_q_emb, dim=-1))
        mold_inputq = torch.sqrt(torch.sum(input_q_emb * input_q_emb, dim=-1))
        next_q_emb = next_q_emb.unsqueeze(2)
        input_q_emb = input_q_emb.unsqueeze(1)
        q_similarity = torch.sum(next_q_emb * input_q_emb, dim=-1)
        molds = mold_nextq.unsqueeze(2) * mold_inputq.unsqueeze(1)
        q_similarity = q_similarity / molds

        zero_embeddings = torch.zeros(
            batch_size, 1, qa_emb.shape[-1], device=qa_emb.device, dtype=qa_emb.dtype
        )
        qa_emb = torch.cat([qa_emb, zero_embeddings], dim=1)

        row_len = torch.arange(1, seq_len + 1, device=qa_emb.device)  # [S]
        col_idx = torch.arange(seq_len, device=qa_emb.device)  # [S]
        similarity_seqs = (col_idx.unsqueeze(0) < row_len.unsqueeze(1)).to(qa_emb.dtype)
        similarity_seqs = similarity_seqs.unsqueeze(0).expand(batch_size, -1, -1)
        q_similarity = q_similarity * similarity_seqs
        q_similarity = torch.where(
            q_similarity > self.att_bound,
            q_similarity,
            torch.zeros_like(q_similarity),
        )

        hist_attention_value, temp_hist_index = torch.topk(
            q_similarity, self.hist_neighbor_num, dim=2
        )
        temp_hist_index = torch.where(
            hist_attention_value > 0, temp_hist_index, hist_neighbor_index
        )
        temp_hist_index = temp_hist_index.clamp(min=0, max=seq_len)
        temp_hist_index = temp_hist_index.reshape(
            batch_size, seq_len * self.hist_neighbor_num
        )
        hist_neighbors_features = torch.gather(
            qa_emb,
            1,
            temp_hist_index.unsqueeze(-1).expand(-1, -1, qa_emb.shape[-1]),
        ).reshape(batch_size, seq_len, self.hist_neighbor_num, qa_emb.shape[-1])
        return hist_neighbors_features


class SGKT(nn.Module):
    def __init__(
        self,
        data_metadata,
        embedding_dim: int,
        hidden_dim: int,
        keep_prob_gnn: float,
        question_neighbor_num: int,
        skill_neighbor_num: int,
        n_hop: int,
        aggregator: str,
        hist_neighbor_num: int,
        att_bound: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_skills = data_metadata["num_skills"]
        self.num_questions = data_metadata["num_questions"]
        self.max_seq_len = data_metadata["max_question_seq_len"] - 1
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.keep_prob_gnn = keep_prob_gnn

        assert self.embedding_dim == self.hidden_dim, (
            f"SGKT requires embedding_dim == hidden_dim for dimension consistency "
            f"in student_status concatenation. Got embedding_dim={self.embedding_dim}, "
            f"hidden_dim={self.hidden_dim}."
        )

        self.feature_embedding = nn.Embedding(
            self.num_skills + self.num_questions + 2, self.embedding_dim
        )

        self.hrg_embedding = HRGEmbedding(
            num_skills=self.num_skills,
            num_questions=self.num_questions,
            embedding_dim=self.embedding_dim,
            question_neighbor_num=question_neighbor_num,
            skill_neighbor_num=skill_neighbor_num,
            n_hop=n_hop,
            keep_prob_gnn=self.keep_prob_gnn,
            aggregator=aggregator,
        )

        self.feature_trans = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.feature_trans_activation = nn.ReLU()
        self.input_trans = nn.Linear(
            self.hidden_dim + self.embedding_dim, self.hidden_dim
        )

        self.sg_embedding = SGEmbedding(embedding_dim=self.embedding_dim)
        self.hist_sampler = HistoryRecap(
            hist_neighbor_num=hist_neighbor_num,
            att_bound=att_bound,
        )
        self.self_attention = SelfAttentionHistory(
            hidden_dim=self.hidden_dim,
            seq_len=self.max_seq_len,
            hist_neighbor_num=hist_neighbor_num,
        )
        self.general_interaction = GeneralInteraction(self.hidden_dim)

    def forward(
        self,
        user_sequence,
        user_response,
        user_mask,
        hrg_data,
        hist_neighbor_index,
        return_states=False,
    ):
        question_indices = user_sequence[:, :-1] + self.num_skills
        next_question_indices = user_sequence[:, 1:] + self.num_skills
        answer_indices = user_response[:, :-1] + self.num_skills + self.num_questions

        question_embs = self.feature_embedding(question_indices)
        next_question_embs = self.feature_embedding(next_question_indices)
        answer_embs = self.feature_embedding(answer_indices)

        hrg_features, hrg_context = self.hrg_embedding(
            hrg_data, question_indices, next_question_indices
        )

        feature_trans_embedding = self.feature_trans_activation(
            self.feature_trans(hrg_features)
        )
        input_fa_embedding = torch.cat([feature_trans_embedding, answer_embs], dim=-1)
        input_trans_embedding = self.input_trans(input_fa_embedding)

        next_hrg_features = hrg_context["next_question_features"]
        next_trans_embedding = self.feature_trans_activation(
            self.feature_trans(next_hrg_features)
        )

        output_series = self.sg_embedding(
            question_embs, answer_embs, input_trans_embedding
        )

        input_trans_embedding = input_trans_embedding[
            :, : hist_neighbor_index.size(1), :
        ]
        question_embs = question_embs[:, : hist_neighbor_index.size(1), :]
        next_question_embs = next_question_embs[:, : hist_neighbor_index.size(1), :]
        E_answring_states = self.self_attention(
            input_embedding=input_trans_embedding,
            hist_neighbor_index=hist_neighbor_index,
        )

        hist_neighbors = self.hist_sampler(
            input_q_emb=question_embs,
            next_q_emb=next_question_embs,
            qa_emb=input_trans_embedding,
            hist_neighbor_index=hist_neighbor_index,
        )

        hist_neighbors_combined = hist_neighbors + E_answring_states

        next_neighbors = self.hrg_embedding.sample_next_neighbors(
            hrg_context["next_aggregate_embedding"],
            num_samples=hrg_data["next_neighbor_num"],
        )

        output_series = output_series[:, : hist_neighbors_combined.size(1), :]
        student_status = torch.cat(
            [output_series.unsqueeze(2), hist_neighbors_combined], dim=2
        )
        next_trans_embedding = next_trans_embedding[:, : next_neighbors.size(1), :]
        knowledge_status = torch.cat(
            [next_trans_embedding.unsqueeze(2), next_neighbors], dim=2
        )

        logits = self.general_interaction(
            student_status, knowledge_status, user_mask[:, :-1]
        )

        if return_states:
            # Extract skill embeddings: feature_embedding.weight[:num_skills]
            skill_embeddings = self.feature_embedding.weight[: self.num_skills]
            return logits, skill_embeddings, output_series
        return logits
