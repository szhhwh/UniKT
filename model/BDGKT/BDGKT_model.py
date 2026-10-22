"""BDGKT knowledge tracing model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BDGKT"]


class BDGKT(nn.Module):
    """BDGKT main model."""

    def __init__(
        self,
        student_num,
        question_num,
        skill_num,
        hidden_size,
        student_max_length,
        question_max_length,
        drop1,
        drop2,
        layer_num,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.l_s = question_max_length
        self.l_q = student_max_length

        self.student_emb = nn.Embedding(student_num, hidden_size)
        self.question_emb = nn.Embedding(question_num, hidden_size)
        self.skill_emb = nn.Embedding(skill_num, hidden_size)
        self.response_emb = nn.Embedding(2, hidden_size)

        self.layers = nn.ModuleList(
            [BDGKTLayer(hidden_size, drop1, drop2) for _ in range(layer_num)]
        )

        self.student_agg = nn.Linear((layer_num + 1) * hidden_size, hidden_size)
        self.question_agg = nn.Linear((layer_num + 1) * hidden_size, hidden_size)

        self.student_knowledge_pre = nn.Sequential(
            nn.Linear(hidden_size, skill_num), nn.Sigmoid()
        )
        self.question_difficulty_pre = nn.Sequential(
            nn.Linear(hidden_size, skill_num), nn.Sigmoid()
        )
        self.dynamic_fusion = nn.Linear(skill_num * 2, skill_num)
        self.qdiff_encoder = nn.Linear(hidden_size, skill_num)

        self.reset_parameters()

    def forward(
        self,
        target_students,
        target_questions,
        target_skills,
        hist_skills,
        hist_q,
        hist_r,
        hist_mask,
        q_ans_s,
        q_ans_r,
        q_ans_mask,
    ):
        """
        Args:
            target_students: [B]
            target_questions: [B]
            target_skills: [B, K], padded with -1
            hist_skills: [B, l_s, K], padded with -1
            hist_q: [B, l_s]
            hist_r: [B, l_s]
            hist_mask: [B, l_s]
            q_ans_s: [B, l_s, l_q]
            q_ans_r: [B, l_s, l_q]
            q_ans_mask: [B, l_s, l_q]

        Returns:
            pred: [B] prediction probability
        """
        # initial embeddings
        s_static = self.student_emb(target_students)  # [B, H]
        q_static = self.question_emb(target_questions)  # [B, H]

        # history question embeddings
        hq_emb = self.question_emb(hist_q)  # [B, l_s, H]
        hr_emb = self.response_emb(hist_r)  # [B, l_s, H]
        hist_mask_f = hist_mask.float().unsqueeze(-1)  # [B, l_s, 1]

        # skill features
        skill_mask = (hist_skills >= 0) & hist_mask.unsqueeze(-1)
        skill_embeddings = self.skill_emb(hist_skills.clamp_min(0))
        skill_feat = (skill_embeddings * skill_mask.unsqueeze(-1)).sum(
            -2
        ) / skill_mask.sum(-1, keepdim=True).clamp(min=1)

        # peer student embeddings
        ans_s_emb = self.student_emb(q_ans_s)  # [B, l_s, l_q, H]
        ans_r_emb = self.response_emb(q_ans_r)  # [B, l_s, l_q, H]
        ans_mask_f = q_ans_mask.float().unsqueeze(-1)  # [B, l_s, l_q, 1]

        # precompute q_abs_raw
        q_abs_raw = torch.cat([hq_emb, skill_feat], -1)  # [B, l_s, 2H]

        # layer processing
        student_layers = [s_static]
        question_layers = [q_static]

        s_dynamic = s_static
        hq_dynamic = hq_emb.clone()

        for layer in self.layers:
            q_abs_h = layer.question_abs_layer(q_abs_raw)  # [B, l_s, H]

            # Step 1: question attention (peer students → history questions)
            hq_dynamic_prev = hq_dynamic
            hq_dynamic = layer.update_questions(
                hq_dynamic,
                q_abs_h,
                ans_s_emb,
                ans_r_emb,
                ans_mask_f,
                hist_mask_f,
            )

            # Step 2: student RNN (history → knowledge state)
            s_dynamic = layer.update_student(
                hq_dynamic_prev,
                q_abs_h,
                hr_emb,
                hist_mask_f,
            )

            student_layers.append(s_dynamic)

            # target question feature: from history or fallback to static
            tq_dynamic = self._get_target_question_feature(
                target_questions,
                hq_dynamic,
                hist_q,
                hist_mask,
            )
            question_layers.append(tq_dynamic)

        # cross-layer aggregation + IRT prediction
        target_s = self.student_agg(torch.cat(student_layers, -1))
        target_q = self.question_agg(torch.cat(question_layers, -1))

        knowledge = self.student_knowledge_pre(target_s)
        diff = self.question_difficulty_pre(target_q)

        factor = torch.sigmoid(
            self.dynamic_fusion(torch.cat([knowledge, 1.0 - diff], -1))
        )

        abs_qdiff = torch.sigmoid(self.qdiff_encoder(q_static))

        raw_prob = 1.0 / (1.0 + torch.exp(-1.702 * 5.0 * (factor - abs_qdiff)))

        target_skill_mask = target_skills >= 0
        skill_prob = raw_prob.gather(-1, target_skills.clamp_min(0))
        pred = (skill_prob * target_skill_mask).sum(-1) / target_skill_mask.sum(
            -1
        ).clamp(min=1)
        pred = pred.clamp(1e-6, 1.0 - 1e-6)

        return pred

    def _get_target_question_feature(
        self, target_questions, hq_dynamic, hist_q, hist_mask
    ):
        """Get latest feature of target question from history,
        or fall back to static embedding.
        """
        B, l_s = hist_q.shape

        result = self.question_emb(target_questions)  # [B, H]

        # hist_q: [B, l_s], target_questions: [B] → match: [B, l_s]
        match = (hist_q == target_questions.unsqueeze(-1)) & hist_mask  # [B, l_s]

        has_match = match.any(dim=1)  # [B]

        if has_match.any():
            # reverse timeline, argmax finds the last True position
            match_rev = torch.flip(match.float(), [1])  # [B, l_s]
            last_from_right = match_rev.argmax(dim=1)  # [B]
            last_idx = l_s - 1 - last_from_right  # [B]

            dynamic_feat = hq_dynamic[
                torch.arange(B, device=hq_dynamic.device), last_idx
            ]  # [B, H]
            result = torch.where(has_match.unsqueeze(-1), dynamic_feat, result)

        return result

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_normal_(weight, gain=gain)


class BDGKTLayer(nn.Module):
    """BDGKT single layer: question attention + student RNN.

    q_abs is passed in from outside (computed once per layer).
    Time-invariant Linear projections are precomputed as batched matmuls.
    """

    def __init__(self, hidden_size, drop1, drop2):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_dropout = nn.Dropout(drop1)
        self.attention_dropout = nn.Dropout(drop2)

        # projections
        self.student_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.question_weight = nn.Linear(hidden_size, hidden_size, bias=False)

        # shared question_abs (once per layer)
        self.question_abs_layer = nn.Linear(hidden_size * 2, hidden_size)

        # question attention (peer students → question)
        self.attention_key = nn.Linear(hidden_size * 3, hidden_size)
        self.attention_query = nn.Linear(hidden_size, hidden_size)
        self.attention_value = nn.Linear(hidden_size * 2, hidden_size)

        # student RNN
        self.knowledge_init = nn.Parameter(torch.randn(1, hidden_size))
        self.question_transform = nn.Linear(hidden_size * 3, hidden_size)
        self.response_output = nn.Linear(hidden_size * 2, hidden_size)
        self.question_query = nn.Linear(hidden_size, hidden_size)
        self.forget_gate = nn.Linear(hidden_size * 2, hidden_size)

    def update_questions(
        self,
        hq_dynamic,
        q_abs,
        ans_s_emb,
        ans_r_emb,
        ans_mask_f,
        hist_mask_f,
    ):
        """Question attention: aggregate peer student information for each history question.

        hq_dynamic: [B, l_s, H]
        q_abs:      [B, l_s, H]
        ans_s_emb:  [B, l_s, l_q, H]
        ans_r_emb:  [B, l_s, l_q, H]
        ans_mask_f: [B, l_s, l_q, 1]
        hist_mask_f:[B, l_s, 1]
        """
        B, l_s, H = hq_dynamic.shape
        l_q = ans_s_emb.size(2)

        hq_proj = self.question_weight(self.feature_dropout(hq_dynamic))
        ans_s_proj = self.student_weight(self.feature_dropout(ans_s_emb))

        ans_s_flat = ans_s_proj.reshape(B * l_s, l_q, H)
        ans_r_flat = ans_r_emb.reshape(B * l_s, l_q, H)
        ans_mask_flat = ans_mask_f.reshape(B * l_s, l_q, 1)
        q_abs_flat = q_abs.reshape(B * l_s, 1, H).expand(-1, l_q, -1)

        key = self.attention_key(torch.cat([ans_s_flat, q_abs_flat, ans_r_flat], -1))
        query = self.attention_query(q_abs_flat)
        value = self.attention_value(torch.cat([q_abs_flat, ans_r_flat], -1))

        alpha = (query * key).sum(-1) / math.sqrt(H)  # [B*l_s, l_q]

        # mask: -inf for invalid, zero out all-empty rows to avoid NaN
        mask_2d = ans_mask_flat.squeeze(-1)  # [B*l_s, l_q]
        has_candidates = mask_2d.any(dim=-1, keepdim=True)  # [B*l_s, 1]
        alpha = alpha.masked_fill(~mask_2d.bool(), float("-inf"))
        alpha = torch.softmax(alpha, dim=-1)
        alpha = torch.where(has_candidates, alpha, torch.zeros_like(alpha))
        alpha = self.attention_dropout(alpha).unsqueeze(-1)  # [B*l_s, l_q, 1]

        # weighted sum
        attended = (alpha * value).sum(dim=1)  # [B*l_s, H]
        attended = attended.reshape(B, l_s, H)

        # valid history + candidates → attention result; otherwise keep projection
        has_cand_3d = has_candidates.reshape(B, l_s, 1)
        result = torch.where(hist_mask_f.bool() & has_cand_3d.bool(), attended, hq_proj)

        return result

    def update_student(
        self,
        hq_dynamic,
        q_abs,
        hr_emb,
        hist_mask_f,
    ):
        """Student RNN: iterate through history questions, update knowledge state.

        hq_dynamic: [B, l_s, H]
        q_abs:      [B, l_s, H]
        hr_emb:     [B, l_s, H]
        hist_mask_f:[B, l_s, 1]
        """
        B, L, H = hq_dynamic.shape

        # Precompute time-invariant Linear projections as batched matmuls.
        # question_transform: Linear(3H, H), split into 3 parts
        W_q_full = self.question_transform.weight  # [H, 3H]
        b_q = self.question_transform.bias
        # qa_partial = W[:, H:2H] @ q_abs_t
        qa_partial = F.linear(q_abs, W_q_full[:, H : 2 * H])  # [B, l_s, H]

        # response_output: Linear(2H, H), split into 2 parts
        W_r_full = self.response_output.weight  # [H, 2H]
        b_r = self.response_output.bias
        # r_partial = W[:, H:] @ hr_t
        r_partial = F.linear(hr_emb, W_r_full[:, H:])  # [B, l_s, H]

        # forget_gate: Linear(2H, H), split into 2 parts
        W_f_full = self.forget_gate.weight  # [H, 2H]
        b_f = self.forget_gate.bias
        # fr_partial = W[:, :H] @ hr_t
        fr_partial = F.linear(hr_emb, W_f_full[:, :H])  # [B, l_s, H]

        # RNN loop
        knowledge = self.knowledge_init.expand(B, -1).clone()

        for t in range(L):
            mask_t = hist_mask_f[:, t]  # [B, 1]
            prev_knowledge = knowledge
            q_t = hq_dynamic[:, t]  # [B, H]

            # q_trans = q_t@W_q + qa_partial_t + knowledge@W_k + b_q
            q_trans = (
                F.linear(q_t, W_q_full[:, :H])
                + qa_partial[:, t]
                + F.linear(knowledge, W_q_full[:, 2 * H :])
            )
            if b_q is not None:
                q_trans = q_trans + b_q

            q_act = torch.tanh(self.question_query(q_trans))

            # resp_out = sigmoid(q_trans@W_r_q + r_partial_t + b_r)
            resp_linear = F.linear(q_trans, W_r_full[:, :H]) + r_partial[:, t]
            if b_r is not None:
                resp_linear = resp_linear + b_r
            resp_out = torch.sigmoid(resp_linear) * q_act

            # forget = sigmoid(fr_partial_t + knowledge@W_f_k + b_f)
            forget_linear = fr_partial[:, t] + F.linear(knowledge, W_f_full[:, H:])
            if b_f is not None:
                forget_linear = forget_linear + b_f
            forget = torch.sigmoid(forget_linear)

            knowledge = forget * knowledge + (1.0 - forget) * resp_out

            # keep original knowledge for invalid positions
            knowledge = torch.where(mask_t.bool(), knowledge, prev_knowledge)

        return knowledge
