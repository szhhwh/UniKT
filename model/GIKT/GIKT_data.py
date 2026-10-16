"""GIKT 数据准备：构建问题-技能二部图邻居表（采样）与同技能历史邻居索引。"""

from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.core import get_logger
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


def sample_hist_neighbors(
    batch_size,
    max_seq_len,
    hist_neighbor_num,
    skill_index,
    pad_index=None,
    deterministic=False,
):
    """同技能历史邻居采样：对位置 t，在历史 [0, t-1] 中找技能相同的位置选 M 个。

    随机模式用于训练；deterministic 模式（评估）固定取最早的前 M 个候选，
    不足时循环填充，保证推理可复现。
    """
    if pad_index is None:
        pad_index = max_seq_len
    if hist_neighbor_num == 0:
        return np.zeros((batch_size, max_seq_len, 0), dtype=np.int64)

    skills = (
        skill_index.numpy()
        if isinstance(skill_index, torch.Tensor)
        else np.asarray(skill_index)
    )
    result = np.full(
        (batch_size, max_seq_len, hist_neighbor_num), pad_index, dtype=np.int64
    )
    result[:, 0, :] = pad_index  # position 0 has no history

    for b in range(batch_size):
        seq_skills = skills[b]
        same_skill = seq_skills[np.newaxis, :] == seq_skills[:, np.newaxis]
        causal = np.tril(np.ones((max_seq_len, max_seq_len), dtype=bool), k=-1)
        valid = same_skill & causal
        for t in range(1, max_seq_len):
            candidates = np.where(valid[t])[0]
            if len(candidates) == 0:
                continue
            if deterministic:
                result[b, t] = candidates[
                    np.arange(hist_neighbor_num) % len(candidates)
                ]
            elif len(candidates) >= hist_neighbor_num:
                result[b, t] = np.random.choice(
                    candidates, hist_neighbor_num, replace=False
                )
            else:
                result[b, t] = np.random.choice(
                    candidates, hist_neighbor_num, replace=True
                )
    return result


class GIKTModelData(QuestionModelData):
    def __init__(self, data_src):
        super().__init__(data_src)

    def prepare_data(self, rc: Any):
        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        np.where(sequences.mask, sequences.user[:, None], 0)

        question_skill_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        logger.info("Building question-skill neighbors (bipartite, sampled)")
        question_neighbors, skill_neighbors = self.build_qs_neighbors(
            question_skill_matrix,
            num_skills,
            num_questions,
            rc.model.question_neighbor_num,
            rc.model.skill_neighbor_num,
        )

        user_skills = to_first_skill_ids(sequences.skills)

        # First skill id per position (used for same-skill history sampling and sim_emb=skill_emb).

        dataset = GIKTDataset(user_sequence, user_response, user_mask, user_skills)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        train_collate_fn = partial(
            gikt_collate_fn, hist_neighbor_num=rc.model.hist_neighbor_num
        )
        # Eval collate samples neighbors deterministically so val/test metrics
        # are reproducible.
        eval_collate_fn = partial(
            gikt_collate_fn,
            hist_neighbor_num=rc.model.hist_neighbor_num,
            deterministic=True,
        )

        logger.info(
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        graph_data = {
            "question_neighbors": torch.from_numpy(question_neighbors).long(),
            "skill_neighbors": torch.from_numpy(skill_neighbors).long(),
            "next_neighbor_num": rc.model.next_neighbor_num,
            "feature_embedding": None,  # bound in trainer to model.feature_embedding.weight
        }
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            graph_data,
            num_skills,
            num_questions,
            train_collate_fn,
            eval_collate_fn,
        )

    def build_qs_neighbors(
        self,
        question_skill_matrix,
        num_skills,
        num_questions,
        question_neighbor_num,
        skill_neighbor_num,
    ):
        """构建问题-技能二部图邻居表并固定采样。

        节点 id 布局：技能 [0, num_skills)，题目 [num_skills, num_skills+num_questions)。
        """
        qs_num = num_skills + num_questions
        question_to_skills = [
            np.where(question_skill_matrix[q] == 1)[0] for q in range(num_questions)
        ]
        skill_to_questions = [
            np.where(question_skill_matrix[:, s] == 1)[0] + num_skills
            for s in range(num_skills)
        ]

        question_neighbors = np.zeros([qs_num, question_neighbor_num], dtype=np.int32)
        skill_neighbors = np.zeros([num_skills, skill_neighbor_num], dtype=np.int32)

        for q_id, neighbors in enumerate(question_to_skills):
            if len(neighbors) == 0:
                continue
            replace = len(neighbors) < question_neighbor_num
            question_neighbors[num_skills + q_id] = np.random.choice(
                neighbors, question_neighbor_num, replace=replace
            )
        for s_id, neighbors in enumerate(skill_to_questions):
            if len(neighbors) == 0:
                continue
            replace = len(neighbors) < skill_neighbor_num
            skill_neighbors[s_id] = np.random.choice(
                neighbors, skill_neighbor_num, replace=replace
            )
        return question_neighbors, skill_neighbors


class GIKTDataset(Dataset):
    def __init__(self, user_sequence, user_response, user_mask, user_skills):
        self.user_sequence = torch.from_numpy(np.asarray(user_sequence)).long()
        self.user_response = torch.from_numpy(np.asarray(user_response)).long()
        self.user_mask = torch.from_numpy(np.asarray(user_mask)).bool()
        self.user_skills = torch.from_numpy(np.asarray(user_skills)).long()

    def __len__(self):
        return len(self.user_sequence)

    def __getitem__(self, idx):
        return {
            "sequence": self.user_sequence[idx],
            "response": self.user_response[idx],
            "mask": self.user_mask[idx],
            "skills": self.user_skills[idx],
        }


def gikt_collate_fn(batch, hist_neighbor_num=3, deterministic=False):
    """批拼接并为每个 batch 计算 hist_neighbor_index。"""
    from torch.utils.data.dataloader import default_collate

    batched = default_collate(batch)
    batch_size, full_seq_len = batched["skills"].shape
    model_seq_len = full_seq_len - 1  # GIKT runs on max_step = full_seq_len - 1
    batched["hist_neighbor_index"] = torch.from_numpy(
        sample_hist_neighbors(
            batch_size,
            model_seq_len,
            hist_neighbor_num,
            batched["skills"][:, :model_seq_len],
            pad_index=model_seq_len,
            deterministic=deterministic,
        )
    ).long()
    return batched
