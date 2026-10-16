from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_multi_skill_ids

logger = get_logger(__name__)

__all__ = ["GRKTModelData", "GRKTDataset"]


class GRKTDataset(Dataset):
    def __init__(self, questions, knows, responses, times, masks):
        self.questions = questions
        self.knows = knows
        self.responses = responses
        self.times = times
        self.masks = masks

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.questions[idx], dtype=torch.long),
            torch.tensor(self.knows[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.times[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
        )


class GRKTModelData(QuestionModelData):
    """GRKT model data preparation.

    - question sequences [num_users, max_seq_len]
    - multi-skill sequences [num_users, max_seq_len, K] (K = max skills per event)
    - response sequences [num_users, max_seq_len]
    - time sequences [num_users, max_seq_len]  (absolute seconds from millisecond
      timestamps / 1000 when available, else 1-indexed seq_pos + 1 as fallback)
    - mask sequences [num_users, max_seq_len]
    Also computes rel_map and pre_map (skill relationship and prerequisite
    graphs) from train-fold co-occurrence statistics, matching
    the sequence partition's training membership.
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        # Load split question sequence data
        sequences = self.load_sequences(optional=("timestamp",))
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        user_knows = to_multi_skill_ids(sequences.skills, padding_value=num_skills)
        K = user_knows.shape[-1]
        if "timestamp" in sequences.features:
            user_time = (
                sequences.features["timestamp"].astype(np.float64) / 1000.0
            ).astype(np.int64)
        else:
            user_time = np.where(
                sequences.mask, np.arange(sequences.mask.shape[1]) + 1, 0
            )

        # K-fold split
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")

        dataset = GRKTDataset(
            user_sequence, user_knows, user_response, user_time, user_mask
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        train_user_indices = partition.train
        rel_map, pre_map = self._build_skill_graph(
            user_sequence[train_user_indices],
            user_knows[train_user_indices],
            user_response[train_user_indices],
            user_mask[train_user_indices],
            num_skills,
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
            num_questions,
            num_skills,
            K,
            rel_map,
            pre_map,
        )

    def _build_skill_graph(self, questions, knows, responses, masks, num_skills):
        """Build skill relationship and prerequisite graphs from the given data.

        Computes four co-occurrence matrices (tt, tf, ft, ff) based on the
        correctness patterns of skill pairs within each user sequence. Then
        derives:
        - rel_map: skill relevance probability (agreement rate)
        - pre_map: prerequisite probability (incorrect→correct pattern)

        Uses the first annotated skill of each event for graph statistics.
        """
        tt = np.zeros((num_skills, num_skills))
        tf = np.zeros((num_skills, num_skills))
        ft = np.zeros((num_skills, num_skills))
        ff = np.zeros((num_skills, num_skills))

        for i in tqdm(range(len(questions)), desc="Building skill graph"):
            mask = masks[i]
            seq_len = int(mask.sum())
            t_vec = np.zeros(num_skills)
            f_vec = np.zeros(num_skills)
            for t in range(seq_len):
                know = knows[i, t, 0]  # First skill (as in original)
                corr = bool(responses[i, t])
                if corr:
                    tt[:, know] += t_vec
                    tt[know, :] += t_vec
                    ft[:, know] += f_vec
                    tf[know, :] += f_vec
                else:
                    ff[:, know] += f_vec
                    ff[know, :] += f_vec
                    tf[:, know] += t_vec
                    ft[know, :] += t_vec
                if corr:
                    t_vec[know] += 1
                else:
                    f_vec[know] += 1

        cold_thresh = 5
        rel_filt = (tt + ff + tf + ft) >= 4 * cold_thresh
        pre_filt = (tf + ft) >= 2 * cold_thresh

        rel_map = (tt + ff) / np.clip(tt + ff + tf + ft, a_min=1, a_max=None)
        pre_map = ft / np.clip(tf + ft, a_min=1, a_max=None)

        for i in range(len(rel_map)):
            rel_map[i, i] = 0
            pre_map[i, i] = 0

        rel_map = rel_map * rel_filt
        pre_map = pre_map * pre_filt

        return rel_map.astype(np.float32), pre_map.astype(np.float32)
