import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import QuestionModelData
from utils.model_data.packed_sequences import to_skill_combination_ids

logger = get_logger(__name__)


class ClusterKTDataset(Dataset):
    """Dataset for ClusterKT model.

    Returns (skill_group_seq, question_seq, response, mask, lagtime).
    """

    def __init__(self, sequences, questions, responses, masks, timestamps, n_et):
        self.sequences = sequences
        self.questions = questions
        self.responses = responses
        self.masks = masks
        self.timestamps = timestamps
        self.n_et = n_et

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        question = torch.tensor(self.questions[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.long)

        # Compute lagtime from timestamps
        timestamps = np.array(self.timestamps[idx], dtype=np.float64)
        lagtime = self._compute_lagtime(timestamps, mask)
        lagtime = torch.tensor(lagtime, dtype=torch.long)

        return sequence, question, response, mask, lagtime

    def _compute_lagtime(self, timestamps, mask):
        """Compute elapsed time intervals from absolute timestamps."""
        max_len = len(timestamps)
        lagtime = np.full(max_len, self.n_et + 1, dtype=np.int64)

        mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        valid = np.where(mask_np > 0)[0]
        if len(valid) <= 1:
            return lagtime

        for i in range(1, len(valid)):
            idx = valid[i]
            prev_idx = valid[i - 1]
            diff = int(timestamps[idx] - timestamps[prev_idx])
            lagtime[idx] = max(0, min(diff, self.n_et))

        lagtime[valid[0]] = self.n_et + 1
        return lagtime


class ClusterKTModelData(QuestionModelData):
    """ClusterKT data loader.

    Extends QuestionModelData with:
    - Skill group mapping: assigns each observed skill set a skill_group ID
    - Timestamp extraction for lagtime computation
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc):
        """Prepare ClusterKT datasets.

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        n_et = rc.model.n_et

        # Build sequences
        data = self.load_split_data(required=("timestamp",))
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        folded, self.num_skill_groups = to_skill_combination_ids(sequences.skills)
        user_sequence = np.where(sequences.mask, folded - 1, 0)
        user_question = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask
        user_timestamp = sequences.features["timestamp"].astype(np.float64)

        # K-fold split
        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        # Build datasets
        dataset = ClusterKTDataset(
            user_sequence,
            user_question,
            user_response,
            user_mask,
            user_timestamp,
            n_et=n_et,
        )
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        return train_dataset, val_dataset, test_dataset
