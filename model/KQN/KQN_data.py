"""Data adapter for the KQN baseline."""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class KQNDataset(Dataset):
    """KC-level sequence dataset for KQN.

    Training/validation items:
        ``(concept, response, mask, question)``.

    Shapes:
        concept: [S] 0-based KC ids; padding values are ignored by ``mask``.
        response: [S] binary labels.
        mask: [S] true for valid sequence positions.
        question: [S] question ids retained for batch contract and analysis;
            KQN itself does not consume question ids.
    """

    def __init__(self, concepts, responses, masks, questions):
        self.concepts = concepts
        self.responses = responses
        self.masks = masks
        self.questions = questions

    def __len__(self) -> int:
        return len(self.concepts)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.concepts[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.questions[idx], dtype=torch.long),
        )


class KQNModelData(SkillModelData):
    """Build KC sequence data for KQN."""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        concepts = to_first_skill_ids(sequences.skills)
        responses = sequences.response
        masks = sequences.mask
        questions = sequences.question

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        dataset = KQNDataset(concepts, responses, masks, questions)
        train_dataset, val_dataset, _ = partition.apply(dataset)

        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"KQN data prepared: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test(window)={len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset
