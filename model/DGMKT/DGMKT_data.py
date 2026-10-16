"""DGMKT 模型数据处理模块"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData
from utils.model_data.packed_sequences import to_first_skill_ids

logger = get_logger(__name__)


class DGMKTDataset(Dataset):
    def __init__(self, sequences, responses, masks, user_ids):
        self.sequences = sequences
        self.responses = responses
        self.masks = masks
        self.user_ids = user_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        user_id = torch.tensor(self.user_ids[idx], dtype=torch.long)
        return sequence, response, mask, user_id


def _build_incidence(
    user_sequence: np.ndarray,
    user_mask: np.ndarray,
    user_ids: np.ndarray,
    num_users: int,
    num_skills: int,
) -> np.ndarray:
    """Count-based student-skill incidence matrix H."""
    users, positions = np.nonzero(user_mask)
    skills = user_sequence[users, positions]
    owners = user_ids[users]
    H = np.zeros((num_users, num_skills), dtype=np.float64)
    np.add.at(H, (owners, skills), 1.0)
    return H


def _build_hypergraph_props(
    H: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Feng HGNN normalization in factored form.

    G = D_v^-1/2 H D_e^-1 H^T D_v^-1/2 is near-dense when few skills cover
    many users, so return the sparse incidence H plus diagonal scales.
    """
    DV = np.maximum(H.sum(axis=1), 1.0)
    DE = np.maximum(H.sum(axis=0), 1.0)
    dv = torch.from_numpy((DV**-0.5).astype(np.float32))
    de = torch.from_numpy((1.0 / DE).astype(np.float32))

    H_f = H.astype(np.float32)
    idx = np.nonzero(H_f)
    H_coo = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack(idx).astype(np.int64)),
        torch.from_numpy(H_f[idx]),
        H_f.shape,
    ).coalesce()
    logger.debug(
        f"HGNN hypergraph built (n_node={H.shape[0]}, n_edge={H.shape[1]}, "
        f"nnz={H_coo._nnz()})"
    )
    return H_coo, dv, de


class DGMKTModelData(SkillModelData):
    """DGMKT 模型数据加载器"""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        sequences = self.load_sequences()
        partition = sequences.partition(rc.data.fold)
        user_sequence = to_first_skill_ids(sequences.skills)
        user_response = sequences.response
        user_mask = sequences.mask

        num_skills = self.data_src.get_metadata("num_skills")
        orig_ids = sequences.user
        num_original = int(self.data_src.get_metadata("num_users"))

        logger.info(
            f"Building DGMKT student hypergraph from the full dataset: "
            f"{num_original} original users x {num_skills} skills"
        )
        H = _build_incidence(
            user_sequence, user_mask, orig_ids, num_original, num_skills
        )
        H_coo, dv, de = _build_hypergraph_props(H)
        num_users = num_original

        dataset = DGMKTDataset(user_sequence, user_response, user_mask, orig_ids)
        train_dataset, val_dataset, _ = partition.apply(dataset)

        test_dataset = self.create_windowlate_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.info(
            f"DGMKT data prepared: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test(window)={len(test_dataset)}"
        )
        seq_len = user_sequence.shape[1]
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            H_coo,
            dv,
            de,
            num_users,
            seq_len,
        )
