"""DGEKT 模型数据处理模块。"""

from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


def _safe_inv_sqrt(values: np.ndarray) -> np.ndarray:
    """Element-wise ``x^-0.5`` with zeros kept as zeros."""
    inv_sqrt = np.zeros_like(values, dtype=np.float64)
    nz = values > 0
    inv_sqrt[nz] = values[nz] ** -0.5
    return inv_sqrt


class DGEKTDataset(Dataset):
    """DGEKT 数据集。

    每个样本返回 ``(question, response, mask)``：
        question: 题目序列 [S]
        response: 答案序列 [S]
        mask: 有效位置掩码 [S]
    """

    def __init__(self, questions, responses, masks):
        self.questions = questions
        self.responses = responses
        self.masks = masks

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.questions[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
        )


class DGEKTModelData(QuestionModelData):
    """DGEKT 模型数据加载器。"""

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与模型所需的图结构。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中 ``info``
            为含 ``num_questions``、``num_skills``、``max_seq_len``、
            ``hyper_factors``、``adj_out``、``adj_in`` 的字典。
        """
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")
        max_seq_len = self.data_src.get_metadata("max_question_seq_len")

        sequences = self.load_sequences()
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask

        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))
        hyper_p, hyper_q = self._build_hyper_factors(q_matrix)
        # Transition counts are collected from the training fold only.
        adj_out, adj_in = self._build_transition_adj(
            user_sequence[partition.train],
            user_response[partition.train],
            user_mask[partition.train],
            num_questions,
        )
        logger.info(
            f"DGEKT graphs: hyper P nnz={hyper_p._nnz()}, Q nnz={hyper_q._nnz()}, "
            f"adj_out nnz={adj_out._nnz()}, adj_in nnz={adj_in._nnz()}"
        )
        logger.info(
            f"DGEKT data: num_questions={num_questions}, num_skills={num_skills}, "
            f"max_seq_len={max_seq_len}"
        )

        dataset = DGEKTDataset(user_sequence, user_response, user_mask)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"DGEKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        info = {
            "num_questions": num_questions,
            "num_skills": num_skills,
            "max_question_seq_len": max_seq_len,
            "hyper_factors": (hyper_p, hyper_q),
            "adj_out": adj_out,
            "adj_in": adj_in,
        }
        return train_dataset, val_dataset, test_dataset, info

    def _build_hyper_factors(self, q_matrix: np.ndarray) -> tuple:
        """超图拉普拉斯 G = Dv^-1/2·H·De^-1·H^T·Dv^-1/2 的因子分解。

        返回 (P, Q) = (Dv^-1/2·H·De^-1, H^T·Dv^-1/2).
        """
        qm = sparse.coo_matrix(q_matrix, dtype=np.float64)
        H = sparse.block_diag([qm, qm]).tocoo()

        node_deg = np.asarray(H.sum(axis=1)).ravel()
        edge_deg = np.asarray(H.sum(axis=0)).ravel()
        inv_sqrt_node = _safe_inv_sqrt(node_deg)
        inv_edge = np.zeros_like(edge_deg, dtype=np.float64)
        nz = edge_deg > 0
        inv_edge[nz] = 1.0 / edge_deg[nz]

        P = (sparse.diags(inv_sqrt_node) @ H @ sparse.diags(inv_edge)).tocoo()
        Q = (H.T @ sparse.diags(inv_sqrt_node)).tocoo()
        return self._to_torch_sparse(P), self._to_torch_sparse(Q)

    def _build_transition_adj(
        self,
        questions: np.ndarray,
        responses: np.ndarray,
        masks: np.ndarray,
        num_questions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """出 / 入转移邻接：相邻交互计数加自环后行归一化。"""
        inter = questions + (1 - responses) * num_questions
        pair_valid = (masks[:, :-1] & masks[:, 1:]).astype(bool)
        src = inter[:, :-1][pair_valid]
        dst = inter[:, 1:][pair_valid]
        size = 2 * num_questions
        # Duplicate (src, dst) entries are summed on CSR conversion.
        counts = sparse.coo_matrix(
            (np.ones(src.size), (src, dst)), shape=(size, size)
        ).tocsr()
        # Transpose before adding self-loops so each direction is normalized
        # independently.
        return (
            self._row_normalize(counts + sparse.eye(size)),
            self._row_normalize(counts.T + sparse.eye(size)),
        )

    def _row_normalize(self, mat: sparse.spmatrix) -> torch.Tensor:
        """Row-stochastic 归一化，零行保持为零。"""
        rowsum = np.asarray(mat.sum(axis=1)).ravel()
        r_inv = np.zeros_like(rowsum, dtype=np.float64)
        nz = rowsum > 0
        r_inv[nz] = 1.0 / rowsum[nz]
        return self._to_torch_sparse((sparse.diags(r_inv) @ mat).tocoo())

    @staticmethod
    def _to_torch_sparse(mat: sparse.coo_matrix) -> torch.Tensor:
        csr = mat.tocsr()
        return torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(np.int64)),
            torch.from_numpy(csr.indices.astype(np.int64)),
            torch.from_numpy(csr.data.astype(np.float32)),
            csr.shape,
        )
