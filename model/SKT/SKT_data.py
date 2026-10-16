"""SKT 模型数据处理模块。

- 题目 / 答案 / 掩码序列（question 级）
- 两张静态图（均由训练折统计，节点为题目）：
  有向转移邻接 ``successor_adj``（答对后继转移计数每题取 TopK 后继）、
  无向相似邻接 ``neighbor_adj``（TopK 后继支撑集的 Jaccard 相似每题取
  TopK 后对称化）
"""

from typing import Any

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class SKTDataset(Dataset):
    """SKT 数据集。

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


class SKTModelData(QuestionModelData):
    """SKT 模型数据加载器。"""

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与模型所需的图结构。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中 ``info``
            为含 ``num_questions``、``max_seq_len``、``neighbor_adj``、
            ``successor_adj`` 的字典。
        """
        num_questions = self.data_src.get_metadata("num_questions")
        max_seq_len = self.data_src.get_metadata("max_question_seq_len")

        data = self.load_split_data()
        sequences = self.pack_sequences(data)
        partition = sequences.partition(rc.data.fold)
        user_sequence = sequences.question
        user_response = sequences.response
        user_mask = sequences.mask

        # Both graphs are derived from the training fold only
        topk = rc.model.graph_topk
        transition = self._build_correct_transition(
            user_sequence[partition.train],
            user_response[partition.train],
            user_mask[partition.train],
            num_questions,
        )
        successor_adj = self._build_successor_adj(transition, topk)
        neighbor_adj = self._build_neighbor_adj(successor_adj, topk)
        logger.info(
            f"SKT graphs: transition nnz={transition.nnz}, "
            f"successor nnz={int(successor_adj.sum())}, "
            f"neighbor nnz={int(neighbor_adj.sum())}"
        )

        dataset = SKTDataset(user_sequence, user_response, user_mask)
        train_dataset, val_dataset, test_dataset = partition.apply(dataset)

        logger.info(
            f"SKT data: num_questions={num_questions}, max_seq_len={max_seq_len}"
        )
        logger.info(
            f"SKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        info = {
            "num_questions": num_questions,
            "max_question_seq_len": max_seq_len,
            "neighbor_adj": neighbor_adj,
            "successor_adj": successor_adj,
        }
        return train_dataset, val_dataset, test_dataset, info

    @staticmethod
    def _build_correct_transition(
        questions: np.ndarray,
        responses: np.ndarray,
        masks: np.ndarray,
        num_questions: int,
    ) -> sparse.csr_matrix:
        """答对后继转移计数矩阵 [Q, Q]（稀疏）。

        统计相邻交互对（q_t 答对 → q_{t+1}）的共现次数，对角线清零。
        """
        pair_valid = (masks[:, :-1] & masks[:, 1:]).astype(bool) & (
            responses[:, :-1] == 1
        )
        src = questions[:, :-1][pair_valid]
        dst = questions[:, 1:][pair_valid]
        counts = sparse.coo_matrix(
            (np.ones(src.size), (src, dst)), shape=(num_questions, num_questions)
        ).tocsr()
        counts.setdiag(0)
        counts.eliminate_zeros()
        return counts

    @staticmethod
    def _topk_rows(weights: sparse.csr_matrix, topk: int) -> np.ndarray:
        """每行保留权重 TopK 的列，返回 bool 邻接 [Q, Q]。"""
        num = weights.shape[0]
        adj = np.zeros((num, num), dtype=bool)
        indptr, indices, data = weights.indptr, weights.indices, weights.data
        for i in range(num):
            lo, hi = indptr[i], indptr[i + 1]
            if hi == lo:
                continue
            k = min(topk, hi - lo)
            cols = indices[lo:hi][np.argpartition(-data[lo:hi], k - 1)[:k]]
            adj[i, cols] = True
        return adj

    @classmethod
    def _build_successor_adj(
        cls, transition: sparse.csr_matrix, topk: int
    ) -> torch.Tensor:
        """有向转移邻接：每题保留转移计数最高的 TopK 个后继。"""
        return torch.from_numpy(cls._topk_rows(transition, topk))

    @classmethod
    def _build_neighbor_adj(
        cls, successor_adj: torch.Tensor, topk: int
    ) -> torch.Tensor:
        """无向相似邻接：后继支撑集的 Jaccard 相似每题取 TopK 后对称化。

        题目 i 的后继集合与题目 j 的后继集合的 Jaccard 相似度作为边权，
        邻接取 ``adj | adj.T`` 保证无向。
        """
        support = sparse.csr_matrix(successor_adj.numpy().astype(np.float64))
        # inter[i, j] = |succ(i) ∩ succ(j)|, computed only at its nonzero positions
        inter = (support @ support.T).tocsr()
        deg = np.asarray(support.sum(axis=1)).ravel()
        inter = inter.tocoo()
        union = deg[inter.row] + deg[inter.col] - inter.data
        jac = sparse.coo_matrix(
            (inter.data / np.maximum(union, 1.0), (inter.row, inter.col)),
            shape=inter.shape,
        ).tocsr()
        jac.setdiag(0)
        jac.eliminate_zeros()
        adj = cls._topk_rows(jac, topk)
        return torch.from_numpy(adj | adj.T)
