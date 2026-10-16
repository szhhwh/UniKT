"""HGKT 模型数据处理模块。

在 LPKT 序列 / 时间词表 / Q-matrix 数据之上，从二值 Q-matrix（exercise
为顶点、concept 为超边）构建超图拉普拉斯因子：
    P = Dv^-1/2·H·De^-1 [Q, M]，Q_f = H^T·Dv^-1/2 [M, Q]，G = P·Q_f
"""

from typing import Any

import numpy as np
import torch
from scipy import sparse
from typing_extensions import override

from model.LPKT.LPKT_data import LPKTModelData
from utils.core import get_logger

logger = get_logger(__name__)


def _safe_inv_sqrt(values: np.ndarray) -> np.ndarray:
    """Element-wise ``x^-0.5`` with zeros kept as zeros."""
    inv_sqrt = np.zeros_like(values, dtype=np.float64)
    nz = values > 0
    inv_sqrt[nz] = values[nz] ** -0.5
    return inv_sqrt


class HGKTModelData(LPKTModelData):
    """HGKT 模型数据加载器（在 LPKT 数据之上追加超图因子）。"""

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与模型所需的元信息。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中 ``info``
            在 LPKT 的基础上追加 ``hyper_factors``。
        """
        train_dataset, val_dataset, test_dataset, info = super().prepare_data(rc)
        q_matrix = self.build_relationship_matrix(("question", "has", "skill"))
        info["hyper_factors"] = self._build_hyper_factors(q_matrix)
        hyper_p, hyper_q = info["hyper_factors"]
        logger.info(f"HGKT hypergraph: P nnz={hyper_p._nnz()}, Q nnz={hyper_q._nnz()}")
        return train_dataset, val_dataset, test_dataset, info

    def _build_hyper_factors(self, q_matrix: np.ndarray) -> tuple:
        """超图拉普拉斯 G = Dv^-1/2·H·De^-1·H^T·Dv^-1/2 的因子分解。

        Args:
            q_matrix: 二值题目-概念矩阵 [Q, M]（γ 平滑仅发生在模型侧）。

        Returns:
            (P, Q_f) = (Dv^-1/2·H·De^-1 [Q, M], H^T·Dv^-1/2 [M, Q])。
        """
        H = sparse.coo_matrix(q_matrix, dtype=np.float64)

        node_deg = np.asarray(H.sum(axis=1)).ravel()
        edge_deg = np.asarray(H.sum(axis=0)).ravel()
        inv_sqrt_node = _safe_inv_sqrt(node_deg)
        inv_edge = np.zeros_like(edge_deg, dtype=np.float64)
        nz = edge_deg > 0
        inv_edge[nz] = 1.0 / edge_deg[nz]

        P = (sparse.diags(inv_sqrt_node) @ H @ sparse.diags(inv_edge)).tocoo()
        Q_f = (H.T @ sparse.diags(inv_sqrt_node)).tocoo()
        return self._to_torch_sparse(P), self._to_torch_sparse(Q_f)

    @staticmethod
    def _to_torch_sparse(mat: sparse.coo_matrix) -> torch.Tensor:
        csr = mat.tocsr()
        return torch.sparse_csr_tensor(
            torch.from_numpy(csr.indptr.astype(np.int64)),
            torch.from_numpy(csr.indices.astype(np.int64)),
            torch.from_numpy(csr.data.astype(np.float32)),
            csr.shape,
        )
