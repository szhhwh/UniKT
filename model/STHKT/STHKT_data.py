"""STHKT 模型数据处理模块。"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class STHKTDataset(Dataset):
    """STHKT 数据集。

    每个样本返回 ``(question, response, mask)``：
        question: 题目序列 [S]（有效位 = 原始 id + 1，0 = padding）
        response: 答案序列 [S]（有效位 0/1，padding 位 2）
        mask: 有效位置掩码 [S]
    """

    def __init__(self, question, response, mask):
        self.question = question
        self.response = response
        self.mask = mask

    def __len__(self) -> int:
        return len(self.question)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.question[idx], dtype=torch.long),
            torch.tensor(self.response[idx], dtype=torch.long),
            torch.tensor(self.mask[idx], dtype=torch.bool),
        )


def _csr_layout(n_rows: int, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(order, ptr)`` grouping edges by source node.

    ``order`` sorts edges so rows are contiguous; ``ptr [n_rows+1]`` gives each
    row's slice boundaries.
    """
    order = np.argsort(src, kind="stable")
    deg = np.bincount(src, minlength=n_rows)
    ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(deg, out=ptr[1:])
    return order, ptr


def _build_gcn_csr(
    nq: int, no: int, q_idx: np.ndarray, o_idx: np.ndarray
) -> dict[str, np.ndarray]:
    """Build the row-restricted normalized CSR of one bipartite subgraph.

    The subgraph is the undirected question–other bipartite graph plus a
    self-loop on every node (``Ã = A + I``). Edge weights follow the GCN
    normalization ``w_uv = 1/sqrt(d̃_u · d̃_v)`` where ``d̃`` counts the
    self-loop. Only question rows are materialized.

    Args:
        nq: question node count (padding node 0 included).
        no: other-side (concept / student) node count.
        q_idx: question endpoints of every edge, in [0, nq).
        o_idx: other-side endpoints of every edge, in [0, no).

    Returns:
        dict with ``ptr [nq+1]``, neighbor ``idx [M]``, normalized ``w [M]``,
        neighbor-table size ``no``, and per-question self-loop weight
        ``self_w [nq] = 1/d̃_q``.
    """
    deg_q = np.bincount(q_idx, minlength=nq).astype(np.float64)
    deg_o = np.bincount(o_idx, minlength=no).astype(np.float64)
    dtilde_q = deg_q + 1.0
    dtilde_o = deg_o + 1.0

    order, ptr = _csr_layout(nq, q_idx)
    w = 1.0 / np.sqrt(dtilde_q[q_idx] * dtilde_o[o_idx])
    self_w = 1.0 / dtilde_q

    return {
        "ptr": ptr,
        "idx": o_idx[order].astype(np.int64),
        "w": w[order].astype(np.float32),
        "no": no,
        "self_w": self_w.astype(np.float32),
    }


def _build_out_csr(
    n_rows: int, src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build a plain unnormalized out-edge CSR (random-walk transitions)."""
    order, ptr = _csr_layout(n_rows, src)
    return ptr, dst[order].astype(np.int64)


class STHKTModelData(QuestionModelData):
    """STHKT 模型数据加载器。"""

    @override
    def prepare_data(self, rc: Any) -> tuple:
        """准备训练 / 验证 / 测试数据与图结构元信息。

        Returns:
            (train_dataset, val_dataset, test_dataset, info) 元组，其中
            ``info`` 含 ``graph_data``、``num_questions``、``num_skills``、
            ``num_users``、``max_seq_len``。
        """
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")
        num_users = self.data_src.get_metadata("num_users")
        max_seq_len = self.data_src.get_metadata("max_seq_len")

        user_sequence, user_response, user_mask, _ = self.load_sequence_data()

        # Shift questions to 1-based (0 = padding); mark padded responses as 2
        # so they index a dedicated row of the answer embedding.
        question = np.where(user_mask == 1, user_sequence + 1, 0)
        response = np.where(user_mask == 1, user_response, 2)

        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        if fold_idx is None:
            raise ValueError("K-fold cross-validation is not enabled (fold < 0).")

        train_slices, val_slices, test_slices = self.split_kfold_data(
            question, response, user_mask, fold_idx=fold_idx
        )
        train_dataset = STHKTDataset(*train_slices)
        val_dataset = STHKTDataset(*val_slices)
        test_dataset = STHKTDataset(*test_slices)

        logger.info(
            f"STHKT dataset sizes: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

        graph_data = self._build_graph_data(num_questions, num_skills, num_users)

        info = {
            "graph_data": graph_data,
            "num_questions": num_questions,
            "num_skills": num_skills,
            "num_users": num_users,
            "max_seq_len": max_seq_len,
        }
        return train_dataset, val_dataset, test_dataset, info

    def _build_graph_data(
        self, num_questions: int, num_skills: int, num_users: int
    ) -> dict[str, Any]:
        """Build the graph-side inputs for both graph modules.

        Question–concept pairs come from the Q-matrix relation;
        question–student pairs from unique (student, question) occurrences
        in the split sequences.
        """
        nq = num_questions + 1  # id 0 is the padding question
        nk = num_skills
        ns = num_users

        qkc_matrix = self.build_relationship_matrix(("question", "has", "skill"))
        qkc_q, qkc_kc = np.nonzero(qkc_matrix)
        qkc_q = qkc_q + 1  # shift into graph question space

        # No user–question relation table exists; 'user' holds real student
        # ids in the split sequences.
        pairs = (
            self.data_src.get_split_question_sequence_data()
            .select(["user", "question"])
            .unique()
        )
        qs_q = pairs["question"].to_numpy() + 1
        qs_s = pairs["user"].to_numpy()

        gcn_qkc = _build_gcn_csr(nq, nk, qkc_q, qkc_kc)
        gcn_qs = _build_gcn_csr(nq, ns, qs_q, qs_s)

        # Random-walk out-edge CSRs are indexed in the joint node space
        # [0, nq) questions | [nq, nq+nk) concepts | [nq+nk, N) students,
        # with row counts covering the full space so any walked node id can
        # look up its transitions directly.
        joint_kc = qkc_kc + nq
        joint_s = qs_s + nq + nk
        N = nq + nk + ns
        rw = {
            "contain": _build_out_csr(N, qkc_q, joint_kc),
            "contain_ed": _build_out_csr(N, joint_kc, qkc_q),
            "finish_ed": _build_out_csr(N, qs_q, joint_s),
            "finish": _build_out_csr(N, joint_s, qs_q),
        }

        logger.info(
            f"STHKT graph: nq={nq}, nk={nk}, ns={ns}, "
            f"qkc_edges={len(qkc_q)}, qs_edges={len(qs_q)}"
        )

        return {
            "gcn": {"qkc": gcn_qkc, "qs": gcn_qs},
            "rw": rw,
        }
