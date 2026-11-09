# SparseKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Towards Robust Knowledge Tracing Models via k-Sparse Attention](https://doi.org/10.1145/3539618.3592073) |
| 作者 | Shuyan Huang, Zitao Liu, Xiangyu Zhao, Weiqi Luo, Jian Weng |
| 发表年份 | 2023 |
| 发表刊物或会议 | Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval |

## 原文摘要

> Knowledge tracing (KT) is the problem of predicting students' future performance based on their historical interaction sequences. With the advanced capability of capturing contextual long-term dependency, attention mechanism becomes one of the essential components in many deep learning based KT (DLKT) models. In spite of the impressive performance achieved by these attentional DLKT models, many of them are often vulnerable to run the risk of overfitting, especially on small-scale educational datasets. Therefore, in this paper, we propose sparseKT, a simple yet effective framework to improve the robustness and generalization of the attention based DLKT approaches. Specifically, we incorporate a k-selection module to only pick items with the highest attention scores. We propose two sparsification heuristics: (1) soft-thresholding sparse attention and (2) top-K sparse attention. We show that our sparseKT is able to help attentional KT models get rid of irrelevant student interactions and improve the predictive performance when compared to 11 state-of-the-art KT models on three publicly available real-world educational datasets. To encourage reproducible research, we make our data and code publicly available at https://github.com/pykt-team/pykt-toolkit1..


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
