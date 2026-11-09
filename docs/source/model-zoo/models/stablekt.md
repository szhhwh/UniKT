# StableKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Enhancing Length Generalization for Attention Based Knowledge Tracing Models with Linear Biases](https://doi.org/10.24963/ijcai.2024/654) |
| 作者 | Xueyi Li, Youheng Bai, Teng Guo, Zitao Liu, Yaying Huang, Xiangyu Zhao, Feng Xia, Weiqi Luo, Jian Weng |
| 发表年份 | 2024 |
| 发表刊物或会议 | Proceedings of the Thirty-ThirdInternational Joint Conference on Artificial Intelligence |

## 原文摘要

> Knowledge tracing (KT) is the task of predicting students' future performance based on their historical learning interaction data. With the rapid advancement of attention mechanisms, many attention based KT models are developed. However, existing attention based KT models exhibit performance drops as the number of student interactions increases beyond the number of interactions on which the KT models are trained. We refer to this as the length generalization of KT model. In this paper, we propose stableKT to enhance length generalization that is able to learn from short sequences and maintain high prediction performance when generalizing on long sequences. Furthermore, we design a multi-head aggregation module to capture the complex relationships between questions and the corresponding knowledge components (KCs) by combining dot-product attention and hyperbolic attention. Experimental results on three public educational datasets show that our model exhibits robust capability of length generalization and outperforms all baseline models in terms of AUC. To encourage reproducible research, we make our data and code publicly available at https://pykt.org.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
