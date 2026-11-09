# extraKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Extending Context Window of Attention Based Knowledge Tracing Models via Length Extrapolation](https://doi.org/10.3233/faia240651) |
| 作者 | Xueyi Li, Youheng Bai, Teng Guo, Ying Zheng, Mingliang Hou, Bojun Zhan, Yaying Huang, Zitao Liu, Boyu Gao, Weiqi Luo |
| 发表年份 | 2024 |
| 发表刊物或会议 | Frontiers in Artificial Intelligence and Applications |

## 原文摘要

> Knowledge tracing (KT) is a prediction task that aims to predict students’ future performance based on their past learning data. The rapid progress in attention mechanisms has led to the emergence of various high-performing attention based KT models. However, in online or personalized education settings, students’ varying learning paths result in different lengths of student interaction sequences, which poses a significant challenge for attention based KT models as their context window sizes are fixed during both training and prediction stages. We refer to this as the length extrapolation of KT model. In this paper, we propose extraKT to facilitate better extrapolation that learn from student interactions with a short context window and continue to perform well across various longer context window sizes at prediction stage. Specifically, we negatively bias attention scores with linearly decreasing penalties that are proportional to query-key distance, which efficiently represents short-term forgetting characteristics of student knowledge states. We conduct comprehensive and rigorous experiments on three real-world educational datasets. The results show that our extraKT model exhibits robust length extrapolation capability and outperforms state-of-the-art baseline models in terms of AUC and accuracy. To encourage reproducible research, we merge our data and code to the publicly available pyKT benchmark at https://github.com/pykt-team/pykt-toolkit.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
