# CSKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [csKT: Addressing cold-start problem in knowledge tracing via kernel bias and cone attention](https://doi.org/10.1016/j.eswa.2024.125988) |
| 作者 | Youheng Bai, Xueyi Li, Zitao Liu, Yaying Huang, Teng Guo, Mingliang Hou, Feng Xia, Weiqi Luo |
| 发表年份 | 2025 |
| 发表刊物或会议 | Expert Systems with Applications, 266, 125988 |

## 原文摘要

> Knowledge tracing (KT) is the task of predicting students' future performances based on their past interactions in online learning systems. When new students enter the system with short interaction sequences, the cold-start problem commonly arises in KT. Although existing deep learning based KT models exhibit impressive performance, it remains challenging for these models to be trained on short student interaction sequences and maintain stable prediction accuracy as the number of student interactions increases. In this paper, we propose cold-start KT (csKT) to address this problem. Specifically, csKT employs kernel bias to guide learning from short sequences and ensure accurate predictions for longer sequences, and it also introduces cone attention to better capture complex hierarchical relationships between knowledge components in cold-start scenarios. We evaluate csKT on four public real-world educational datasets, where it demonstrated superior performance over a broad range of deep learning based KT models using common evaluation metrics in cold-start scenarios. Additionally, we conduct ablation studies and produce visualizations to verify the effectiveness of our csKT model. In support of reproducible research, we have made all datasets and the corresponding code publicly accessible at https://pykt.org/.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
