# MocKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [MoC-KT: Mixture of Convolutions for Knowledge Tracing](https://doi.org/10.1145/3847658) |
| 作者 | Mingliang Hou, Zitao Liu, Renqiang Luo, Jiliang Tang, Weiqi Luo, Feng Xia |
| 发表年份 | 2026 |
| 发表刊物或会议 | ACM Transactions on Information Systems |

## 原文摘要

> Knowledge tracing (KT) aims to predict learners’ mastery levels of knowledge components (KCs) or test items based on their interaction records with educational content. Despite significant advancements in KT models, such as RNN-based sequence models and Transformer-based attention models, a critical limitation persists: the inability to account for the heterogeneity in learners’ cognitive behavior data. This limitation leads to the “cognitive mirage” phenomenon, where seemingly similar historical interaction sequences result in divergent future outcomes, hindering prediction accuracy. To address this challenge, we propose a novel framework, mixture of convolutions for KT (MoC-KT). The framework introduces multi-scale causal convolutional kernels and adaptive segmentation to disentangle learners’ long-term stable progression from short-term fluctuations, such as guessing or fatigue. Additionally, the Kerple-enhanced attention mechanism incorporates a distance-sensitive decay function to prioritize local dependencies while suppressing irrelevant distant interactions. This mechanism effectively balances attention between capturing local short-term fluctuations and preserving global long-term progression, thereby mitigating the cognitive mirage problem caused by learner data heterogeneity. Experimental results on four real-world datasets show that MoC-KT consistently outperforms 26 state-of-the-art KT models, offering higher predictive accuracy and more robust handling of complex learning data. The source code of MoC-KT is available at https://pykt.org/ .


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
