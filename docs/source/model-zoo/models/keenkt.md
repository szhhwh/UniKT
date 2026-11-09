# KeenKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [KeenKT: Knowledge Mastery-State Disambiguation for Knowledge Tracing](https://doi.org/10.1609/aaai.v40i23.39004) |
| 作者 | Zhifei Li, Lifan Chen, Jiali Yi, Xiaoju Hou, Yue Zhao, Wenxin Huang, Miao Zhang, Kui Xiao, Bing Yang |
| 发表年份 | 2026 |
| 发表刊物或会议 | Proceedings of the AAAI Conference on Artificial Intelligence, 40(23) |

## 原文摘要

> Knowledge Tracing (KT) aims to dynamically model a student’s mastery of knowledge concepts based on their historical learning interactions. Most current methods rely on single-point estimates, which cannot distinguish true ability from outburst or carelessness, creating ambiguity in judging mastery. To address this issue, we propose a Knowledge Mastery-State Disambiguation for Knowledge Tracing model (KeenKT), which represents a student’s knowledge state at each interaction using a Normal-Inverse-Gaussian (NIG) distribution, thereby capturing the fluctuations in student learning behaviors. Furthermore, we design an NIG-distance-based attention mechanism to model the dynamic evolution of the knowledge state. In addition, we introduce a diffusion-based denoising reconstruction loss and a distributional contrastive learning loss to enhance the model’s robustness. Extensive experiments on six public datasets demonstrate that KeenKT outperforms state-of-the-art KT models in terms of prediction accuracy and sensitivity to behavioral fluctuations. The proposed method yields the maximum AUC improvement of 5.85% and the maximum ACC improvement of 6.89%.


## 代码仓库

[HubuKG/KeenKT](https://github.com/HubuKG/KeenKT)
