# TFKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Time-frequency domain coupling-Oriented knowledge tracing via learning behavior decoupling](https://doi.org/10.1016/j.knosys.2025.115236) |
| 作者 | Tao Huang, Gang Xue, Junjie Hu, Shengze Hu, Huali Yang, Xinjia Ou, Zhichao Zhang |
| 发表年份 | 2026（2025 年在线发表） |
| 发表刊物或会议 | Knowledge-Based Systems, 335, 115236 |

## 原文摘要

> Knowledge tracing (KT) determines students' knowledge state by analyzing their learning behaviors, serving as an important technology in the intelligent education transformation initiative. Recent KT research has focused on interpretability and the representation of intrinsic cognitive mechanisms. However, most existing methods primarily focus on temporal features derived from student interactions, ignoring the dynamic nature of knowledge state evolution. To address this issue, we propose a novel method called time-frequency domain coupling-oriented knowledge tracing via learning behavior decoupling (TFKT). This method models both the temporal dynamics and frequency-domain characteristics of students' knowledge state to capture their fine-grained evolution. Particularly, we first decouple students' learning behaviors into guessing, slipping, and normal responses, and extract biased behavioral features through a dual-channel mechanism to capture knowledge construction representation. Subsequently, a Chebyshev Kolmogorov-Arnold network combined with a gated residual network is used to model multi-scale correlations among exercises, establishing the initial knowledge state space. Finally, a causal Fourier network with learnable masks projects the knowledge state into the frequency domain, decomposing it into low, middle, and high-frequency components for joint time-frequency coupling using a monotonic attention mechanism to simulate students' knowledge state changes at a fine-grained level. Extensive experiments across four datasets validated the explainability and feasibility of the TFKT method in modeling the knowledge mastery evolution of students.

## 代码仓库

[anonymous11111-create/TFKT/tree/master](https://github.com/anonymous11111-create/TFKT/tree/master)
