# RobustKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Enhancing Knowledge Tracing through Decoupling Cognitive Pattern from Error-Prone Data](https://doi.org/10.1145/3696410.3714486) |
| 作者 | Teng Guo, Yu Qin, Yubin Xia, Mingliang Hou, Zitao Liu, Feng Xia, Weiqi Luo |
| 发表年份 | 2025 |
| 发表刊物或会议 | Proceedings of the ACM on Web Conference 2025 |

## 原文摘要

> Knowledge tracing (KT) aims to predict students' future performance based on their past learning activities. However, no one is perfect. Factors such as carelessness, fatigue, and stress often cause students to make mistakes on problems they have already mastered, leading to anomalies in their historical learning data. These anomalies disrupt inherent patterns in the data, misleading the KT model. Extracting cognitive patterns that accurately reflect students' knowledge mastery from such error-prone data remains a significant challenge. Against this background, this paper proposes a novel KT method named RoubstKT, inspired by educational measurement theory and frequency-based decomposition. A cognitive decoupling analyzer is proposed to decouple the student's cognitive pattern and random factors from the data through smoothing and subtraction operations, then recombine them using a gating mechanism or adaptive parameter fusion strategy. To more effectively diagnose students' knowledge mastery, we employ a decay-based attention mechanism that focuses on random behaviors at adjacent time steps. We conducted comprehensive experiments based on real-world datasets and targeted datasets with added random noise. The experimental results demonstrated the effectiveness of the proposed method.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
