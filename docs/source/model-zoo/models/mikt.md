# MIKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Interpretable Knowledge Tracing with Multiscale State Representation](https://doi.org/10.1145/3589334.3645373) |
| 作者 | Jianwen Sun, Fenghua Yu, Qian Wan, Qing Li, Sannyuya Liu, Xiaoxuan Shen |
| 发表年份 | 2024 |
| 发表刊物或会议 | Proceedings of the ACM Web Conference 2024 |

## 原文摘要

> Knowledge Tracing (KT) is vital for education, continuously monitoring students' knowledge states (mastery of knowledge) as they interact with online education materials. Despite significant advancements in deep learning-based KT models, existing approaches often struggle to strike the right balance in granularity, leading to either overly coarse or excessively fine tracing and representation of students' knowledge states, thereby limiting their performance. Additionally, achieving a high-performing model while ensuring interpretability presents a challenge. Therefore, in this paper, we propose a novel approach called Multiscale-state-based Interpretable Knowledge Tracing (MIKT). Specifically, MIKT traces students' knowledge states on two scales: a coarse-grained representation to trace students' domain knowledge state, and a fine-grained representation to monitor their conceptual knowledge state. Furthermore, the classical psychological measurement model, IRT (Item Response Theory), is introduced to explain the prediction process of MIKT, enhancing its interpretability without sacrificing performance. Additionally, we extended the Rasch representation method to effectively handle scenarios where questions are associated with multiple concepts, making it more applicable to real-world situations. We extensively compared MIKT with 20 state-of-the-art KT models on four widely-used public datasets. Experimental results demonstrate that MIKT outperforms other models while maintaining its interpretability. Moreover, experimental observations have revealed that our proposed extended Rasch representation method not only benefits MIKT but also significantly improves the performance of other KT baseline models. The code can be found at https://github.com/lilstrawberry/MIKT.


## 代码仓库

[lilstrawberry/MIKT](https://github.com/lilstrawberry/MIKT)
