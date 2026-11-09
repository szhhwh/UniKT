# BDGKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [BDGKT: Bidirectional Dynamic Graph Knowledge Tracing](https://doi.org/10.1016/j.knosys.2026.115532) |
| 作者 | Xinjia Ou, Tao Huang, Shengze Hu, Huali Yang, Zhuoran Xu, Junjie Hu, Jing Geng |
| 发表年份 | 2026 |
| 发表刊物或会议 | Knowledge-Based Systems |
| DOI | [10.1016/j.knosys.2026.115532](https://doi.org/10.1016/j.knosys.2026.115532) |

## 原文摘要

> Knowledge tracing (KT) aims to model the evolution of students' knowledge states by analyzing their historical learning trajectories and predicting future performance. However, current KT methods primarily focus on unidirectional relationship modeling, overlooking the bidirectional dynamic interaction mechanisms between learners and questions. Student knowledge states shape question adaptability through group patterns (e.g., difficulty calibration), whereas dynamic transformation of question features provides progressive guidance signals for knowledge advancement across learning stages. In this study, we propose a novel bidirectional dynamic graph KT (BDGKT) method for modeling the information flow between students and questions while capturing knowledge state evolution and question characteristic transformation. Specifically, we first introduce a dynamic graph construction based on homogeneous student groups that uses a spatiotemporal constraint strategy to reduce computational costs while improving information propagation quality. Subsequently, we design a bidirectional message propagation mechanism to capture time-evolving bidirectional dynamic signals. To update question nodes (from students to questions), we introduce a state-aware attention mechanism that aggregates student nodes and responses, revealing group-level question commonalities. By contrast, to update student nodes (from questions to students), we propose an evolution mechanism that aggregates question nodes and responses based on timestamps, allowing us to track the evolution of student knowledge states. Extensive experiments on four real-world datasets validate the effectiveness and compatibility of our method. Furthermore, BDGKT improves interpretability by exploring question absolute information (group-agnostic) and relative information (group-dependent).

## 代码仓库

[Oia-10/BDGKT](https://github.com/Oia-10/BDGKT)
