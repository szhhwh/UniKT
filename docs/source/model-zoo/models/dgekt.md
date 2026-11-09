# DGEKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [DGEKT: A Dual Graph Ensemble Learning Method for Knowledge Tracing](https://doi.org/10.1145/3638350) |
| 作者 | Chaoran Cui, Yumo Yao, Chunyun Zhang, Hebo Ma, Yuling Ma, Zhaochun Ren, Chen Zhang, James Ko |
| 发表年份 | 2024 |
| 发表刊物或会议 | ACM Transactions on Information Systems, 42(3) |

## 原文摘要

> Knowledge tracing aims to trace students' evolving knowledge states by predicting their future performance on concept-related exercises. Recently, some graph-based models have been developed to incorporate the relationships between exercises to improve knowledge tracing, but only a single type of relationship information is generally explored. In this article, we present a novel Dual Graph Ensemble learning method for Knowledge Tracing (DGEKT), which establishes a dual graph structure of students' learning interactions to capture the heterogeneous exercise--concept associations and interaction transitions by hypergraph modeling and directed graph modeling, respectively. To combine the dual graph models, we introduce the technique of online knowledge distillation. This choice arises from the observation that, while the knowledge tracing model is designed to predict students' responses to the exercises related to different concepts, it is optimized merely with respect to the prediction accuracy on a single exercise at each step. With online knowledge distillation, the dual graph models are adaptively combined to form a stronger ensemble teacher model, which provides its predictions on all exercises as extra supervision for better modeling ability. In the experiments, we compare DGEKT against eight knowledge tracing baselines on three benchmark datasets, and the results demonstrate that DGEKT achieves state-of-the-art performance.

## 代码仓库

[Yumo216/DGEKT](https://github.com/Yumo216/DGEKT)
