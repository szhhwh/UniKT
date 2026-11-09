# GKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Graph-Based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network](https://doi.org/10.1145/3350546.3352513) |
| 作者 | Hiromi Nakagawa, Yusuke Iwasawa, Yutaka Matsuo |
| 发表年份 | 2019 |
| 发表刊物或会议 | IEEE/WIC/ACM International Conference on Web Intelligence |
| DOI | [10.1145/3350546.3352513](https://doi.org/10.1145/3350546.3352513) |

## 原文摘要

> Recent advancements in computer-assisted learning systems have caused an increase in the research of knowledge tracing, wherein student performance on coursework exercises is predicted over time. From the viewpoint of data structure, the coursework can be potentially structured as a graph. Incorporating this graph-structured nature into the knowledge tracing model as a relational inductive bias can improve its performance; however, previous methods, such as deep knowledge tracing, did not consider such a latent graph structure. Inspired by the recent successes of the graph neural network (GNN), we herein propose a GNN-based knowledge tracing method, i.e., graph-based knowledge tracing. Casting the knowledge structure as a graph enabled us to reformulate the knowledge tracing task as a time-series node-level classification problem in the GNN. As the knowledge graph structure is not explicitly provided in most cases, we propose various implementations of the graph structure. Empirical validations on two open datasets indicated that our method could potentially improve the prediction of student performance and demonstrated more interpretable predictions compared to those of the previous methods, without the requirement of any additional information.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
