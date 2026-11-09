# STHKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [STHKT: Spatiotemporal Knowledge Tracing with Topological Hawkes Process](https://doi.org/10.1016/j.eswa.2024.125248) |
| 作者 | Shuting Li, Shuanghong Shen, Yu Su, Xinjie Sun, Junyu Lu, Qi Mo, Zhenyi Wu, Qi Liu |
| 发表年份 | 2025 |
| 发表刊物或会议 | Expert Systems with Applications, 259, 125248 |

## 原文摘要

> Knowledge Tracing (KT) is a method that seeks to forecast students’ future performance based on their historical interactions with intelligent tutoring systems. Various KT techniques have been developed from distinct perspectives, such as probability, logic, and deep learning, to scrutinize interaction sequences. However, most current methods predominantly concentrate on temporal or structural information, considering simple question sequences or interaction sequences, but fail to describe the topological dependencies between sequences, which can lead to changes in causal structures,as referred to as dynamic structural information in the paper. The dynamic information includes the clustering effect among concepts and questions, as well as the group effect among students and questions: Questions with the same knowledge concepts tend to present increasingly similar response patterns in the sequence, and concurrently, students who answer similar questions often exhibit closer knowledge states. We introduce the STHKT method, which employs a tripartite heterogeneous graph to capture relationships among concepts, questions, and students. By incorporating temporal information through a topological graph, we employ two Graph Convolution Networks (GCNs) to model the two effects separately. Following this, we apply the Hawkes Process to introduce a spatiotemporal attention mechanism that combines both effects. Through comparative experiments, ablation studies and case studies, we validate the efficacy of key components in the model. Experiment results indicate that the STHKT model can effectively model the two major effects, integrate dynamic structural information, and to some extent improve the accuracy of predicting students’ response probabilities. It is the first time that these two effects have been considered in the process of knowledge tracing, and it is adaptable to large-scale online education platforms.
