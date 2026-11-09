# ClusterKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Cluster-Driven Knowledge Tracing: Joint Learning-Forgetting Effects Modeling via State Dependency](https://doi.org/10.1016/j.eswa.2025.128619) |
| 作者 | Huazheng Luo, Zhichang Zhang, Lingyun Cui, Huiming Hu, Bichen Yu |
| 发表年份 | 2025 |
| 发表刊物或会议 | Expert Systems with Applications |
| DOI | [10.1016/j.eswa.2025.128619](https://doi.org/10.1016/j.eswa.2025.128619) |

## 原文摘要

> Knowledge tracing (KT) is a crucial technique in intelligent tutoring systems, which involves the analyses of learners' interactions with educational contents to trace their knowledge states. This state-tracing capacity helps to identify learners' individual learning needs and deliver personalized guidance. In practice, learners' knowledge states demonstrate continuous evolution driven by the dynamic interplay between knowledge acquisition and decay throughout long-term learning process. Most existing KT methods assess these learning and forgetting effects by leveraging learners' proficiency of individual knowledge concepts in isolation, leading to scattered and redundant state estimation results. In this paper, we propose a cluster-driven knowledge tracing (ClusterKT) model, which aims to jointly estimate the learning-forgetting effects based on the latent dependencies of knowledge states. Specifically, we introduce a k-means inspired clustering algorithm that can easily categorizes high-correlation exercises and performances into same clusters. Given the inherent state dependency, these well-separated clusters facilitate (1) the assessment of current learning gains drawing on their previous performances in related exercises, and (2) the quantification of knowledge decay over time by calculating three essential forgetting factors. Extensive experimental evaluations across four widely used datasets demonstrate that ClusterKT achieves enhanced predictive accuracy and heightened interpretability over current KT approaches. The source code is publicly accessible at https://github.com/Lzhenghua/ClusterKT.


## 代码仓库

[Lzhenghua/ClusterKT](https://github.com/Lzhenghua/ClusterKT)
