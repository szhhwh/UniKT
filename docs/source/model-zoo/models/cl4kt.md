# CL4KT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Contrastive Learning for Knowledge Tracing](https://doi.org/10.1145/3485447.3512105) |
| 作者 | Wonsung Lee, Jaeyoon Chun, Youngmin Lee, Kyoungsoo Park, Sungrae Park |
| 发表年份 | 2022 |
| 发表刊物或会议 | Proceedings of the ACM Web Conference 2022 |
| DOI | [10.1145/3485447.3512105](https://doi.org/10.1145/3485447.3512105) |

## 原文摘要

> Knowledge tracing is the task of understanding student's knowledge acquisition processes by estimating whether to solve the next question correctly or not. Most deep learning-based methods tackle this problem by identifying hidden representations of knowledge states from learning histories. However, due to the sparse interactions between students and questions, the hidden representations can be easily over-fitted and often fail to capture student's knowledge states accurately. This paper introduces a contrastive learning framework for knowledge tracing that reveals semantically similar or dissimilar examples of a learning history and stimulates to learn their relationships. To deal with the complexity of knowledge acquisition during learning, we carefully design the components of contrastive learning, such as architectures, data augmentation methods, and hard negatives, taking into account pedagogical rationales. Our extensive experiments on six benchmarks show statistically significant improvements from the previous methods. Further analysis shows how our methods contribute to improving knowledge tracing performances.


## 代码仓库

[UpstageAI/cl4kt](https://github.com/UpstageAI/cl4kt)
