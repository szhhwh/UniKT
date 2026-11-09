# PSKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Remembering Is Not Applying: Interpretable Knowledge Tracing for Problem-solving Processes](https://doi.org/10.1145/3664647.3681049) |
| 作者 | Tao Huang, Xinjia Ou, Huali Yang, Shengze Hu, Jing Geng, Junjie Hu, Zhuoran Xu |
| 发表年份 | 2024 |
| 发表刊物或会议 | Proceedings of the 32nd ACM International Conference on Multimedia（ACM MM 2024） |

## 原文摘要

> Knowledge Tracing (KT) is a critical service in distance education, predicting students' future performance based on their responses to learning resources. The reasonable assessment of the knowledge state, along with accurate response prediction, is crucial for KT. However, existing KT methods prioritize fitting results and overlook attention to the problem-solving process. They equate the knowledge students memorize before problem-solving with the knowledge that can be acquired or applied during problem-solving, leading to dramatic fluctuations in knowledge states between mastery and non-mastery, with low interpretability. This paper explores knowledge transformation in problem-solving and proposes an interpretable model, Problem-solving Knowledge Tracing (PSKT). Specifically, we first present a knowledge-centered problem representation that enhances its expression by adjusting problem variability. Then, we meticulously designed a Sequential Neural Network (SNN) with three stages: (1) Before problem-solving, we model students' personalized problem space and simulate their acquisition of problem-related knowledge through a gating mechanism. (2) During problem-solving, we evaluate knowledge application and calculate response with a four-parameter IRT. (3) After problem-solving, we quantify student knowledge internalization and forgetting using an incremental indicator. The SNN, inspired by problem-solving and constructivist learning theories, is an interpretable model that attributes learner performance to subjective problems (difficulty, discrimination), objective knowledge (knowledge acquisition and application), and behavior (guessing and slipping). Experimental results show PSKT's advantages in prediction accuracy, reasonable knowledge state assessment, and learning process explanation. The code is available at https://github.com/Oia-10/PSKT.


## 代码仓库

[Oia-10/PSKT](https://github.com/Oia-10/PSKT)
