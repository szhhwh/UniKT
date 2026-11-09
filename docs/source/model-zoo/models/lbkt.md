# LBKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Learning Behavior-oriented Knowledge Tracing](https://doi.org/10.1145/3580305.3599407) |
| 作者 | Bihan Xu, Zhenya Huang, Jiayu Liu, Shuanghong Shen, Qi Liu, Enhong Chen, Jinze Wu, Shijin Wang |
| 发表年份 | 2023 |
| 发表刊物或会议 | Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining |

## 原文摘要

> Exploring how learners' knowledge states evolve during the learning activities is a critical task in online learning systems, which can facilitate personalized services downstream, such as course recommendation. Most of existing methods have devoted great efforts to analyzing learners' knowledge states according to their responses (i.e., right or wrong) to different questions. However, the significant effect of learners' learning behaviors (e.g., answering speed, the number of attempts) is omitted, which can reflect their knowledge acquisition deeper and ensure the reliability of the response. In this paper, we propose a Learning Behavior-oriented Knowledge Tracing (LBKT) model, with the goal of explicitly exploring the learning behavior effects on learners' knowledge states. Specifically, we first analyze and summarize several dominated learning behaviors including Speed, Attempts and Hints in the learning process. As the characteristics of different learning behaviors vary greatly, we separately estimate their various effects on learners' knowledge acquisition in a quantitative manner. Then, considering that different learning behaviors are closely dependent with each other, we assess the fused effect of multiple learning behaviors by capturing their complex dependent patterns. Finally, we integrate the forgetting factor with learners' knowledge acquisition to comprehensively update their changing knowledge states in learning. Extensive experimental results on several public datasets demonstrate that our model generates better performance prediction for learners against existing methods. Moreover, LBKT shows good interpretability in tracking learners' knowledge state by incorporating the learning behavior effects. Our codes are available at https://github.com/xbh0720/LBKT.

## 代码仓库

[xbh0720/LBKT](https://github.com/xbh0720/LBKT)
