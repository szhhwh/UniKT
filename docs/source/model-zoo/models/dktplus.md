# DKTPlus

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization](https://doi.org/10.1145/3231644.3231647) |
| 作者 | Chun-Kit Yeung, Dit-Yan Yeung |
| 发表年份 | 2018 |
| 发表刊物或会议 | Proceedings of the Fifth Annual ACM Conference on Learning at Scale |
| DOI | [10.1145/3231644.3231647](https://doi.org/10.1145/3231644.3231647) |

## 原文摘要

> Knowledge tracing is one of the key research areas for empowering personalized education. It is a task to model students' mastery level of a knowledge component (KC) based on their historical learning trajectories. In recent years, a recurrent neural network model called deep knowledge tracing (DKT) has been proposed to handle the knowledge tracing task and literature has shown that DKT generally outperforms traditional methods. However, through our extensive experimentation, we have noticed two major problems in the DKT model. The first problem is that the model fails to reconstruct the observed input. As a result, even when a student performs well on a KC, the prediction of that KC's mastery level decreases instead, and vice versa. Second, the predicted performance for KCs across time-steps is not consistent. This is undesirable and unreasonable because student's performance is expected to transit gradually over time. To address these problems, we introduce regularization terms that correspond to reconstruction and waviness to the loss function of the original DKT model to enhance the consistency in prediction. Experiments show that the regularized loss function effectively alleviates the two problems without degrading the original task of DKT.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
