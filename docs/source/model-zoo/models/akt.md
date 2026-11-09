# AKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Context-Aware Attentive Knowledge Tracing](https://doi.org/10.1145/3394486.3403282) |
| 作者 | Aritra Ghosh, Neil Heffernan, Andrew S. Lan |
| 发表年份 | 2020 |
| 发表刊物或会议 | Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining |
| DOI | [10.1145/3394486.3403282](https://doi.org/10.1145/3394486.3403282) |

## 原文摘要

> Knowledge tracing (KT) refers to the problem of predicting future learner performance given their past performance in educational applications. Recent developments in KT using flexible deep neural network-based models excel at this task. However, these models often offer limited interpretability, thus making them insufficient for personalized learning, which requires using interpretable feedback and actionable recommendations to help learners achieve better learning outcomes. In this paper, we propose attentive knowledge tracing (AKT), which couples flexible attention-based neural network models with a series of novel, interpretable model components inspired by cognitive and psychometric models. AKT uses a novel monotonic attention mechanism that relates a learner's future responses to assessment questions to their past responses; attention weights are computed using exponential decay and a context-aware relative distance measure, in addition to the similarity between questions. Moreover, we use the Rasch model to regularize the concept and question embeddings; these embeddings are able to capture individual differences among questions on the same concept without using an excessive number of parameters. We conduct experiments on several real-world benchmark datasets and show that AKT outperforms existing KT methods (by up to $6\%$ in AUC in some cases) on predicting future learner responses. We also conduct several case studies and show that AKT exhibits excellent interpretability and thus has potential for automated feedback and personalization in real-world educational settings.


## 代码仓库

[arghosh/AKT](https://github.com/arghosh/AKT)
