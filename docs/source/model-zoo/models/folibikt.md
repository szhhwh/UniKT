# FoLibiKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Forgetting-aware Linear Bias for Attentive Knowledge Tracing](https://doi.org/10.1145/3583780.3615191) |
| 作者 | Yoonjin Im, Eunseong Choi, Heejin Kook, Jongwuk Lee |
| 发表年份 | 2023 |
| 发表刊物或会议 | Proceedings of the 32nd ACM International Conference on Information and Knowledge Management（CIKM 2023） |

## 原文摘要

> Knowledge Tracing (KT) aims to track proficiency based on a question-solving history, allowing us to offer a streamlined curriculum. Recent studies actively utilize attention-based mechanisms to capture the correlation between questions and combine it with the learner's characteristics for responses. However, our empirical study shows that existing attention-based KT models neglect the learner's forgetting behavior, especially as the interaction history becomes longer. This problem arises from the bias that overprioritizes the correlation of questions while inadvertently ignoring the impact of forgetting behavior. This paper proposes a simple-yet-effective solution, namely Forgetting-aware Linear Bias (FoLiBi), to reflect forgetting behavior as a linear bias. Despite its simplicity, FoLiBi is readily equipped with existing attentive KT models by effectively decomposing question correlations with forgetting behavior. FoLiBi plugged with several KT models yields a consistent improvement of up to 2.58% in AUC over state-of-the-art KT models on four benchmark datasets.


## 代码仓库

- [skewondr/FoLiBi](https://github.com/skewondr/FoLiBi)
- [pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
