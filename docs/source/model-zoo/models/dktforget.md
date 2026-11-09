# DKTForget

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Augmenting Knowledge Tracing by Considering Forgetting Behavior](https://doi.org/10.1145/3308558.3313565) |
| 作者 | Koki Nagatani, Qian Zhang, Masahiro Sato, Yan-Ying Chen, Francine Chen, Tomoko Ohkuma |
| 发表年份 | 2019 |
| 发表刊物或会议 | The World Wide Web Conference |
| DOI | [10.1145/3308558.3313565](https://doi.org/10.1145/3308558.3313565) |

## 原文摘要

> Computer-aided education systems are now seeking to provide each student with personalized materials based on a student's individual knowledge. To provide suitable learning materials, tracing each student's knowledge over a period of time is important. However, predicting each student's knowledge is difficult because students tend to forget. The forgetting behavior is mainly because of two reasons: the lag time from the previous interaction, and the number of past trials on a question. Although there are a few studies that consider forgetting while modeling a student's knowledge, some models consider only partial information about forgetting, whereas others consider multiple features about forgetting, ignoring a student's learning sequence. In this paper, we focus on modeling and predicting a student's knowledge by considering their forgetting behavior. We extend the deep knowledge tracing model [17], which is a state-of-the-art sequential model for knowledge tracing, to consider forgetting by incorporating multiple types of information related to forgetting. Experiments on knowledge tracing datasets show that our proposed model improves the predictive performance as compared to baselines. Moreover, we also examine that the combination of multiple types of information that affect the behavior of forgetting results in performance improvement.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
