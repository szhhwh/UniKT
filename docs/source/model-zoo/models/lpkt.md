# LPKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Learning Process-consistent Knowledge Tracing](https://doi.org/10.1145/3447548.3467237) |
| 作者 | Shuanghong Shen, Qi Liu, Enhong Chen, Zhenya Huang, Wei Huang, Yu Yin, Yu Su, Shijin Wang |
| 发表年份 | 2021 |
| 发表刊物或会议 | ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD) |

## 原文摘要

> Knowledge tracing (KT), which aims to trace students' changing knowledge state during their learning process, has improved students' learning efficiency in online learning systems. Recently, KT has attracted much research attention due to its critical significance in education. However, most of the existing KT methods pursue high accuracy of student performance prediction but neglect the consistency of students' changing knowledge state with their learning process. In this paper, we explore a new paradigm for the KT task and propose a novel model named Learning Process-consistent Knowledge Tracing (LPKT), which monitors students' knowledge state through directly modeling their learning process. Specifically, we first formalize the basic learning cell as the tuple exercise---answer time---answer. Then, we deeply measure the learning gain as well as its diversity from the difference of the present and previous learning cells, their interval time, and students' related knowledge state. We also design a learning gate to distinguish students' absorptive capacity of knowledge. Besides, we design a forgetting gate to model the decline of students' knowledge over time, which is based on their previous knowledge state, present learning gains, and the interval time. Extensive experimental results on three public datasets demonstrate that LPKT could obtain more reasonable knowledge state in line with the learning process. Moreover, LPKT also outperforms state-of-the-art KT methods on student performance prediction. Our work indicates a potential future research direction for KT, which is of both high interpretability and accuracy.


## 相关论文：LPKT-S

| 字段 | 内容 |
| --- | --- |
| 题目 | [Monitoring Student Progress for Learning Process-Consistent Knowledge Tracing](https://doi.org/10.1109/TKDE.2022.3221985) |
| 作者 | Shuanghong Shen, Enhong Chen, Qi Liu, Zhenya Huang, Wei Huang, Yu Yin, Yu Su, Shijin Wang |
| 发表年份 | 2023 |
| 发表刊物或会议 | IEEE Transactions on Knowledge and Data Engineering |

### 原文摘要

> Knowledge tracing (KT) is the task of tracing students' evolving knowledge state during learning, which has improved the learning efficiency. To facilitate KT's development, most existing methods pursue high accuracy of student performance prediction but neglect the consistency between students' dynamic knowledge state with their learning process. Moreover, they focus on learning outcomes at a single learning interaction, while student progress at continuous learning interactions is more instructive. In this paper, we explore a new paradigm for the KT task and propose a novel model named Learning Process-consistent Knowledge Tracing (LPKT), which captures the evolution of students' knowledge state through monitoring their learning progress. Specifically, we utilize both the positive effect of the learning gain and the negative effect of forgetting in learning to calculate student progress in continuous learning interactions. Then, considering that the rate of progress is student-specific, we extend LPKT to LPKT-S by explicitly distinguishing the individual progress rate of each student. Extensive experimental results on three public datasets demonstrate that LPKT and LPKT-S could obtain more appropriate knowledge states in line with the learning process. Moreover, LPKT and LPKT-S outperform state-of-the-art KT methods on student performance prediction. Our work indicates a promising future research direction for KT, which is highly interpretable and accurate.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
