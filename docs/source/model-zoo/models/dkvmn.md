# DKVMN

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Dynamic Key-Value Memory Networks for Knowledge Tracing](https://doi.org/10.1145/3038912.3052580) |
| 作者 | Jiani Zhang, Xingjian Shi, Irwin King, Dit-Yan Yeung |
| 发表年份 | 2017 |
| 发表刊物或会议 | Proceedings of the 26th International Conference on World Wide Web |
| DOI | [10.1145/3038912.3052580](https://doi.org/10.1145/3038912.3052580) |

## 原文摘要

> Knowledge Tracing (KT) is a task of tracing evolving knowledge state of students with respect to one or more concepts as they engage in a sequence of learning activities. One important purpose of KT is to personalize the practice sequence to help students learn knowledge concepts efficiently. However, existing methods such as Bayesian Knowledge Tracing and Deep Knowledge Tracing either model knowledge state for each predefined concept separately or fail to pinpoint exactly which concepts a student is good at or unfamiliar with. To solve these problems, this work introduces a new model called Dynamic Key-Value Memory Networks (DKVMN) that can exploit the relationships between underlying concepts and directly output a student's mastery level of each concept. Unlike standard memory-augmented neural networks that facilitate a single memory matrix or two static memory matrices, our model has one static matrix called key, which stores the knowledge concepts and the other dynamic matrix called value, which stores and updates the mastery levels of corresponding concepts. Experiments show that our model consistently outperforms the state-of-the-art model in a range of KT datasets. Moreover, the DKVMN model can automatically discover underlying concepts of exercises typically performed by human annotations and depict the changing knowledge state of a student.


## 代码仓库

[jennyzhang0215/DKVMN](https://github.com/jennyzhang0215/DKVMN)
