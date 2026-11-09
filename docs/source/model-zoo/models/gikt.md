# GIKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [GIKT: A Graph-Based Interaction Model for Knowledge Tracing](https://doi.org/10.1007/978-3-030-67658-2_18) |
| 作者 | Yang Yang, Jian Shen, Yanru Qu, Yunfei Liu, Kerong Wang, Yaoming Zhu, Weinan Zhang, Yong Yu |
| 发表年份 | 2021 |
| 发表刊物或会议 | Machine Learning and Knowledge Discovery in Databases |
| DOI | [10.1007/978-3-030-67658-2_18](https://doi.org/10.1007/978-3-030-67658-2_18) |

## 原文摘要

> With the rapid development in online education, knowledge tracing (KT) has become a fundamental problem which traces students' knowledge status and predicts their performance on new questions. Questions are often numerous in online education systems, and are always associated with much fewer skills. However, the previous literature fails to involve question information together with high-order question-skill correlations, which is mostly limited by data sparsity and multi-skill problems. From the model perspective, previous models can hardly capture the long-term dependency of student exercise history, and cannot model the interactions between student-questions, and student-skills in a consistent way. In this paper, we propose a Graph-based Interaction model for Knowledge Tracing (GIKT) to tackle the above problems. More specifically, GIKT utilizes graph convolutional network (GCN) to substantially incorporate question-skill correlations via embedding propagation. Besides, considering that relevant questions are usually scattered throughout the exercise history, and that question and skill are just different instantiations of knowledge, GIKT generalizes the degree of students' master of the question to the interactions between the student's current state, the student's history states, the target question, and related skills. Experiments on three datasets demonstrate that GIKT achieves the new state-of-the-art performance, with 2%--6% absolute AUC improvement.


## 代码仓库

[ApexEDM/GIKT](https://github.com/ApexEDM/GIKT)
