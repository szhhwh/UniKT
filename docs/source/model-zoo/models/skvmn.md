# SKVMN

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Knowledge Tracing with Sequential Key-Value Memory Networks](https://doi.org/10.1145/3331184.3331195) |
| 作者 | Ghodai Abdelrahman, Qing Wang |
| 发表年份 | 2019 |
| 发表刊物或会议 | Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval |

## 原文摘要

> Can machines trace human knowledge like humans? Knowledge tracing (KT) is a fundamental task in a wide range of applications in education, such as massive open online courses (MOOCs), intelligent tutoring systems, educational games, and learning management systems. It models dynamics in a student's knowledge states in relation to different learning concepts through their interactions with learning activities. Recently, several attempts have been made to use deep learning models for tackling the KT problem. Although these deep learning models have shown promising results, they have limitations: either lack the ability to go deeper to trace how specific concepts in a knowledge state are mastered by a student, or fail to capture long-term dependencies in an exercise sequence. In this paper, we address these limitations by proposing a novel deep learning model for knowledge tracing, namely Sequential Key-Value Memory Networks (SKVMN). This model unifies the strengths of recurrent modelling capacity and memory capacity of the existing deep learning KT models for modelling student learning. We have extensively evaluated our proposed model on five benchmark datasets. The experimental results show that (1) SKVMN outperforms the state-of-the-art KT models on all datasets, (2) SKVMN can better discover the correlation between latent concepts and questions, and (3) SKVMN can trace the knowledge state of students dynamics, and a leverage sequential dependencies in an exercise sequence for improved predication accuracy.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
