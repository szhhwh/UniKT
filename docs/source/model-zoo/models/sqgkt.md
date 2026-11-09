# SQGKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [SQGKT: Student-question interaction graph-based knowledge tracing](https://doi.org/10.1016/j.eswa.2025.129175) |
| 作者 | Tao Xu, Yingying Zhao, Juntao Zhang, Jiaming Deng, Zhilong Zhao, Daojun Han, Fengsi Wang, Xiangqian Wei |
| 发表年份 | 2026 |
| 发表刊物或会议 | Expert Systems with Applications |

## 原文摘要

> Knowledge tracing (KT) aims to predict students' performance by assessing their knowledge states and has been widely applied in online education. However, existing KT models have not delved into the implicit information in the process of students' answering, such as learning ability, attempt counts, hint counts, etc., so modeling the dynamic relationships among students, questions, and skills based on students' answer details remains a challenge. In addition, existing models overlook the high-order relationships between students and questions, resulting in the inability to utilize the answering behaviors of students other than the target student. To address these problems, we propose a new graph-based knowledge tracing model named Student-Question interaction Graph-based Knowledge Tracing (SQGKT) to trace students' knowledge states and predict the probability that students answer the question correctly next time. Firstly, we quantify learning ability, attempts, and hints to assess students' learning performance on questions. Then, we construct a student-question graph and a question-skill graph to describe the multiple relationships among students, questions, and skills, respectively. Next, we design a graph embedding and fusion module to perform graph convolution on two graphs and fuse their question embeddings, forming enhanced question representations that integrate students' answering performance and skill information. Finally, we employ LSTM to trace students' knowledge states and predict their performance. We conduct extensive experiments on three public datasets (i.e., ASSISTments 2009, ASSISTments 2012, and Junyi), and the experimental results demonstrate that our SQGKT significantly outperforms all baselines, with improvements of at least 2.40 \% in AUC and at least 4.72 \% in ACC.

## 代码仓库

[Yingying933/SQGKT](https://github.com/Yingying933/SQGKT)
