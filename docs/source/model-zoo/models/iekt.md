# IEKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Tracing Knowledge State with Individual Cognition and Acquisition Estimation](https://doi.org/10.1145/3404835.3462827) |
| 作者 | Ting Long, Yunfei Liu, Jian Shen, Weinan Zhang, Yong Yu |
| 发表年份 | 2021 |
| 发表刊物或会议 | Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval |

## 原文摘要

> Knowledge tracing, which dynamically estimates students' learning states by predicting their performance on answering questions, is an essential task in online education. One typical solution for knowledge tracing is based on Recurrent Neural Networks (RNNs), which represent students' knowledge states with the hidden states of RNNs. Such type of methods normally assumes that students have the same cognition level and knowledge acquisition sensitivity on the same question. Thus, they (i) predict students' responses by referring to their knowledge states and question representations, and (ii) update the knowledge states according to the question representations and students' responses. No explicit cognition level or knowledge acquisition sensitivity is considered in the above two processes. However, in real-world scenarios, students have different understandings on a question and have various knowledge acquisition after they finish the same question. In this paper, we propose a novel model called Individual Estimation Knowledge Tracing (IEKT), which estimates the students' cognition on the question before response prediction and assesses their knowledge acquisition sensitivity on the questions before updating the knowledge state. In the experiments, we compare IEKT with 11 knowledge tracing baselines on four benchmark datasets, and the results show IEKT achieves the state-of-the-art performance.


## 代码仓库

- [ApexEDM/iekt](https://github.com/ApexEDM/iekt)
- [pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
