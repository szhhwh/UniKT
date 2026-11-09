# SAKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [A Self-Attentive model for Knowledge Tracing](https://doi.org/10.48550/arxiv.1907.06837) |
| 作者 | Shalini Pandey, George Karypis |
| 发表年份 | 2019 |
| 发表刊物或会议 | arXiv (Cornell University) |

## 原文摘要

> Knowledge tracing is the task of modeling each student's mastery of knowledge concepts (KCs) as (s)he engages with a sequence of learning activities. Each student's knowledge is modeled by estimating the performance of the student on the learning activities. It is an important research area for providing a personalized learning platform to students. In recent years, methods based on Recurrent Neural Networks (RNN) such as Deep Knowledge Tracing (DKT) and Dynamic Key-Value Memory Network (DKVMN) outperformed all the traditional methods because of their ability to capture complex representation of human learning. However, these methods face the issue of not generalizing well while dealing with sparse data which is the case with real-world data as students interact with few KCs. In order to address this issue, we develop an approach that identifies the KCs from the student's past activities that are \textit{relevant} to the given KC and predicts his/her mastery based on the relatively few KCs that it picked. Since predictions are made based on relatively few past activities, it handles the data sparsity problem better than the methods based on RNN. For identifying the relevance between the KCs, we propose a self-attention based approach, Self Attentive Knowledge Tracing (SAKT). Extensive experimentation on a variety of real-world dataset shows that our model outperforms the state-of-the-art models for knowledge tracing, improving AUC by 4.43% on average.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
