# QIKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations](https://doi.org/10.1609/aaai.v37i12.26661) |
| 作者 | Jiahao Chen, Zitao Liu, Shuyan Huang, Qiongqiong Liu, Weiqi Luo |
| 发表年份 | 2023 |
| 发表刊物或会议 | Proceedings of the AAAI Conference on Artificial Intelligence |

## 原文摘要

> Knowledge tracing (KT) is a crucial technique to predict students' future performance by observing their historical learning processes. Due to the powerful representation ability of deep neural networks, remarkable progress has been made by using deep learning techniques to solve the KT problem. The majority of existing approaches rely on the \emph{homogeneous question} assumption that questions have equivalent contributions if they share the same set of knowledge components. Unfortunately, this assumption is inaccurate in real-world educational scenarios. Furthermore, it is very challenging to interpret the prediction results from the existing deep learning based KT models. Therefore, in this paper, we present QIKT, a question-centric interpretable KT model to address the above challenges. The proposed QIKT approach explicitly models students' knowledge state variations at a fine-grained level with question-sensitive cognitive representations that are jointly learned from a question-centric knowledge acquisition module and a question-centric problem solving module. Meanwhile, the QIKT utilizes an item response theory based prediction layer to generate interpretable prediction results. The proposed QIKT model is evaluated on three public real-world educational datasets. The results demonstrate that our approach is superior on the KT prediction task, and it outperforms a wide range of deep learning based KT models in terms of prediction accuracy with better model interpretability. To encourage reproducible results, we have provided all the datasets and code at \url{https://pykt.org/}.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
