# ATKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Enhancing Knowledge Tracing via Adversarial Training](https://doi.org/10.1145/3474085.3475554) |
| 作者 | Xiaopeng Guo, Zhijie Huang, Jie Gao, Mingyu Shang, Maojing Shu, Jun Sun |
| 发表年份 | 2021 |
| 发表刊物或会议 | Proceedings of the 29th ACM International Conference on Multimedia |
| DOI | [10.1145/3474085.3475554](https://doi.org/10.1145/3474085.3475554) |

## 原文摘要

> We study the problem of knowledge tracing (KT) where the goal is to trace the students' knowledge mastery over time so as to make predictions on their future performance. Owing to the good representation capacity of deep neural networks (DNNs), recent advances on KT have increasingly concentrated on exploring DNNs to improve the performance of KT. However, we empirically reveal that the DNNs based KT models may run the risk of overfitting, especially on small datasets, leading to limited generalization. In this paper, by leveraging the current advances in adversarial training (AT), we propose an efficient AT based KT method (ATKT) to enhance KT model's generalization and thus push the limit of KT. Specifically, we first construct adversarial perturbations and add them on the original interaction embeddings as adversarial examples. The original and adversarial examples are further used to jointly train the KT model, forcing it is not only to be robust to the adversarial examples, but also to enhance the generalization over the original ones. To better implement AT, we then present an efficient attentive-LSTM model as KT backbone, where the key is a proposed knowledge hidden state attention module that adaptively aggregates information from previous knowledge hidden states while simultaneously highlighting the importance of current knowledge hidden state to make a more accurate prediction. Extensive experiments on four public benchmark datasets demonstrate that our ATKT achieves new state-of-the-art performance. Code is available at: https://github.com/xiaopengguo/ATKT.


## 代码仓库

[xiaopengguo/ATKT](https://github.com/xiaopengguo/ATKT)
