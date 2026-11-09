# MCKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Multi-level Contrastive Learning for Knowledge Tracing](https://doi.org/10.1145/3759920) |
| 作者 | Xiaoxuan Shen, Fenghua Yu, Qian Wan, Ruxia Liang, Jianwen Sun |
| 发表年份 | 2025 |
| 发表刊物或会议 | ACM Transactions on Knowledge Discovery from Data |

## 原文摘要

> Knowledge Tracing (KT) is the task of predicting students’ future performance based on their past interactions with educational resources. A key aspect of KT is representation learning, which aims to capture meaningful features from students’ learning behaviors to improve prediction performance. Recently, contrastive learning methods have shown great promise in representation learning. As a result, KT models based on contrastive learning have been introduced to enhance representation learning for KT. However, these models have posed several challenges. Firstly, most of these models adopt the contrastive learning approach used in other fields, which involves data augmentation followed by contrastive learning, yet effectively applying data augmentation in KT remains an open challenge. Secondly, these models typically apply contrastive learning to only one of the fundamental components of KT: questions, interactions, or knowledge states, thereby limiting their overall performance. To address these issues, this article proposes a Multi-level Contrastive learning model for Knowledge Tracing (MCKT). MCKT (The code can be found at https://github.com/lilstrawberry/MCKT .) does not rely on data augmentation strategies; instead, it deeply integrates domain knowledge and performs contrastive learning at three levels: questions, interactions, and knowledge states. Experimental results on four publicly available datasets, compared against a total of 20 state-of-the-art KT models, demonstrate that MCKT consistently outperforms other models. Subsequent experiments further validate the effectiveness of the multi-level contrastive learning approach.
