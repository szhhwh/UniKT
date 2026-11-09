# MTKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Learning multi-granularity temporal characteristics for attention based knowledge tracing](https://doi.org/10.1016/j.neucom.2025.131338) |
| 作者 | Youheng Bai, Xueyi Li, Zitao Liu, Yaying Huang, Teng Guo, Mingliang Hou, Weiqi Luo |
| 发表年份 | 2025 |
| 发表刊物或会议 | Neurocomputing |

## 原文摘要

> Knowledge tracing (KT) plays a crucial role in personalized learning by predicting students' future performance based on their historical interaction data. Despite advancements in KT models, they often struggle to effectively model the intricate temporal dynamics inherent in student learning processes, particularly in capturing multi-granular temporal information, accommodating irregular time intervals, and handling varying sequence lengths within student interaction data. To address these challenges, we propose MTKT to model multi-scale time patterns and long-term behaviors in student interaction data. Specifically, MTKT introduces three key components. First, a multi-aspect embedding layer captures both temporal information and interaction behaviors from student learning data. Second, a dual attention module with linear decaying bias applies negative bias to attention scores, effectively modeling long-term forgetting in students' knowledge states across varying sequence lengths. Finally, a causal interaction convolution module captures fine-grained representations in both short-term and long-term interactions. We evaluate MTKT on three public educational datasets and compare it with state-of-the-art KT models. Our experiments show that MTKT consistently outperforms existing models in student performance prediction and length-adaptive prediction. To facilitate reproducibility and future research, we make our code and processed datasets publicly available at https://pykt.org.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
