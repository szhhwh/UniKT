# MCSKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [An efficient knowledge tracing model via Mamba Contextual Encoding and Dynamic Sparse Attention mechanism](https://doi.org/10.1016/j.engappai.2026.114312) |
| 作者 | RuiJuan Zhang, Feng Zhang, Cong Liu |
| 发表年份 | 2026 |
| 发表刊物或会议 | Engineering Applications of Artificial Intelligence |

## 原文摘要

> Knowledge Tracing (KT) predicts learners' future performance by analyzing their historical learning records. While deep learning-based knowledge tracing models have significantly improved prediction performance, they suffer from substantial computational overhead and inefficiency when handling long interaction sequences. To address this problem, we propose an efficient knowledge tracing model named Mamba Contextual Encoding and Dynamic Sparse Attention Mechanism-based Knowledge Tracing (MCSKT). Firstly, leveraging the selective state space structure and linear-time complexity of Mamba, we design a dual-encoder composed of a question encoder and a knowledge encoder, which structurally disentangles the contextual dependencies at the question level and concept level. This design enhances semantic modeling capability while maintaining computational efficiency. Secondly, we propose a dynamic k-sparse attention mechanism to overcome the adaptability constraints inherent in traditional sparse attention methods that rely on manually configured static thresholds. This novel mechanism dynamically adjusts the filtering range of historical interactions, adaptively balancing noise suppression and critical information retention, while significantly reducing computational complexity. Experimental results demonstrate that MCSKT achieves an average improvement of 3.7\% in Area Under the Curve (AUC) and 2.9\% in Accuracy (ACC) across four public datasets. Moreover, compared with the state-of-the-art model, MCSKT achieves substantial acceleration, running approximately 10.1 times faster during training and 3.5 times faster during inference. In addition, the growth rate of time consumption for MCSKT is markedly slower than that of competing models as sequence length increases, highlighting its advantage in processing long-sequence data.
