# CIKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Disentangling Response Sequences with Causal Invariance for Knowledge Tracing](https://doi.org/10.1016/j.engappai.2025.113380) |
| 作者 | Shengze Hu, Junjie Hu, Huali Yang, Jing Geng, Xinjia Ou, Zhuoran Xu, Han Wang, Tao Huang |
| 发表年份 | 2026（2025 年在线发表） |
| 发表刊物或会议 | Engineering Applications of Artificial Intelligence, 165, 113380 |

## 原文摘要

> To support personalized learning applications, Knowledge Tracing (KT) predicts students' future performance by analyzing their historical interactions with questions. Despite advancements in neural network designs and integrated educational principles, existing KT methods often overlook data selection biases, leading to spurious correlations between future performance and response sequences. To address this, we propose a novel KT method based on Causal Invariance (CIKT), which robustly identifies causal relationships in response sequences. From a causal perspective, we develop a structural causal model for KT, guiding the design of CIKT's four modules: a response sequence encoder, an attention-based causal identifier, a sequence causal intervener, and a future response predictor. The encoder extracts representations of future questions and response units. The causal identifier uses question-concept graphs and combines knowledge and temporal associations to estimate causal and trivial scores for each future question, effectively disentangling sequences into causal and trivial subsequences. Grounded in causal invariance, the intervener applies intervention operations to remove, replace, and invert elements within trivial subsequences, generating diverse sequences. The predictor then assigns the same performance prediction task to these intervened sequences to discover invariant causal relationships. Additionally, a question difficulty prediction task for trivial subsequences is introduced to prevent prediction shortcuts, reflecting group mastery levels. Extensive experiments on real-world datasets demonstrate CIKT's superiority and good interpretability.
