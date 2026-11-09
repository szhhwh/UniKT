# FAKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [A Frequency-Aware Mixture of Heterogeneous Experts Framework for Knowledge Tracing](https://doi.org/10.1145/3774904.3792272) |
| 作者 | Youheng Bai, Mingliang Hou, Teng Guo, Zitao Liu, Weiqi Luo |
| 发表年份 | 2026 |
| 发表刊物或会议 | Proceedings of the ACM Web Conference 2026 |
| DOI | [10.1145/3774904.3792272](https://doi.org/10.1145/3774904.3792272) |

## 原文摘要

> Knowledge tracing (KT) aims to personalize online education on large-scale web-based platforms by modeling students' evolving knowledge states from their interaction sequences. However, most KT models rely on a single encoder architecture (e.g., self-attention or RNN), with fixed inductive biases that fails to capture the diversity of learning behaviors. Specifically, student learning unfolds across multiple timescales, and interaction sequences contain diverse frequency components ranging from short-term variations to long-term trends. Our data-driven analysis reveals that existing encoders exhibit characteristic frequency biases (e.g., self-attention tends to emphasize low-frequency patterns), highlighting the limitations of any single architecture. To address this problem, we propose FA-KT, a frequency-aware mixture of heterogeneous experts framework. FA-KT combines self-attention, Mamba, CNN, and LSTM experts, each with complementary frequency biases. A frequency-aware router analyzes each sequence's frequency characteristics and adaptively combines experts to create dynamic, personalized encoders for individual students. Across five benchmark datasets, FA-KT consistently outperforms 20 strong KT baselines in predicting future performance. Code is available at https://pykt.org/.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
