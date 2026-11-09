# HDHKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [HDHKT: Heterogeneous graph and difficulty-aware hypergraph fusion for knowledge tracing](https://doi.org/10.1016/j.patcog.2026.115226) |
| 作者 | Hao Wang, Xianglian Peng, Weigang Lu, Jing Qin, Shugang Zhang |
| 发表年份 | 2027（2026 年在线发表） |
| 发表刊物或会议 | Pattern Recognition，卷 183，文章编号 115226 |
| DOI | [10.1016/j.patcog.2026.115226](https://doi.org/10.1016/j.patcog.2026.115226) |

## 原文摘要

> Accurate knowledge tracing (KT) for predicting students’ future performance is critical for personalized learning and intelligent educational systems. Recent approaches extend sequential modeling with graph structures to capture structural dependencies in learning processes, yielding promising results. However, existing methods still suffer from insufficient modeling of heterogeneous educational entities and relations as well as limited capacity for higher-order structure modeling. To address these challenges, we propose Heterogeneous graph and Difficulty-aware Hypergraph fusion for Knowledge Tracing (HDHKT). Specifically, we construct a heterogeneous graph to model diverse relations among questions, concepts, and auxiliary entities, including templates and assignments. To capture higher-order structures, we introduce a difficulty-aware weighted hypergraph over questions. Furthermore, we design a Mixture-of-Experts (MoE)-based dynamic fusion mechanism to adaptively integrate representations from different structural views. Experiments on ASSISTments 2009, ASSISTments 2012, and EdNet show that HDHKT achieves state-of-the-art performance, with AUC improvements of up to 2.75% and ACC gains of up to 1.76% over strong baselines, while maintaining competitive performance across datasets. Ablation studies examine the contribution of each component, and qualitative analyses further demonstrate its interpretability and effectiveness in tracking knowledge state dynamics. The code for HDHKT is available at https://github.com/szhhwh/HDHKT.

## 代码仓库

[szhhwh/HDHKT](https://github.com/szhhwh/HDHKT)
