# CDFKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Reducing causal attribution ambiguity in knowledge tracing via cognitive decoupling and adaptive fusion](https://doi.org/10.1016/j.inffus.2026.104648) |
| 作者 | Yanqiang Wu, Youheng Bai, Zhiwei Huang, Yibo Lv, Mingliang Hou, Zitao Liu |
| 发表年份 | 2027 |
| 发表刊物或会议 | Information Fusion, 137, 104648 |

## 原文摘要

> Knowledge tracing is a fundamental task in intelligent tutoring systems, aimed at modeling students' learning trajectories. Although deep learning based knowledge tracing methods have achieved remarkable success, they typically encode knowledge mastery and performance fluctuations into a unified feature space. This conflation leads to ambiguity in causal attribution and thereby hinders precise state tracking. To address this issue, inspired by cognitive science, we propose a novel model named Cognitive Decouple-then-Fuse Knowledge Tracing (CDFKT). CDFKT explicitly decouples the student knowledge state into knowledge accumulation and situational adaptability. Specifically, we employ a Mamba-based architecture to capture stable knowledge accumulation and design a novel sequence-step decay temporal encoder to model situational adaptability under discrete interaction steps. Although these two cognitive components are modeled separately, they interact throughout the learning process. To capture this interplay, we further introduce a feature-wise adaptive fusion mechanism to integrate these heterogeneous signals. Extensive experiments on three real-world educational datasets demonstrate that CDFKT achieves state-of-the-art performance. Our code is publicly available at: https://pykt.org.

## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
