# DGMKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Leveraging Student Profiles and the Mamba Framework to Enhance Knowledge Tracing](https://doi.org/10.1007/978-3-032-06109-6_12) |
| 作者 | Mingxing Shao, Tiancheng Zhang, Minghe Yu, Zhenghao Liu, Yifang Yin, Hengyu Liu, Ge Yu |
| 发表年份 | 2025 |
| 发表刊物或会议 | ECML PKDD 2025（Lecture Notes in Computer Science, 16019, pp. 201–218） |

## 原文摘要

> Knowledge tracing (KT) predicts students' knowledge mastery based on their interaction history to forecast future performance. Although current KT methods have achieved good results, because of the lacking of students' information, these methods can only make sequence-level inference, assuming that students are independent and homogeneous. Additionally, due to the typically long student sequences in KT tasks, mainstream RNN-based student state modeling methods suffer from long-sequence forgetting, while attention-based models require manually set bias functions. To address these issues, this paper proposes a Dual-Graph Mamba framework for Knowledge Tracing (DGMKT), which models student profiles based on students' interaction sequence through a Dual-Graph Student-Profile Aware Module (DGSPM). Meanwhile, we model student mastery states based on Mamba, avoiding the long-sequence forgetting problem in RNN-based models and the need for bias functions in attention-based models for KT tasks. To the best of our knowledge, this is the first application of the Mamba architecture in KT tasks. We evaluate DGMKT on four datasets and compare it with ten baselines to demonstrate its superiority. Furthermore, we showcase its broad adaptability by integrating DGSPM with various KT models.
