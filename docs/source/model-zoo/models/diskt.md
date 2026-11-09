# DisKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Disentangled Knowledge Tracing for Alleviating Cognitive Bias](https://doi.org/10.1145/3696410.3714607) |
| 作者 | Yiyun Zhou, Zheqi Lv, Shengyu Zhang, Jingyuan Chen |
| 发表年份 | 2025 |
| 发表刊物或会议 | Proceedings of the ACM Web Conference 2025（WWW '25） |

## 原文摘要

> In the realm of Intelligent Tutoring System (ITS), the accurate assessment of students' knowledge states through Knowledge Tracing (KT) is crucial for personalized learning. However, due to data bias, i.e., the unbalanced distribution of question groups ( e.g., concepts), conventional KT models are plagued by cognitive bias, which tends to result in cognitive underload for overperformers and cognitive overload for underperformers. More seriously, this bias is amplified with the exercise recommendations by ITS. After delving into the causal relations in the KT models, we identify the main cause as the confounder effect of students' historical correct rate distribution over question groups on the student representation and prediction score. Towards this end, we propose a Disentangled Knowledge Tracing (DisKT) model, which separately models students' familiar and unfamiliar abilities based on causal effects and eliminates the impact of the confounder in student representation within the model. Additionally, to shield the contradictory psychology ( e.g., guessing and mistaking) in the students' biased data, DisKT introduces a contradiction attention mechanism. Furthermore, DisKT enhances the interpretability of the model predictions by integrating a variant of Item Response Theory. Experimental results on 11 benchmarks and 3 synthesized datasets with different bias strengths demonstrate that DisKT significantly alleviates cognitive bias and outperforms 16 baselines in evaluation accuracy.


## 代码仓库

[zyy-2001/DisKT](https://github.com/zyy-2001/DisKT)
