# SKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Structure-Based Knowledge Tracing: An Influence Propagation View](https://doi.org/10.1109/icdm50108.2020.00063) |
| 作者 | Shiwei Tong, Qi Liu, Wei Huang, Zhenya Hunag, Enhong Chen, Chuanren Liu, Haiping Ma, Shijin Wang |
| 发表年份 | 2020 |
| 发表刊物或会议 | 2020 IEEE International Conference on Data Mining (ICDM) |

## 原文摘要

> Knowledge Tracing (KT) is a fundamental but challenging task in online education that traces learners' evolving knowledge states. Much attention has been drawn to this area and several works such as Bayesian and Deep Knowledge Tracing have been proposed. Recent works have explored the value of relations among concepts and proposed to introduce knowledge structure into KT tasks. However, the propagated influence among concepts, which has been shown to be a key factor in human learning by the educational theories, is still under-explored. In this paper, we propose a new framework called Structure-based Knowledge Tracing (SKT), which exploits the multiple relations in knowledge structure to model the influence propagation among concepts. In the SKT framework, we consider both the temporal effect on the exercising sequence and the spatial effect on the knowledge structure. We take advantages of two novel formulations in modeling the influence propagation on the knowledge structure with multiple relations. For undirected relations such as similarity relations, the synchronization propagation method is adopted, where the influence propagates bidirectionally between neighbor concepts. For directed relations such as prerequisite relations, the partial propagation method is applied, where the influence can only unidirectionally propagate from a predecessor to a successor. Meanwhile, we employ the gated functions to update the states of concepts temporally and spatially. We demonstrate the effectiveness and interpretability of SKT with extensive experiments.


## 代码仓库

[bigdata-ustc/XKT/tree/master/XKT/SKT](https://github.com/bigdata-ustc/XKT/tree/master/XKT/SKT)
