# DenoiseKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Denoised Attention and Question-Augmented Representations for Knowledge Tracing](https://doi.org/10.24963/ijcai.2025/1069) |
| 作者 | Jiwei Deng, Youheng Bai, Mingliang Hou, Teng Guo, Zitao Liu, Weiqi Luo |
| 发表年份 | 2025 |
| 发表刊物或会议 | Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence（IJCAI 2025） |

## 原文摘要

> Knowledge tracing (KT) is an essential task in online education systems. It aims to predict the future performance of students based on their historical learning interaction data. Despite significant advancements in attention-based KT models, they still face some limitations: inaccurate input representation and excessive student forgetting modeling. These limitations often lead to the attention noise problem: the model assigns non-negligible attention weight to some information that is cognitively irrelevant in nature, thereby generating interference signals. To address this problem, we propose a novel KT model, i.e., DenoiseKT. DenoiseKT effectively models the difficulty of the questions and utilizes graph neural network to capture the complex relationship between questions, thereby refining the representations of input features. Additionally, the denoised attention mechanism introduces a weight factor to reduce the model's attention weight distribution on irrelevant information. We extensively compare DenoiseKT with 22 state-of-the-art KT models on 4 widely-used public datasets. Experimental results show that DenoiseKT can effectively solve the attention noise problem and outperform other models. The source code of DenoiseKT is available at https://pykt.org.


## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)
