# SGKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [SGKT: Session graph-based knowledge tracing for student performance prediction](https://doi.org/10.1016/j.eswa.2022.117681) |
| 作者 | Zhengyang Wu, Li Huang, Qionghao Huang, Changqin Huang, Yong Tang |
| 发表年份 | 2022 |
| 发表刊物或会议 | Expert Systems with Applications |

## 原文摘要

> Knowledge tracing is a modeling method of students' knowledge mastery. The deep knowledge tracing (DKT) model uses long short-term memory (LSTM) to process the sequence data of students exercises. However, the LSTM-based model pays more attention to the short-term response status of students while ignoring the long-term learning process. Moreover, existing graph-based knowledge tracing models focus on the static relationship between exercises and skills, ignoring the dynamic graphs formed by students exercises in a session. In this work, we propose a novel knowledge tracing model which is based on an exercise session graph, named session graph based knowledge tracing (SGKT). The session graph is used to model the students' answering process. In addition, a relationship graph is used to model the relationship between exercises and skills. Then we use gated graph neural networks to obtain the students' knowledge state from the session graph and use graph convolutional networks to obtain the embedding representations of exercises and skills in the relationship graph. Next, through the interaction mechanism, multiple interaction states composed of knowledge states and embedding representations are obtained. The attention mechanism is used to find the focus from these states and make predictions. Experiments are conducted on three publicly available datasets and the results show that our approach has advantages over some existing baseline methods.


## 代码仓库

[CCNUZFW/SGKT](https://github.com/CCNUZFW/SGKT)
