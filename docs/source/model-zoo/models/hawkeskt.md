# HawkesKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Temporal Cross-Effects in Knowledge Tracing](https://doi.org/10.1145/3437963.3441802) |
| 作者 | Chenyang Wang, Weizhi Ma, Min Zhang, Chuancheng Lv, Fengyuan Wan, Huijie Lin, Taoran Tang, Yiqun Liu, Shaoping Ma |
| 发表年份 | 2021 |
| 发表刊物或会议 | Proceedings of the 14th ACM International Conference on Web Search and Data Mining |
| DOI | [10.1145/3437963.3441802](https://doi.org/10.1145/3437963.3441802) |

## 原文摘要

> Knowledge tracing (KT) aims to model students' knowledge level based on their historical performance, which plays an important role in computer-assisted education and adaptive learning. Recent studies try to take temporal effects of past interactions into consideration, such as the forgetting behavior. However, existing work mainly relies on time-related features or a global decay function to model the time-sensitive effects. Fine-grained temporal dynamics of different cross-skill impacts have not been well studied (named as temporal cross-effects). For example, cross-effects on some difficult skills may drop quickly, and the effects caused by distinct previous interactions may also have different temporal evolutions, which cannot be captured in a global way. In this work, we investigate fine-grained temporal cross-effects between different skills in KT. We first validate the existence of temporal cross-effects in real-world datasets through empirical studies. Then, a novel model, HawkesKT, is proposed to explicitly model the temporal cross-effects inspired by the point process, where each previous interaction will have different time-sensitive impacts on the mastery of the target skill. HawkesKT adopts two components to model temporal cross-effects: 1) mutual excitation represents the degree of cross-effects and 2) kernel function controls the adaptive temporal evolution. To the best of our knowledge, we are the first to introduce Hawkes process to model temporal cross-effects in KT. Extensive experiments on three benchmark datasets show that HawkesKT is superior to state-of-the-art KT methods. Remarkably, our method also exhibits excellent interpretability and shows significant advantages in training efficiency, which makes it more applicable in real-world large-scale educational settings.


## 代码仓库

[THUwangcy/HawkesKT](https://github.com/THUwangcy/HawkesKT)
