# CDFKT

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 题目 | [Reducing causal attribution ambiguity in knowledge tracing via cognitive decoupling and adaptive fusion](https://doi.org/10.1016/j.inffus.2026.104648) |
| 作者 | Yanqiang Wu, Youheng Bai, Zhiwei Huang, Yibo Lv, Mingliang Hou, Zitao Liu |
| 发表年份 | 2027 |
| 发表刊物或会议 | Information Fusion, 137, 104648（卷期日期：2027 年 1 月） |
| DOI | [10.1016/j.inffus.2026.104648](https://doi.org/10.1016/j.inffus.2026.104648) |

## 摘要概述

CDFKT（Cognitive Decouple-then-Fuse Knowledge Tracing）将学生状态拆分为知识积累与情境适应性，缓解知识掌握与短期表现波动混合建模带来的归因歧义。模型以 Mamba 编码知识积累，用按离散交互步数衰减的时序编码器刻画情境适应性，再通过逐特征自适应门融合两个分支，预测下一道题的作答结果。

## 代码仓库

[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit)，源实现位于 [`pykt/models/cdfkt.py`](https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/cdfkt.py)。

## 在 UniKT 中运行

CDFKT 使用题目级作答序列。题目嵌入由所属知识点的平均嵌入初始化，知识点映射由数据集的 `question_skill` 关系构造。模型依赖 `mamba-ssm` 和 `causal-conv1d`，使用包含这些依赖的 `mamba` 环境在 CUDA 上运行：

```bash
pixi run -e mamba python train.py -m CDFKT -d nips2020_t34 --data.fold 0
```

默认超参数沿用源训练入口的 `nips_task34` fold 2 调参配置，训练、验证和测试划分遵循 UniKT 的题目级序列协议。完整参数可通过以下命令查看：

```bash
pixi run -e mamba python train.py -m CDFKT --help
```
