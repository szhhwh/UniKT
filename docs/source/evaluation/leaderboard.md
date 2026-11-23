# 论文基准排行榜（KDD 2027）

本排行榜固定呈现 UniKT 论文（KDD 2027）在 14 个公开数据集（15 个评测设置）上 43 个模型的 5 折交叉验证基准结果。如需查看后续新增模型与持续更新成绩，请切换至[最新排行榜](live-leaderboard.md)。

:::{tip}
- **筛选与检索**：可通过顶部标签切换不同数据集，在输入框中输入模型名称实时检索。
- **评测规范**：详细的数据集划分、早停策略与指标对齐原则请参考[评测协议与指标](protocol.md)。
:::

```{raw} html
<div class="arena-search">
  <svg class="arena-search-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
  <input id="arena-filter" type="text">
</div>
<div id="arena-tabs"></div>
<div id="arena-table" data-board="paper"></div>
<div id="arena-count"></div>
```

```{raw} html
<script src="../_static/arena.js"></script>
```

## 评测规范概述

为保证跨模型与跨交互粒度的统一对比，基准实验遵循以下评测设置：

- **数据划分**：按学生级无泄露切分固定测试集（20%）与训练池（80%），在训练池上执行 5 折交叉验证，汇报 5 折均值与标准差。
- **早停策略**：监控验证集 AUC，容忍轮数设为 `patience = 10`，测试阶段自动加载验证集 AUC 最优的模型检查点进行评估。
- **预测目标**：给定历史交互序列预测下一步题目作答正确概率（下一题因果预测）。
- **交互粒度与评估协议**：榜单测试指标均在题目级（Q-level）预测上计算二分类 AUC 与 ACC（阈值 0.5）；知识组件级（KC-level）模型在测试阶段采用 all-in-one 协议将单题关联知识组件（KC）的预测概率取算术均值后再评估，以消除题内标签泄露。
- **数据集设置与学生级子采样**：11 个数据集在统一预处理后的全量保留学生上评测；`EdNet-KT1`（5,000 名学生）、`Junyi15`（5,000 名学生）与 `XES3G5M`（2,500 名学生）在切分前执行学生级随机子采样（复现时在 `data_process.py process` 预处理阶段指定 `--data.sample_size`）；`ASSISTments 2017` 同时提供题目级重构 `A17PQ` 与原始步级记录 `A17` 两种预处理设置；`ABKT` 因需模型专用协议单独标注，不纳入共享协议直接排名。
