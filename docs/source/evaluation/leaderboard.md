# 基准排行榜

本排行榜展示 UniKT 框架在主流公开教育数据集上的统一基准评测结果。所有模型均在统一的数据预处理、学生级 5 折交叉验证划分与标准化指标计算口径下完成评估。

:::{tip}
- **筛选与检索**：可通过顶部标签切换不同数据集，在输入框中输入模型名称实时检索。
- **查看 5 折详情**：将鼠标悬停在各模型的 AUC 或 ACC 数值上，可查看该模型全部 5 折的具体测试成绩。
- **评测规范**：详细的数据集划分、早停策略与指标对齐原则请参考[评测协议与指标](protocol.md)。
:::

```{raw} html
<div class="arena-search">
  <svg class="arena-search-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
  <input id="arena-filter" type="text">
</div>
<div id="arena-tabs"></div>
<div id="arena-table"></div>
<div id="arena-count"></div>
```

```{raw} html
<script src="../_static/arena.js"></script>
```

## 评测规范概述

为确保不同知识追踪模型结果的绝对可比性，UniKT 遵循以下统一基准设置：

- **验证协议**：采用严格的学生级分层 5 折交叉验证，杜绝同一学生在时序上的数据泄露，汇报 5 折测试集均值与标准差。
- **早停机制**：监控验证集 AUC 指标，设置容忍轮数 `patience = 10`，训练终止时自动恢复验证集表现最优的检查点。
- **因果对齐**：预测严格遵守因果时序，根据历史作答状态预测下一步作答表现（下一题预测口径）。
- **指标口径**：统一计算全序列各有效预测位置的二分类 AUC 与 ACC（阈值 0.5）。
