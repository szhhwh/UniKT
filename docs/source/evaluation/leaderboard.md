# 排行榜

本榜单呈现 UniKT 在 14 个公开数据集、15 个评测设置上的 5 折交叉验证结果：先按学生留出 20% 固定测试集，再在剩余 80% 上执行 5 折交叉验证，指标取 5 折均值与标准差。

数据划分、早停策略与指标口径见[评测协议与指标](protocol.md)；各数据集的来源、学生规模与许可证见[数据集](../datasets/index.md)。

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
