---
hide-toc: true
---

# UniKT

:::{container} unikt-home-lead
知识追踪，即 Knowledge Tracing，根据学生的历史作答序列估计其知识状态，并预测下一道题能否答对。UniKT 在统一的数据管线、学生级交叉验证划分与指标口径下训练和评测知识追踪模型，使不同模型的结果可以公平对比、完整复现。
:::

:::::{container} unikt-home-actions
[快速上手](getting-started/quick-start.md)

[模型库](model-zoo/index.md)

[GitHub](https://github.com/szhhwh/UniKT)
:::::

:::{container} unikt-home-figure
<svg viewBox="0 0 680 106" role="img" aria-label="作答序列 (q_t, r_t) 更新知识状态 h_t，并预测下一题答对概率 ŷ_{t+1} / Answer sequence (q_t, r_t) updates the knowledge state h_t and predicts the probability of answering the next question correctly ŷ_{t+1}">
  <g class="unikt-fig-label">
    <text x="114" y="18">(q<tspan baseline-shift="sub" font-size="10">t</tspan>, r<tspan baseline-shift="sub" font-size="10">t</tspan>)</text>
    <text x="333" y="18">h<tspan baseline-shift="sub" font-size="10">t</tspan></text>
    <text x="560" y="18">ŷ<tspan baseline-shift="sub" font-size="10">t+1</tspan></text>
  </g>
  <line class="unikt-fig-axis" x1="30" y1="64" x2="198" y2="64"/>
  <circle class="unikt-fig-correct" cx="30" cy="64" r="12"/>
  <circle class="unikt-fig-correct" cx="72" cy="64" r="12"/>
  <circle class="unikt-fig-wrong" cx="114" cy="64" r="12"/>
  <circle class="unikt-fig-correct" cx="156" cy="64" r="12"/>
  <circle class="unikt-fig-wrong" cx="198" cy="64" r="12"/>
  <path class="unikt-fig-arrow" d="M226 64h36m-6-5 6 5-6 5"/>
  <g class="unikt-fig-state">
    <rect x="282" y="32" width="18" height="14" fill-opacity=".85"/>
    <rect x="303" y="32" width="18" height="14" fill-opacity=".9"/>
    <rect x="324" y="32" width="18" height="14" fill-opacity=".55"/>
    <rect x="345" y="32" width="18" height="14" fill-opacity=".7"/>
    <rect x="366" y="32" width="18" height="14" fill-opacity=".45"/>
    <rect x="282" y="49" width="18" height="14" fill-opacity=".2"/>
    <rect x="303" y="49" width="18" height="14" fill-opacity=".35"/>
    <rect x="324" y="49" width="18" height="14" fill-opacity=".6"/>
    <rect x="345" y="49" width="18" height="14" fill-opacity=".75"/>
    <rect x="366" y="49" width="18" height="14" fill-opacity=".8"/>
    <rect x="282" y="66" width="18" height="14" fill-opacity=".5"/>
    <rect x="303" y="66" width="18" height="14" fill-opacity=".45"/>
    <rect x="324" y="66" width="18" height="14" fill-opacity=".25"/>
    <rect x="345" y="66" width="18" height="14" fill-opacity=".3"/>
    <rect x="366" y="66" width="18" height="14" fill-opacity=".15"/>
    <rect x="282" y="83" width="18" height="14" fill-opacity=".1"/>
    <rect x="303" y="83" width="18" height="14" fill-opacity=".15"/>
    <rect x="324" y="83" width="18" height="14" fill-opacity=".3"/>
    <rect x="345" y="83" width="18" height="14" fill-opacity=".5"/>
    <rect x="366" y="83" width="18" height="14" fill-opacity=".65"/>
  </g>
  <path class="unikt-fig-arrow" d="M402 64h36m-6-5 6 5-6 5"/>
  <circle class="unikt-fig-next" cx="470" cy="64" r="12"/>
  <text class="unikt-fig-mark" x="470" y="69">?</text>
  <rect class="unikt-fig-track" x="500" y="60" width="130" height="8" rx="4"/>
  <rect class="unikt-fig-value" x="500" y="60" width="107" height="8" rx="4"/>
  <text class="unikt-fig-number" x="640" y="69">0.82</text>
</svg>
:::

:::{container} unikt-home-stats
- **64** 个模型实现
- **14** 个公开数据集
- **1.5 亿+** 条学习交互
- **120 万+** 名学生
:::

## 文档导航

:::{container} unikt-home-nav
- [快速开始](getting-started/index.md)
  配置 Pixi 环境，完成第一次数据处理与训练。
- [模型库](model-zoo/index.md)
  按模型查阅论文、作者实现与接入说明。
- [数据集](datasets/index.md)
  查阅数据来源、规模、字段与处理流程。
- [评测](evaluation/index.md)
  基准排行榜、交叉验证协议、独立测试、案例分析与效率评测。
- [用户指南](user-guides/index.md)
  训练、超参数搜索、实验追踪与 Web 平台。
- [开发者指南](developer-guides/index.md)
  接入新模型、新数据集、多阶段训练与回调。
- [API 参考](api/index.md)
  命令行参数与 Python 接口。
- [更新日志](changelog.md)
  框架与文档的版本变更。
:::

```{toctree}
:caption: 开始使用
:maxdepth: 1
:hidden:

getting-started/index
```

```{toctree}
:caption: 研究资源
:maxdepth: 1
:hidden:

datasets/index
model-zoo/index
```

```{toctree}
:caption: 评测
:maxdepth: 1
:hidden:

evaluation/index
```

```{toctree}
:caption: 用户指南
:maxdepth: 1
:hidden:

user-guides/index
```

```{toctree}
:caption: 开发与参考
:maxdepth: 1
:hidden:

developer-guides/index
api/index
changelog
```
