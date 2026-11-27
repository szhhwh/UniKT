<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/KT--Exp-%E7%9F%A5%E8%AF%86%E8%BF%BD%E8%B8%AA-blue?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=black">
    <img src="https://img.shields.io/badge/KT--Exp-%E7%9F%A5%E8%AF%86%E8%BF%BD%E8%B8%AA-blue?style=for-the-badge&logo=pytorch&logoColor=white" alt="KT-Exp">
  </picture>
</p>

<h3 align="center">模块化知识追踪实验框架</h3>

<p align="center">
  <a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white">
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch&logoColor=white">
  </a>
  <a href="https://pixi.sh/">
    <img alt="Pixi" src="https://img.shields.io/badge/Pixi-Ready-orange?style=flat-square">
  </a>
</p>

---

<p align="center">
  <b>简体中文</b> | <a href="https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/README.md">English</a>
</p>

<p align="center">
  <a href="#-核心特性">核心特性</a> •
  <a href="#-支持的模型">模型</a> •
  <a href="#-安装">安装</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-文档">文档</a> •
</p>

---

## 🔥 核心特性

- **KT 模型**：DKT、AKT、GKT、GIKT、HGIKT、SGKT、SQGKT、DyGKT、SimpleKT、DKVMN、DTransformer、HawkesKT、LBKT、QIKT、ReKT、RobustKT、UKT、ABKT、BDGKT、ClusterKT 等
- **多数据集支持**：ASSISTments（2009/2012/2017）、EdNet-KT1
- **Optuna 集成**：自动化超参数搜索，支持并行优化
- **实验追踪**：SwanLab 集成，实时可视化训练指标
- **模块化设计**：注册表架构，易于扩展新模型

## 🤖 支持的模型

| 模型简写 | 模型论文链接 | 模型代码仓库链接 |
|------|------|------|
| **DKT** | [Deep Knowledge Tracing (NeurIPS 2015)](https://papers.nips.cc/paper/5654-deep-knowledge-tracing) | [GitHub](https://github.com/chrispiech/DeepKnowledgeTracing) |
| **AKT** | [Context-Aware Attentive Knowledge Tracing (KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403282) | [GitHub](https://github.com/arghosh/AKT) |
| **GKT** | [Graph-based Knowledge Tracing (WI 2019)](https://dl.acm.org/doi/10.1145/3350546.3352513) | [GitHub](https://github.com/jhljx/GKT) |
| **GIKT** | [A Graph-based Interaction Model for KT (ECML-PKDD 2020)](https://arxiv.org/abs/2009.05991) | [GitHub](https://github.com/ApexEDM/GIKT) |
| **SGKT** | [Session Graph-based Knowledge Tracing (ESA 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0957417422009770) | [GitHub](https://github.com/CCNUZFW/SGKT) |
| **SQGKT** | [Student-Question Interaction Graph-based KT (ESA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425027915) | [GitHub](https://github.com/Yingying933/SQGKT) |
| **DyGKT** | [Dynamic Graph Learning for Knowledge Tracing (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671773) | [GitHub](https://github.com/PengLinzhi/DyGKT) |
| **SimpleKT** | [A Simple But Tough-to-Beat Baseline (ICLR 2023)](https://openreview.net/forum?id=9HiGqC9C-KA) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **DKVMN** | [Dynamic Key-Value Memory Networks for Knowledge Tracing (WWW 2017)](https://dl.acm.org/doi/10.1145/3038912.3052580) | [GitHub](https://github.com/jennyzhang0215/DKVMN) |
| **DTransformer** | [Tracing Knowledge Instead of Patterns: Stable KT with Diagnostic Transformer (WWW 2023)](https://dl.acm.org/doi/10.1145/3543507.3583255) | [GitHub](https://github.com/yxonic/DTransformer) |
| **HawkesKT** | [Temporal Cross-Effects in Knowledge Tracing (WSDM 2021)](https://dl.acm.org/doi/10.1145/3437963.3441802) | [GitHub](https://github.com/THUwangcy/HawkesKT) |
| **LBKT** | [Learning Behavior-oriented Knowledge Tracing (KDD 2023)](https://dl.acm.org/doi/10.1145/3580305.3599407) | [GitHub](https://github.com/bigdata-ustc/EduKTM) |
| **QIKT** | [Improving Interpretability of Deep Sequential KT with Question-centric Cognitive Representations (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26661) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **ReKT** | [Revisiting Knowledge Tracing: A Simple and Powerful Model (ACM MM 2024)](https://openreview.net/forum?id=GYomxff6HZ) | [GitHub](https://github.com/lilstrawberry/ReKT) |
| **RobustKT** | [Enhancing Knowledge Tracing through Decoupling Cognitive Pattern from Error-Prone Data (WWW 2025)](https://dl.acm.org/doi/10.1145/3696410.3714486) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **UKT** | [Uncertainty-aware Knowledge Tracing (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/35007) | [GitHub](https://github.com/UncertaintyForKnowledgeTracing/UKT) |
| **ABKT** | [Ability Boosted Knowledge Tracing (Information Sciences 2022)](https://www.sciencedirect.com/science/article/pii/S0020025522001876) | [GitHub](https://github.com/ccnu-mathits/ABKT) |
| **BDGKT** | [Bidirectional Dynamic Graph Knowledge Tracing (Knowledge-Based Systems 2026)](https://doi.org/10.1016/j.knosys.2026.115532) | [Github](https://github.com/Oia-10/BDGKT) |
| **ClusterKT** | [Cluster-driven Knowledge Tracing: Joint Learning-Forgetting Effects Modeling via State Dependency (ESA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425022389) | [GitHub](https://github.com/Lzhenghua/ClusterKT) |

## 📦 安装

### 方式一：使用 Pixi（推荐）

```bash
# 克隆仓库
git clone https://github.com/OUC-LuminaTrail/kt-exp-graph.git
cd kt-exp-graph

# 激活 GPU 环境（CUDA 12.8, Python 3.12, PyTorch 2.10）

# 或使用 Mamba 环境（用于基于 Mamba 的模型）
pixi shell -e mamba
```

### 方式二：手动配置 Conda 环境

<details>
<summary>点击展开手动安装指南</summary>

#### GPU 环境（CUDA 12.8）

```bash
# 创建环境
conda create -n ktexp python=3.12
conda activate ktexp

# 安装 PyTorch（CUDA 12.8）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 安装 PyTorch Geometric 依赖
pip install pyg_lib==0.6.0 torch-scatter==2.1.2 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

# 安装核心依赖
conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
pip install swanlab
```

#### CPU 环境

```bash
# 创建环境
conda create -n ktexp python=3.12
conda activate ktexp

# 安装 PyTorch（CPU）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# 安装 PyTorch Geometric 依赖
pip install pyg_lib==0.6.0 torch-scatter==2.1.2 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.10.0+cpu.html

# 安装核心依赖
conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
pip install swanlab
```

</details>

## 🚀 快速开始

### 1. 数据准备

将原始数据放置于 `data/assistments09/raw/`（详见 [ASSISTments 2009](../docs/source/datasets/catalog/assistments09.md)）：

```bash
python data_process.py process -d assistments09
```

### 2. 训练模型

```bash
# 训练 GIKT 模型
python train.py -m GIKT -d assistments09

# K 折交叉验证
python train.py -m GIKT -d assistments09 --fold 0
```

### 3. 超参数搜索

```bash
# Optuna 搜索 50 轮
python optuna_search.py -m GIKT -d assistments09 --n_trials 50
```

## 📊 支持的数据集

| 数据集 | 数据来源 |
|--------|----------|
| ASSISTments 2009 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ASSISTments 2012 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ASSISTments 2015 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| ASSISTments 2017 | [ASSISTmentsData](https://www.etrialstestbed.org/data-sets) |
| EdNet-KT1 | [GitHub](https://github.com/riiid/ednet) |
| Junyi 2015 | [Junyi Academy](https://pslcdatashop.web.cmu.edu/Files?datasetId=1198) |
| Slepemapy | [SLEP](https://www.fi.muni.cz/adaptivelearning/) |

## 📚 文档

| 文档 | 内容 |
|------|------|
| [快速开始](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/quick_start.md) | 环境配置与最小示例 |
| [数据处理](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/data_processing.md) | 下载、清洗、K 折划分 |
| [模型训练](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/training.md) | 训练流程、K 折验证、早停机制 |
| [超参搜索](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/hyperparameter_search.md) | Optuna 配置与可视化 |
| [案例分析](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/case_analysis.md) | 推理、用户筛选、可视化 |
| [架构设计](https://github.com/OUC-LuminaTrail/kt-exp-graph/blob/main/docs/architecture.md) | 框架设计与扩展指南 |

## 📁 项目结构

```
kt-exp-graph/
├── configs/           # 配置文件
│   └── optuna/        # Optuna 搜索空间
├── data/              # 处理后的数据集
├── docs/              # 项目文档
├── model/             # 模型实现
│   ├── GIKT/
│   ├── HGIKT/
│   ├── SGKT/
│   └── layers/        # 共享组件
├── runs/              # 实验输出
├── utils/             # 框架工具
│   ├── training/      # 训练基础设施
│   ├── config/        # 配置管理
│   └── data_process/  # 数据处理工具
├── train.py           # 训练入口
├── data_process.py    # 数据处理入口
└── optuna_search.py   # 超参搜索
```

## 🤝 致谢

本框架参考了以下优秀开源项目：

- **[pyKT](https://github.com/pykt-team/pykt-toolkit)** - AKT、SimpleKT、DKT、GKT 实现参考
