# Web 交互实验平台

本指南介绍如何使用 UniKT 内置的 Web 实验管理平台。该平台采用前后端分离架构（后端基于 FastAPI + Uvicorn，前端基于 Vue 3 + Vite + TypeScript），提供涵盖训练任务提交、实时终端日志流式推送、超参数搜索可视化、数据集预处理控制台以及硬件资源监控的完整图形化交互体验。

## 目标与适用场景

- 无需手写繁琐的命令行参数，通过可视化表单快速配置并提交模型训练与评测任务。
- 实时通过 WebSocket 监听任务终端输出，支持随时终止或重试运行中的任务。
- 在线创建 Optuna 超参数搜索任务，直观查看收敛曲线与试验参数对比。
- 图形化管理 15 个数据集处理器（涵盖 14 个公开学术数据集）的一键下载与分步预处理。
- 实时监控服务器 GPU 显存利用率、计算核心负载、CPU 与内存动态。

## 前置条件

1. **环境准备**：使用 Pixi 切换至支持 Web 功能的环境或通过 `-e web` 运行命令：
   ```bash
   pixi run -e web <task>
   ```
2. **首次安装依赖**：前端依赖已由 Pixi 与 npm 共同托管。初次使用时确保环境已解析就绪。

## 启动服务

UniKT 提供了多种启动命令以适应不同的开发与生产部署需求：

### 1. 开发模式

启动全栈热重载开发服务器（同时运行 FastAPI 后端与 Vite 前端）：

```bash
pixi run web-dev
```

- **前端交互界面**：访问 [http://localhost:5173](http://localhost:5173)
- **后端 OpenAPI 接口文档**：访问 [http://localhost:8765/docs](http://localhost:8765/docs)

```{tip}
若需更改前端端口，可以在 `.env` 中设置 `KT_WEB_PORT=5173`。
```

### 2. 独立分步启动

若需对前后端进行独立调试，可以分别启动：

```bash
# 启动后端 API（端口 8765）
pixi run web-backend

# 新终端中启动前端 Vite 开发服务器（端口 5173）
pixi run web-frontend
```

### 3. 常驻服务模式

在团队共享服务器或局域网工作站长期提供访问时，先构建前端静态资源，并通过静态资源服务器启动预览服务，后端保持在本地回路安全运行：

```bash
pixi run web-serve
```

此命令将自动执行 `npm run build`，在后台启动后端 API，并通过 Vite Preview 绑定 `0.0.0.0`，允许局域网内其他机器通过服务器 IP 访问 Web 平台。

## Web 平台核心功能说明

Web 平台划分为四大核心功能模块：

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`play;1.2em;sd-text-primary` 任务调度与实时监控
- 表单根据 `ModelConfig` 动态反射生成参数
- 基于 WebSocket 实时查看终端日志流
- 支持在线一键终止、重试与多任务对比
:::

:::{grid-item-card} {octicon}`sliders;1.2em;sd-text-primary` 超参数搜索管理
- 可视化配置 Optuna 搜索空间与采样算法
- Trial 优化目标收敛曲线实时呈现
- 超参数重要度分析与一键导出最佳配置
:::

:::{grid-item-card} {octicon}`database;1.2em;sd-text-primary` 数据预处理面板
- 12+ 公开基准数据集一键下载与解压
- 自动数据清洗、特征提取与零泄露 K 折划分
- 步骤级进度追踪与元数据实时预览
:::

:::{grid-item-card} {octicon}`meter;1.2em;sd-text-primary` 硬件与资源监控
- 显卡 GPU 利用率与显存水位实时折线图
- CPU 核心负载与系统内存变化趋势
- 异常任务排查与服务器健康状态看板
:::
::::

### 1. 训练任务提交与实时监控

- **任务提交**：
  - 下拉框选择目标模型，系统动态同步所有支持的模型列表。
  - 选择预处理好的数据集与 K 折折数。
  - 模型专属超参数表单会根据该模型的 `ModelConfig` 动态渲染，附带默认值与取值范围校验。
  - 支持配置早停监控指标、随机种子、确定性算法及 GPU 设备编号。
- **任务列表与状态看板**：
  - 集中展示所有处于排队、运行、完成或失败状态的任务看板。
- **实时日志流**：
  - 点击任意任务进入详情页，通过 WebSocket 实时接收终端输出，无需登录服务器。
  - 支持对运行中的任务执行一键终止，或对异常中断的任务重新运行。

### 2. 超参数搜索管理

- **创建搜索试验**：
  - 选择模型与数据集后，系统自动列出该模型 `metadata={'optuna': ...}` 声明的所有可搜索参数及其上下界。
  - 配置试验总数、TPE 或 NSGA-II 采样算法，以及中位数或百分位数剪枝策略。
- **交互式试验可视化**：
  - 动态展示各 Trial 的优化目标收敛折线图。
  - 超参数对目标指标的相关性权重分析。
  - 支持导出最优超参数字典，并直接从界面一键生成重跑任务。

### 3. 数据集预处理面板

- **数据集全局清单**：
  - 查看 UniKT 支持的 15 个标准数据集处理器，涵盖 ASSISTments 系列、EdNet-KT1、Junyi2015、KDD Cup 2010 与 XES3G5M 等。
  - 显示每个数据集的下载状态、处理状态及磁盘占用大小。
- **一键下载与流水线处理**：
  - 点击“下载”按钮从官方源或镜像自动拉取原始数据。
  - 点击“处理”按钮触发自动化清洗、题目与知识点映射，以及 5 折 Parquet 序列生成。
  - 界面实时显示预处理流水线进度条与状态反馈。

### 4. 硬件与系统资源监控

- **实时 GPU 仪表盘**：
  - 检测系统中所有可用的 NVIDIA GPU 设备。
  - 实时绘制各 GPU 的计算核心利用率、显存占用量及当前温度。
- **主机系统资源动态**：
  - 监控主机 CPU 总体占用率与每个核心的负载分布。
  - 系统 RAM 物理内存使用水位曲线。
  - 帮助实验人员及时排查由于 DataLoader `num_workers` 过多导致的内存泄漏或 CPU 瓶颈。

## 预期验证与问题排查

启动 Web 服务后，正常终端将显示类似输出：

```text
INFO:     Started server process [285412]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)

  VITE v8.0.13  ready in 420 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 常见排查提示

1. **端口冲突**：若 `8765` 或 `5173` 被占用，可检查后台残留进程或在 `.env` 中修改 `KT_WEB_PORT`。
2. **WebSocket 断连**：若通过 Nginx 反向代理访问 Web 平台，请确保 Nginx 配置中开启了 `Upgrade` 与 `Connection: Upgrade` 头部转发。
