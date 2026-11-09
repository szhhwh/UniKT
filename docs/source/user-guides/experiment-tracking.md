# 实验追踪与配置归档

本指南介绍 UniKT 的实验追踪与元数据无损归档体系，包括如何配置 SwanLab 云端及本地看板、如何使用飞书通知接收训练预警，以及如何利用生成的 `run_config.yaml` 与 `run_metadata.yaml` 实现实验的严格可复现。

## 目标与适用场景

- 实时监控模型训练过程中的训练损失、验证损失以及各项验证性能指标。
- 跟踪硬件资源消耗情况（GPU 显存占用、利用率）。
- 配置离线/局域网私有化部署模式或直接关闭云端上报。
- 保证每一次实验的配置完全自包含、可追溯并可一键复现。

## 前置条件

1. **环境准备**：已进入 Pixi 环境（SwanLab 依赖已内置安装）。
2. **环境变量文件**：在仓库根目录下基于 `.env.example` 复制生成 `.env` 文件：
   ```bash
   cp .env.example .env
   ```

## SwanLab 实验追踪配置

UniKT 默认集成 [SwanLab](https://swanlab.cn/) 作为轻量级、开箱即用的实验追踪看板（同时兼容 Weights & Biases）。

### 1. 云端追踪模式

在 `.env` 文件中填入你的凭据信息：

```bash
# .env 文件配置
SWANLAB_MODE=cloud
SWANLAB_API_KEY=your_swanlab_api_key_here
KT_SWANLAB_WORKSPACE=your_workspace_name
KT_SWANLAB_PROJECT=UniKT

# 可选：飞书/Lark 实验通知机器人 Webhook
LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxxxx
LARK_SECRET=your_lark_secret_if_any
```

首次使用时，也可以在终端运行一键登录：
```bash
swanlab login
```

启动训练后，控制台将输出 SwanLab 实验看板链接：
```text
swanlab: Tracking run: https://swanlab.cn/@my-workspace/UniKT-Experiments/runs/run-20261109_120000
```

### 2. 本地离线模式

在无外网连接的集群服务器或保密计算环境中，可切换为本地模式。训练数据将仅保存在本机的 `swanlog/` 目录下，不向外网发送任何数据：

```bash
# 在 .env 中修改
SWANLAB_MODE=local
```

启动本地看板服务查看历史实验：
```bash
swanlab watch swanlog
```
在浏览器中打开 `http://localhost:5092` 即可浏览本地所有训练曲线。

### 3. 完全禁用云端追踪

如果仅需要本地控制台输出与 CSV 文件记录，可以在命令行中通过 `--general.cloud_tracking false` 完全关闭指标上报器：

```bash
python train.py -m GIKT -d assistments09 --general.cloud_tracking false
```

### 4. 切换为 Weights & Biases

若团队习惯使用 W&B，可通过环境变量轻松切换：

```bash
# 在 .env 中配置
KT_TRACKING_BACKEND=wandb
KT_WANDB_PROJECT=UniKT
KT_WANDB_ENTITY=your_team
```

## 追踪指标命名规范

UniKT 自动记录各阶段指标，并在看板中采用层级分明的前缀规范：

| 命名空间 | 指标名称示例 | 说明 |
| :--- | :--- | :--- |
| **训练阶段** | `Train/loss`<br/>`Train/lr` | 每个 Epoch 结束时的平均训练损失与当前优化器学习率 |
| **验证阶段** | `Val/auc`<br/>`Val/acc`<br/>`Val/rmse`<br/>`Val/loss` | 用于早停决策与超参数评估的验证集指标 |
| **多阶段提升模型** | `STAGE1/Val/auc`<br/>`STAGE2/Val/auc` | 针对 ABKT 等多阶段训练模型的专属分阶段指标 |
| **测试阶段** | `Test/auc`<br/>`Test/acc`<br/>`Test/rmse` | 最终使用最佳模型权重在独立测试集上评测的指标 |
| **早停决策轨迹** | `EarlyStopping/best_score`<br/>`EarlyStopping/wait_epochs` | 当前累计的最优指标值与已连续未提升的 Epoch 计数器 |

```{note}
为了兼顾网络吞吐与训练效率，UniKT 默认采用异步指标上报器（`METRIC_LOGGING_ASYNC=1`），在独立的后台线程中提交网络请求，不阻塞 GPU 反向传播与前向计算。
```

## 实验归档产物

任何一次训练（无论是常规训练、超参数搜索还是多折评估）结束时，都会在指定的 `runs/` 目录下落盘一套自包含的归档文件。

```text
runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128/
├── run_config.yaml         # 完整的静态配置快照
├── run_metadata.yaml       # 动态运行时元数据
├── metrics_train.csv       # 训练集指标历史
├── metrics_val.csv         # 验证集指标历史
├── metrics_test.csv        # 测试集最终指标
├── best_model.pth          # 最优模型权重
└── run.log                 # 完整控制台日志
```

### 1. `run_config.yaml`：无损复现配置

该文件保存了本次运行的所有输入参数（类型安全的 dataclass 树序列化），排除了任何运行时易变对象：

```yaml
general:
  device: cuda:0
  seed: 42
  deterministic: true
  cloud_tracking: true
compile:
  compile: false
early_stopping:
  monitor: auc
  mode: max
  patience: 10
experiment:
  model_name: GIKT
data:
  dataset: assistments09
  fold: 0
  kfold: 5
  max_seq_len: 200
model:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
  hidden_dim: 100
```

你可以随时将此文件直接喂给 `train.py` 来 100% 精确复现实验：
```bash
python train.py --config runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128/run_config.yaml
```

### 2. `run_metadata.yaml`：运行时环境特征

为了防止“环境差异导致不可复现”，UniKT 还会生成一份元数据辅助文件，忠实记录本次训练在硬件和网络层面的真实属性：

```yaml
execution:
  start_time: '2026-11-09T12:00:00'
  duration_seconds: 142.5
environment:
  pytorch_version: 2.10.0
  cuda_available: true
  cuda_device_name: NVIDIA GeForce RTX 4090
model_summary:
  total_parameters: 412850
  trainable_parameters: 412850
  optimizer: AdamW
```

该机制保证了学术研究与工程协作中的可追溯性与严谨性。
