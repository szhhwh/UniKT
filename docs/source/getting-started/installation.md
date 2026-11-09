# 环境安装与配置

UniKT 采用 [Pixi](https://pixi.sh/) 进行依赖和多环境生命周期管理。通过严格的版本锁定（`pixi.lock`），Pixi 确保了跨不同实验机器与计算节点环境的一致性，有效避免了 PyTorch、CUDA 与 PyTorch Geometric（PyG）等底层 C++/CUDA 扩展库的版本漂移。

---

## 前置要求

- **操作系统**：Linux x86_64（推荐 Ubuntu 22.04 / 24.04 等主流发行版，需要 glibc >= 2.38）。
- **硬件支持**：
  - GPU 环境：推荐 NVIDIA GPU（计算能力 >= 7.0），驱动版本需支持 CUDA 12.8（驱动版本 >= 525.60.13）。
  - CPU 环境：支持任意 x86_64 架构机器，适合轻量调试与无 GPU 服务器。
- **包管理工具**：[Pixi](https://pixi.sh/)（>= 0.79）。

---

## 安装 Pixi

若尚未安装 Pixi，可通过官方单行脚本完成安装：

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

安装完成后，根据终端提示重新加载 Shell 配置（如 `source ~/.bashrc` 或重启终端），确保 `pixi` 命令就绪：

```bash
pixi --version
```

---

## 预置环境

UniKT 在 `pixi.toml` 中定义了 6 套隔离环境：

::::{tab-set}

:::{tab-item} GPU
| 配置项 | 详细参数 |
| :--- | :--- |
| **环境标识** | `default` |
| **硬件后端** | CUDA 12.8、NVIDIA GPU |
| **核心依赖** | Python 3.12、PyTorch 2.10.0+cu128、PyG 2.7.0、torch-scatter 2.1.2、pyg_lib 0.6.0 |

**激活命令**

```bash
pixi shell
```
:::

:::{tab-item} CPU
| 配置项 | 详细参数 |
| :--- | :--- |
| **环境标识** | `cpu` |
| **硬件后端** | x86_64 CPU |
| **核心依赖** | Python 3.12、PyTorch 2.10.0+cpu、PyG 2.7.0 CPU、torch-scatter 2.1.2、pyg_lib 0.6.0 |

**激活命令**

```bash
pixi shell -e cpu
```
:::

:::{tab-item} Mamba
| 配置项 | 详细参数 |
| :--- | :--- |
| **环境标识** | `mamba` |
| **硬件后端** | CUDA 12.8、NVIDIA GPU |
| **核心依赖** | mamba-ssm 2.3.1、causal-conv1d 1.6.1、PyTorch 2.10.0+cu128 |

**激活命令**

```bash
pixi shell -e mamba
```
:::

:::{tab-item} xLSTM
| 配置项 | 详细参数 |
| :--- | :--- |
| **环境标识** | `xlstm` |
| **硬件后端** | CUDA 12.8、NVIDIA GPU、CUDA-NVCC |
| **核心依赖** | xlstm 2.0.5、mlstm_kernels 2.0.4、cuda-nvcc 12.8 |

**激活命令**

```bash
pixi shell -e xlstm
```
:::

:::{tab-item} Web
| 配置项 | 详细参数 |
| :--- | :--- |
| **环境标识** | `web` |
| **硬件后端** | x86_64 CPU、Node.js 运行时 |
| **核心依赖** | FastAPI 0.115、Uvicorn 0.39、SQLAlchemy 2.0、Pydantic 2.0、Vite |

**激活命令**

```bash
pixi shell -e web
```
:::

:::{tab-item} Docs
| 配置项 | 详细参数 |
| :--- | :--- |
| **环境标识** | `docs` |
| **硬件后端** | x86_64 CPU |
| **核心依赖** | Sphinx 8.0、Furo 2025、MyST-Parser 4.0、Pandoc 3.8 |

**激活命令**

```bash
pixi shell -e docs
```
:::

::::

退出当前激活的环境执行 `exit`。

### 单次命令执行

直接在目标环境中运行单次任务：

```bash
pixi run -e <env> <command>
```

示例：

```bash
pixi run -e cpu python -m pytest tests -q
```

---

## 环境变量配置

UniKT 使用 `python-dotenv` 自动管理实验追踪、告警通知及运行参数。在项目根目录下复制模板文件并进行配置：

```bash
cp .env.example .env
```

打开 `.env` 文件，根据需求填入相应的配置项：

```ini
# Lark (飞书) 训练完成/异常 Webhook 通知（可选）
LARK_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxxx
LARK_SECRET=

# 实验追踪后端类型: swanlab | wandb (默认: swanlab)
KT_TRACKING_BACKEND=swanlab

# SwanLab 实验追踪配置
# 模式可选: cloud (云端同步) 或 local (本地离线记录)
SWANLAB_MODE=cloud
KT_SWANLAB_WORKSPACE=my-workspace
KT_SWANLAB_PROJECT=UniKT-Experiments

# Weights & Biases (W&B) 配置 (当 KT_TRACKING_BACKEND=wandb 时生效)
KT_WANDB_PROJECT=UniKT
KT_WANDB_ENTITY=my-team

# 全局日志输出级别: DEBUG, INFO, WARNING, ERROR (默认: INFO)
LOG_LEVEL=INFO

# 异步指标日志落盘 (1 = 后台独立线程写入, 0 = 同步阻塞写入)
METRIC_LOGGING_ASYNC=1

# Web 前端预览服务默认端口
KT_WEB_PORT=5173
```

:::{tip}
若使用 SwanLab 云端记录，建议在初次运行前执行 `swanlab login` 绑定个人或团队 API Key。若在离线集群上训练，可设置 `SWANLAB_MODE=local`，指标将仅保存在本地的 `runs/swanlog/` 目录中。
:::

---

## 安装验证

完成环境配置后，可执行以下命令验证环境是否就绪：

### 1. 验证 PyTorch 与 CUDA 状态

```bash
pixi run python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}')"
```

预期输出应显示版本号（如 `2.10.0+cu128`）且 `CUDA Available: True`。

### 2. 运行自动化单元测试

框架自带基于 CPU 的完整单元测试套件：

```bash
pixi run test
```

预期输出样例：

```text
........................................................................ [  6%]
........................................................................ [ 13%]
...
........................................................................ [ 94%]
.............................................................            [100%]
1068 passed, 1 skipped in 40.05s
```

---

## 文档与 Web 开发任务

Pixi 同样封装了项目级常用脚本：

```bash
# 代码风格检查与格式化
pixi run lint       # ruff check
pixi run fmt        # ruff format
pixi run ckf        # lint + fmt + mypy 类型检查

# 文档构建任务
pixi run docs-build # 编译中文官方 HTML 文档
pixi run docs-dev   # 启动本地热重载文档预览服务

# Web 平台任务
pixi run web-backend # 启动 FastAPI 后端 (8765 端口)
pixi run web-frontend# 启动 Vite 前端开发服务器
pixi run web-dev     # 并发启动前后端服务
```
