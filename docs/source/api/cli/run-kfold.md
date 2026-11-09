# run_kfold.sh 脚本参考

`scripts/run_kfold.sh` 是 UniKT 提供的轻量级 Shell 驱动脚本，用于自动化执行单模型或全模型库在指定 K 折上的连续训练与评测。

## 基本语法

```bash
bash scripts/run_kfold.sh <model_name|ALL> <folds_list> [ARGS...]
bash scripts/run_kfold.sh -l | --list
```

### 参数与选项

| 参数 / 选项 | 说明 | 示例 |
| :--- | :--- | :--- |
| `model_name` | 指定单一已注册模型名称，或传入 `ALL` 针对框架中所有已发现的模型执行跑批 | `GIKT`, `AKT`, `ALL` |
| `folds_list` | 待运行的折索引序列，支持空格分隔（加引号）或逗号分隔 | `"0 1 2 3 4"` 或 `"0,1,2,3,4"` |
| `ARGS...` | 透传给底层 `train.py` 的额外参数（例如数据集 `-d`、轮数 `--model.epochs` 等） | `-d assistments09 --model.epochs 100` |
| `-l, --list` | 打印当前框架中所有通过 AST 静态扫描发现的模型列表并退出 | `bash scripts/run_kfold.sh --list` |

---

## 核心机制

1. **AST 零开销静态模型发现**：
   脚本通过调用 Python 内部的静态 AST 扫描程序列出所有模型：
   ```bash
   python -c "import model; from utils.core.registry import TRAINERS; print(' '.join(sorted(TRAINERS.keys())))"
   ```
   该过程无需实际导入庞大的 PyTorch 权重或依赖库，可在任何环境中毫秒级响应。
2. **错误阻断机制**：
   在循环执行各折时，若某一折因异常退出（退出码非 0），脚本将立即中断退出，并向终端返回该折的错误码，防止静默失败或在错误的配置下空跑算力。
3. **解释器环境注入**：
   脚本默认使用当前环境的 `python` 命令。若需使用特定环境管理器（如 Pixi），可通过环境变量 `PYTHON` 进行覆盖：
   ```bash
   PYTHON="pixi run python" bash scripts/run_kfold.sh GIKT "0 1" -d assistments09
   ```

---

## 常用命令示例

### 1. 查阅所有支持的模型
```bash
bash scripts/run_kfold.sh --list
```

### 2. 对单一模型执行标准 5 折交叉验证
```bash
bash scripts/run_kfold.sh GIKT "0 1 2 3 4" \
    -d assistments09 \
    --model.epochs 150 \
    --model.batch_size 128
```

### 3. 执行特定两折的快速验证
```bash
bash scripts/run_kfold.sh AKT "0,1" -d ednet_kt1 --model.learning_rate 0.0005
```

### 4. 批量运行全量模型基准评测
```bash
bash scripts/run_kfold.sh ALL "0 1 2 3 4" -d assistments09
```
此时脚本将依次遍历框架内置的所有知识追踪模型，为每个模型连续完成 5 折训练。
