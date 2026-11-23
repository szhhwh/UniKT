# 评测协议与指标

为了降低单次随机划分带来的方差与偏差，提供可复现的基准评估结果，UniKT 采用固定留出测试集结合训练池 5 折交叉验证，并汇报 5 折均值与标准差。具体模型成绩可查阅[最新排行榜](live-leaderboard.md)与[论文基准排行榜（KDD 2027）](leaderboard.md)。

## 标准 5 折交叉验证

### 学生级无泄露切分协议

在知识追踪任务中，若直接对全部答题记录随机打乱切分，同一学生不同时段的记录会同时落入训练集与测试集，导致评估虚高。

UniKT 在固定随机种子下按学生随机切分（长序列切分均在划分后执行）：先按学生人数近似留出 **20%** 作为固定测试集（`fold = -1`），再将剩余 **80%** 学生均匀划分为 5 折（`fold = 0..4`）。

:::::{container} unikt-split-diagram

::::{container} unikt-split-bar

:::{container} unikt-seg-train
**训练集 64%**

训练池其余 4 折
:::

:::{container} unikt-seg-val
**验证集 16%**

训练池第 `k` 折
:::

:::{container} unikt-seg-test
**固定测试集 20%**

`fold = -1`
:::

::::

:::::

当指定 `--data.fold k` 运行时，各数据集切分对照如下：

| 折参数 `--data.fold` | 训练集 Train (64%) | 验证集 Val (16%，早停监控) | 测试集 Test (20%，最终评估) |
| :---: | :---: | :---: | :---: |
| **0** | 折 1, 2, 3, 4 | **折 0** | 固定测试集 (`fold = -1`) |
| **1** | 折 0, 2, 3, 4 | **折 1** | 固定测试集 (`fold = -1`) |
| **2** | 折 0, 1, 3, 4 | **折 2** | 固定测试集 (`fold = -1`) |
| **3** | 折 0, 1, 2, 4 | **折 3** | 固定测试集 (`fold = -1`) |
| **4** | 折 0, 1, 2, 3 | **折 4** | 固定测试集 (`fold = -1`) |

- **Test 集（20%）**：`fold = -1` 固定学生序列，5 折全程一致，不参与训练与早停。
- **Val 集（16%）**：第 `k` 折学生序列，用于早停监控（`patience = 10`）与最佳权重保存。
- **Train 集（64%）**：训练池其余 4 折学生序列，用于参数学习。

### 使用辅助脚本 `run_kfold.sh`

UniKT 在 `scripts/run_kfold.sh` 中提供了多折连续执行脚本，支持训练参数透传与多折自动化串联。

#### A. 查看所有支持的模型列表

使用 `-l` 或 `--list` 选项列出当前代码库中所有已支持的模型：

```bash
bash scripts/run_kfold.sh -l
```

#### B. 运行单模型的 5 折交叉验证

向脚本传入模型名称、折索引列表及待透传的训练参数：

```bash
bash scripts/run_kfold.sh GIKT "0 1 2 3 4" -d assistments09 --model.epochs 150 --model.batch_size 128
```

支持空格分隔（`"0 1 2 3 4"`）或逗号分隔（`"0,1,2,3,4"`）。脚本将依次启动第 0 至第 4 折的训练任务。若某折执行失败（Exit code $\neq$ 0），脚本将立即中断并保留错误现场，避免无意义的算力消耗。

#### C. 运行特定子集折

若某次训练在某几折中断，或只需在特定折验证：

```bash
# 仅执行第 0 折和第 2 折
bash scripts/run_kfold.sh GIKT "0 2" -d assistments09 --model.epochs 100
```

#### D. 原生命令行手动执行

如果不使用 bash 脚本，也可以直接在终端通过原生循环调用 `train.py`：

```bash
for fold in 0 1 2 3 4; do
  python train.py -m GIKT -d assistments09 --data.fold $fold --model.epochs 150 --model.batch_size 128
done
```

### 5 折结果汇总与统计指标

完成 5 折训练后，`runs/normal/` 目录下将产生 5 个独立的实验子目录：

```text
runs/normal/
├── GIKT_assistments09_20261109-100000_fold0_bs128/
├── GIKT_assistments09_20261109-102500_fold1_bs128/
├── GIKT_assistments09_20261109-105000_fold2_bs128/
├── GIKT_assistments09_20261109-111500_fold3_bs128/
└── GIKT_assistments09_20261109-114000_fold4_bs128/
```

每个运行目录下的 `metrics_test.csv` 均记录了该折在测试集上的最终评测指标。可以使用如下代码按折去重取最新结果并计算 5 折均值与标准差：

```python
import glob
import pandas as pd

# 0..4 每折选取最新的测试指标文件，避免重跑同一折时重复计入
metrics_files = []
for fold in range(5):
    candidates = sorted(
        glob.glob(f"runs/normal/GIKT_assistments09_*_fold{fold}_bs128/metrics_test.csv")
    )
    if not candidates:
        raise FileNotFoundError(f"Missing metrics_test.csv for fold {fold}")
    metrics_files.append(candidates[-1])

dfs = [pd.read_csv(f) for f in metrics_files]
summary = pd.concat(dfs, ignore_index=True)
assert len(summary) == 5, f"Expected 5 folds, got {len(summary)}"

print("=== 5 折交叉验证测试集统计 ===")
print(f"有效折数: {len(summary)}")
for metric in ["auc", "acc", "rmse"]:
    if metric in summary.columns:
        mean_val = summary[metric].mean()
        std_val = summary[metric].std()
        print(f"{metric.upper():<6}: {mean_val:.4f} ± {std_val:.4f}")
```

输出示例：
```text
=== 5 折交叉验证测试集统计 ===
有效折数: 5
AUC   : 0.7685 ± 0.0032
ACC   : 0.7321 ± 0.0041
RMSE  : 0.4175 ± 0.0028
```
