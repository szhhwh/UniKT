# 评测协议与指标

为了消除单次随机划分的数据集偏差，为学术论文与基准评测提供统计显著的评估结果，知识追踪研究普遍执行 5 折交叉验证并报告各折平均指标。各模型在该协议下的具体评测成绩可查阅[基准排行榜](leaderboard.md)。

## 标准 5 折交叉验证

### 知识追踪中的学生分层切分协议

在知识追踪任务中，若直接对全部答题记录进行随机打乱切分，会导致严重的数据泄露：同一学生不同时段的答题记录被同时割裂在训练集与测试集中，使评估虚高。

UniKT 严格遵守教育数据挖掘的标准评估协议。系统按学生总答题量与正答率进行分层，将所有学生均匀划分为 5 折，划分标记保存在标准化 Parquet 文件的 `fold` 列中。

当指定 `--data.fold k` 运行时，各数据集切分对照如下：

| 折参数 `--data.fold` | 训练集 Train | 验证集 Val（早停监控） | 测试集 Test（最终评估） |
| :---: | :---: | :---: | :---: |
| **0** | 折 1, 2, 3 | 折 4 | **折 0** |
| **1** | 折 2, 3, 4 | 折 0 | **折 1** |
| **2** | 折 3, 4, 0 | 折 1 | **折 2** |
| **3** | 折 4, 0, 1 | 折 2 | **折 3** |
| **4** | 折 0, 1, 2 | 折 3 | **折 4** |

- **Test 集**：第 `k` 折学生序列，完全不参与训练与早停决策。
- **Val 集**：第 `(k - 1) % 5` 折学生序列，用于早停决策与最佳权重保存。
- **Train 集**：剩余 3 折学生序列，用于参数学习。

### 使用辅助脚本 `run_kfold.sh`

UniKT 在 `scripts/run_kfold.sh` 中提供了多折连续执行脚本，支持参数安全透传与多进程自动化串联。

#### A. 查看所有支持的模型列表

使用 `-l` 或 `--list` 选项列出当前代码库中所有已支持的模型：

```bash
bash scripts/run_kfold.sh -l
```

#### B. 运行单模型的 5 折交叉验证

向脚本传入模型名称、折索引列表及待透传的训练参数：

```bash
bash scripts/run_kfold.sh GIKT "0 1 2 3 4" -d assistments09 --model.epochs 150
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
  python train.py -m GIKT -d assistments09 --data.fold $fold --model.epochs 150
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

每个运行目录下的 `metrics_test.csv` 均记录了该折在测试集上的最终评测指标。可以使用如下代码快速计算 5 折均值与标准差：

```python
import glob
import pandas as pd

# 匹配特定实验的所有折测试指标文件
metrics_files = sorted(
    glob.glob("runs/normal/GIKT_assistments09_*_fold*_bs128/metrics_test.csv")
)

dfs = [pd.read_csv(f) for f in metrics_files]
summary = pd.concat(dfs, ignore_index=True)

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
