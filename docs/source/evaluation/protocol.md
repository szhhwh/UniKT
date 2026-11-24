# 评测协议与指标

为了降低单次随机划分带来的方差与偏差，提供可复现的基准评估结果，UniKT 采用固定留出测试集结合训练池 5 折交叉验证，并汇报 5 折均值与标准差。具体模型成绩可查阅[排行榜](leaderboard.md)。

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

训练池第 k 折
:::

:::{container} unikt-seg-test
**固定测试集 20%**

fold = -1
:::

::::

:::::

默认划分为 `--data.fold 0`：第 0 折学生序列作验证集（16%，早停监控），训练池其余 4 折作训练集（64%），另留出 20% 学生作固定测试集（`fold = -1`）用于最终评估。

5 折基准评估依次指定 `--data.fold 0` 至 `--data.fold 4`。验证折随参数切换，训练集自动取其余 4 折，固定测试集在所有折上保持一致。

### 学生级子采样与数据集设置

- 11 个数据集在统一预处理后的全量保留学生上评测。
- `EdNet-KT1`（5,000 名）、`Junyi2015`（5,000 名）、`XES3G5M`（2,500 名）在切分前执行学生级随机子采样，复现时在 `data_process.py process` 阶段指定 `--data.sample_size`。
- `ASSISTments 2017` 提供两种预处理设置：题目级重构 `ASSISTments17-PQ` 与原始步级记录 `ASSISTments17`。
- `ABKT` 需模型专用协议，榜单单独标注，不纳入共享协议直接排名。

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

支持空格分隔（`"0 1 2 3 4"`）或逗号分隔（`"0,1,2,3,4"`）。脚本将依次启动第 0 至第 4 折的训练任务。

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

