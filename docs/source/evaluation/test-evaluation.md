# 独立测试集评估

在模型训练完成后，框架会在测试集上自动执行最终评测。但在以下研究场景中，独立评估工具 `evaluate.py` 非常有用：
- 训练时使用了 `--general.skip_test true` 跳过了测试集评测。
- 需要切换计算设备（例如在无 GPU 的机器上使用 `--general.device cpu` 验证）。
- 需测试 `last_checkpoint.pth` 与 `best_model.pth` 的性能差异。

## 运行独立评估

`evaluate.py` 会自动读取训练目录下归档的 `run_config.yaml`，完全恢复原模型的超参数与网络结构：

```bash
python evaluate.py --evaluate.run_dir runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128
```

## 常用评估选项

```bash
# 评估最后轮次的检查点（而非默认的 best_model.pth）
python evaluate.py \
  --evaluate.run_dir runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128 \
  --evaluate.checkpoint last_checkpoint.pth

# 临时覆盖评估设备与批大小
python evaluate.py \
  --evaluate.run_dir runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128 \
  --general.device cpu \
  --model.batch_size 256
```

| 参数项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--evaluate.run_dir` | `str` | 必填 | 训练运行的输出目录（必须包含 `run_config.yaml` 和权重文件） |
| `--evaluate.checkpoint` | `str` | `best_model.pth` | 运行目录下待评估的权重文件名 |
