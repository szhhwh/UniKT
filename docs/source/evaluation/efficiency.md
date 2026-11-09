# 效率评测

本指南介绍如何使用 UniKT 的能效评测工具（`efficiency.py`）对知识追踪模型在目标硬件上的推理延迟、吞吐量、训练步耗时、显存峰值占用以及理论计算量进行标准化基准测试，并评估 PyTorch 2.0+ `torch.compile` 的加速表现。

## 目标与适用场景

- 测量模型在实际推理场景下的端到端延迟（Mean、P50、P90、P99 分位数）与单秒吞吐量。
- 统计模型在不同批大小下的 GPU 显存峰值与显存溢出临界点。
- 评估 `torch.compile`（`default`, `reduce-overhead`, `max-autotune`）对知识追踪模型的加速比。
- 支持两种启动模式：**全新初始化快速评测**（无需等待训练）与**真实检查点评测**（完全继承实际训练参数）。

## 前置条件

1. **环境准备**：建议在具有独立 GPU（CUDA）的环境下运行。CPU 环境同样支持，但 CUDA 相关的显存采样与动态图优化将被自适应跳过。
2. **数据就绪**：评测会构建真实的数据批次以确保输入张量维度与实际训练评估完全一致，因此需要目标数据集已完成预处理。

## 评测模式与基本命令

### 模式一：全新初始化快速评测

如果你正在设计一个新模型架构，或者想要在短时间内比较多个模型的硬件资源消耗，可以直接使用 `-m` 和 `-d` 进行快速基准测试（模型采用随机权重初始化，无需提前训练）：

```bash
python efficiency.py -m GIKT -d assistments09
```

若有单独导出的权重文件，可通过 `--efficiency.weights` 注入：

```bash
python efficiency.py -m SAKT -d assistments09 \
  --efficiency.weights runs/normal/SAKT_assistments09_20261109-120000_fold0_bs128/best_model.pth
```

### 模式二：基于已训练检查点评测

若需要严格评测某个已完成实验的实际表现，使用 `--efficiency.run_dir` 参数。系统会自动读取该目录下的 `run_config.yaml`，完全复现训练时的网络层数、嵌入维度及最大序列截断长度：

```bash
python efficiency.py \
  --efficiency.run_dir runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128
```

## 进阶能效扫描

`efficiency.py` 原生支持笛卡尔积交叉扫描，只需一条命令即可完成不同批大小与不同编译模式的系统化评测对比。

### 1. 扫描批大小

对比批大小对显存增长与推理吞吐量的影响。每次批大小测试都会在隔离的子环境中重新初始化训练器，保证 CUDA 显存分配器干净：

```bash
python efficiency.py -m SAKT -d assistments09 \
  --efficiency.general.batch_sizes 32,64,128,256
```

### 2. 评测 PyTorch 编译加速

评测 PyTorch 2.0+ 编译技术带来的性能收益。系统将在同一硬件上对比基线（`off`，未编译）与不同编译模式下的推理耗时：

```bash
python efficiency.py -m SAKT -d assistments09 \
  --efficiency.general.compile_modes off,default,reduce-overhead
```

### 3. 组合交叉扫描

同时扫描批大小与编译模式，评估两者联合优化空间：

```bash
python efficiency.py -m SAKT -d assistments09 \
  --efficiency.general.batch_sizes 32,64 \
  --efficiency.general.compile_modes off,default
```

## 评测阶段与参数配置

UniKT 能效评测系统由多个相互独立的评测阶段构成，默认全部执行。可以通过 `--efficiency.general.modes` 按需挑选：

```bash
# 仅执行推理延迟评测
python efficiency.py -m AKT -d assistments09 \
  --efficiency.general.modes inference \
  --efficiency.inference.iters 500
```

| 评测阶段 | 标识符 | 核心测量内容 | 常用定制参数 |
| :--- | :--- | :--- | :--- |
| **推理延迟与吞吐** | `inference` | 前向推理延迟（P50/P90/P99）、吞吐量、推理显存峰值 | `--efficiency.inference.iters 200`<br/>`--efficiency.inference.repeats 3` |
| **训练步耗时** | `train` | 完整训练步耗时（前向 + Loss + 反向传播 + 优化器更新） | `--efficiency.train.iters 50`<br/>`--efficiency.train.repeats 1` |
| **模型复杂度分析** | `profile` | 参数量、理论浮点计算量 FLOPs、乘加累加数 | `--efficiency.profile.flops true` |
| **算子算力追踪** | `trace` | PyTorch Profiler 算子级分析，导出前 N 个耗时最大的 CUDA/CPU 算子 | `--efficiency.trace.iters 20`<br/>`--efficiency.trace.top_ops 15` |
| **滑动窗口推断** | `windowlate` | 在长序列流式预测下的滑窗更新延迟 | `--efficiency.windowlate.iters 200` |

### 通用控制参数

| 参数项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--efficiency.general.warmup_iters` | `int` | `50` | 正式计时前丢弃的预热轮数（用于消除 cuDNN 动态寻优与时钟爬升偏差） |
| `--efficiency.general.resource_sample_interval` | `float` | `0.05` | 后台资源监控采样间隔（秒） |
| `--efficiency.general.output_dir` | `str` | `null` | 评测报告 JSON 与日志输出路径（默认保存在实验时间戳目录） |

## 报告解读与结果验证

评测完成后，终端将打印富文本对齐的汇总表格，并将结构化评测数据写入 `efficiency_report.json`。

### 终端输出示例

```text
Efficiency Report  GIKT on assistments09  (2026-11-09 12:00:00)
  configured batch_size=128  modes=inference,train,profile
  dataset padded lengths (metadata): max_question_seq_len=200  max_skill_seq_len=200

                      Inference                       
 Iterations               200 x 3                     
 Data split               validation                  
 Batch size               128                         
 Count basis              measured_batch              
 Valid tokens / batch     9,741                       
 Latency mean             3.125 ms                    
 Latency p50 / p95 / p99  3.082 / 3.421 / 3.821 ms    
 Latency cv               0.021                       
 Repeat cv (stability)    0.010                       
 Throughput (sustained)   41,025 interactions/s       
 Per interaction          24,375 ns                   

                        Train                         
 Iterations               50 x 3                      
 Data split               train                       
 Batch size               128                         
 Step time mean           9.452 ms                    
 Step time p50 / p95      9.281 / 10.124 ms           
 Throughput (sustained)   13,542 interactions/s       

                       Profile                        
 Trainable params         412,850 (0.41 M)            
 Non-trainable params     0                           
 Theoretical GFLOPs       1.24                        

      Resource Usage — inference      
 CPU%          168.4       192.1      
 Process RSS   1540.0 MiB  1580.2 MiB 
 GPU util      65.2%       78.4%      
 GPU power     142.0 W     165.5 W    
 GPU mem used  384.5 MiB   384.5 MiB  
 GPU temp      54.0 °C     58.0 °C    

Report exported to: runs/efficiency/GIKT_assistments09_20261109-120000_fold0_bs128/efficiency_report.json
```

### 扫描报告文件结构

在进行多参数扫描时，生成的 `efficiency_report.json` 会将各配置组合以字典形式记录：

```json
{
  "device": "NVIDIA GeForce RTX 4090",
  "pytorch_version": "2.10.0",
  "runs": [
    {
      "batch_size": 32,
      "compile_mode": "off",
      "inference_latency_ms": 1.45,
      "throughput_samples_per_s": 22068,
      "peak_vram_mb": 182.4
    },
    {
      "batch_size": 32,
      "compile_mode": "default",
      "inference_latency_ms": 0.98,
      "throughput_samples_per_s": 32653,
      "peak_vram_mb": 178.1
    }
  ]
}
```
通过对比 `compile_mode: default` 与 `off` 的数据，可以直接获得当前硬件与数据配置下的加速比（如上述示例中的 1.48x 加速）。
