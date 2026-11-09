# efficiency.py 命令行参考

`efficiency.py` 是 UniKT 内置的硬件级模型计算能效评测工具，用于测量模型的推断延迟、训练吞吐率、显存峰值消耗、PyTorch 算子级耗时以及 `torch.compile` 加速收益。

## 运行模式

`efficiency.py` 支持两种互斥的入口模式：

1. **从头构建模式**：通过 `-m` 与 `-d` 实例化模型与数据（可使用随机初始化权重，或指定 `--efficiency.weights` 加载权重）：
   ```bash
   python efficiency.py -m GIKT -d assistments09
   ```
2. **实验归档复现模式**：指定训练实验目录 `--efficiency.run_dir`，系统自动加载归档的 `run_config.yaml` 与最优权重检查点：
   ```bash
   python efficiency.py --efficiency.run_dir runs/normal/GIKT_assistments09_20261109-120000_fold0_bs128
   ```

---

## 参数配置命名空间

### 1. 顶层入口参数 (`--efficiency.*`)

| 参数 | 类型 | 默认值 | 描述 |
| :--- | :--- | :--- | :--- |
| `--efficiency.run_dir` | `str \| null` | `None` | 训练实验输出目录路径，自动复原其配置与权重 |
| `--efficiency.checkpoint` | `str` | `best_model.pth` | 在 `run_dir` 模式下加载的权重文件名 |
| `--efficiency.weights` | `str \| null` | `None` | 独立模型权重检查点文件路径（用于从头构建模式） |

---

### 2. 通用评测参数 (`--efficiency.general.*`)

| 参数 | 类型 | 默认值 | 描述与取值 |
| :--- | :--- | :--- | :--- |
| `--efficiency.general.modes` | `str` | `""` | 指定运行的评测阶段（逗号分隔）。留空表示运行所有已注册阶段：`inference,training,profile,trace,windowlate` |
| `--efficiency.general.batch_sizes` | `str` | `""` | 批量扫描的 Batch Size 列表（如 `32,64,128`）。每个批大小将重置 CUDA 分配器独立评测 |
| `--efficiency.general.compile_modes`| `str` | `""` | 批量扫描的 PyTorch 编译模式列表（如 `off,default,reduce-overhead`）。可与 `batch_sizes` 形成笛卡尔积扫描 |
| `--efficiency.general.warmup_iters` | `int` | `50` | 预热迭代轮数（用于消除 cuDNN autotune 与 GPU 升频延迟波动） |
| `--efficiency.general.resource_sample_interval` | `float` | `0.05` | 后台硬件资源（GPU 显存、利用率）采样时间间隔（秒） |
| `--efficiency.general.output_dir` | `str \| null` | `None` | 报告输出目录（留空时保存至实验日志目录） |

---

### 3. 各阶段专有参数

#### 推断阶段 (`--efficiency.inference.*`)
- `--efficiency.inference.iters` (`int`, 默认 `100`)：推断计时测量的迭代次数。
- `--efficiency.inference.cuda_graph` (`bool`, 默认 `False`)：是否使用 CUDA Graph 录制加速。

#### 训练阶段 (`--efficiency.training.*`)
- `--efficiency.training.iters` (`int`, 默认 `100`)：前向 + 反向 + 优化器单步更新的计时迭代次数。

#### 算子画像与剖析 (`--efficiency.trace.*`)
- 利用 PyTorch Profiler 捕获算子执行栈，统计 GPU/CPU 核函数耗时前 10 的核心算子。

---

## 常用命令示例

### 1. 单阶段快速推断基准测试
```bash
python efficiency.py -m AKT -d assistments09 \
    --efficiency.general.modes inference \
    --efficiency.inference.iters 200
```

### 2. 跨批次与编译模式矩阵扫描
测试模型在不同批大小以及开启/关闭 `torch.compile` 下的吞吐与显存表现：
```bash
python efficiency.py -m SAKT -d assistments09 \
    --efficiency.general.batch_sizes 32,64,128 \
    --efficiency.general.compile_modes off,default,reduce-overhead
```

### 3. 评测已保存的最佳模型检查点
```bash
python efficiency.py \
    --efficiency.run_dir runs/normal/GIKT_assist09_20260520-232131_fold0_bs128
```

---

## 报告产出与终端输出

基准测试结束后，控制台将输出格式化的性能报表，并在实验目录下保存 `efficiency_report.json`：

```text
Efficiency Report  AKT on assistments09  (2026-11-09 12:00:00)
  configured batch_size=128  modes=inference,train,profile
  dataset padded lengths (metadata): max_question_seq_len=200  max_skill_seq_len=200

                      Inference                       
 Iterations               200 x 3                     
 Data split               validation                  
 Batch size               128                         
 Count basis              measured_batch              
 Valid tokens / batch     9,741                       
 Latency mean             3.425 ms                    
 Latency p50 / p95 / p99  3.382 / 3.654 / 3.782 ms    
 Latency cv               0.024                       
 Repeat cv (stability)    0.012                       
 Throughput (sustained)   37,426 interactions/s       
 Per interaction          26,719 ns                   

                        Train                         
 Iterations               50 x 3                      
 Data split               train                       
 Batch size               128                         
 Step time mean           10.152 ms                   
 Step time p50 / p95      9.982 / 10.842 ms           
 Throughput (sustained)   12,610 interactions/s       

                       Profile                        
 Trainable params         1,428,609 (1.43 M)          
 Non-trainable params     0                           
 Theoretical GFLOPs       4.35                        

      Resource Usage — inference      
 CPU%          184.2       210.5      
 Process RSS   1842.0 MiB  1890.5 MiB 
 GPU util      78.5%       92.0%      
 GPU power     185.2 W     210.4 W    
 GPU mem used  1248.5 MiB  1248.5 MiB 
 GPU temp      58.0 °C     62.0 °C    

Report written to: runs/efficiency/AKT_assistments09_20261109-120000_fold0_bs128/efficiency_report.json
```
