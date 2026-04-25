# 第 3 章 轴承与齿轮箱特征频率计算示例

## 实验名称

第 3 章 轴承与齿轮箱故障机理基础实验。

## 实验目标

本示例对应教材第三章《轴承与齿轮箱故障机理基础》，目标不是训练诊断模型，而是帮助学生建立从机械几何参数到特征频率的完整计算链条。示例默认使用 CWRU 6205-2RS JEM SKF 轴承参数，依次完成轴承特征频率计算、齿轮箱啮合频率计算以及 CWRU 多工况参考值核对。

## 依赖环境

- Python 3.9 或以上
- `numpy`
- `pandas`

```bash
python -m pip install numpy pandas
```

## 目录说明

- `characteristic_freq_calculator.py`：轴承和齿轮箱特征频率计算的核心模块
- `step1_compute_bearing_freqs.py`：计算轴承 BPFO、BPFI、BSF、CF 并输出摘要表
- `step2_compute_gearbox_freqs.py`：计算齿轮箱 GMF 与边带频率
- `step3_verify_cwru_freqs.py`：在 CWRU 四个工况转速下计算特征频率，并与文献参考值对比

## 运行顺序

### 任务一：计算轴承特征频率

```bash
python step1_compute_bearing_freqs.py --output-dir ./outputs
```

使用默认的 CWRU 轴承参数在四个常用转速下计算特征频率，结果保存为 `bearing_characteristic_freqs.csv`。

若只需计算指定转速：

```bash
python step1_compute_bearing_freqs.py --rpm 1797 --output-dir ./outputs
```

### 任务二：计算齿轮箱啮合频率

```bash
python step2_compute_gearbox_freqs.py --pinion-teeth 20 --gear-teeth 60 --pinion-rpm 1500 --output-dir ./outputs
```

修改 `--pinion-teeth`、`--gear-teeth`、`--pinion-rpm` 可适配不同齿轮配置。边带阶数由 `--num-sidebands` 控制，默认 3 阶。

### 任务三：核对 CWRU 参考值

```bash
python step3_verify_cwru_freqs.py --output-dir ./outputs
```

脚本计算 CWRU 四个工况的特征频率，并在 1797 rpm 下与文献（Randall & Antoni, 2011）参考值逐项对比，输出差值以验证计算正确性。

## 预期输出

- `bearing_characteristic_freqs.csv`：四个转速下的 BPFO、BPFI、BSF、CF（Hz）
- `gearbox_sidebands.csv`：GMF 及上下各三阶边带频率（Hz）
- `cwru_bearing_freqs_all_conditions.csv`：CWRU 四工况特征频率表
- `cwru_bearing_freqs_reference_check.csv`：1797 rpm 计算值与文献参考值的差值对比

## 说明

本示例只负责特征频率计算，不涉及信号采集、频谱分析或模型训练。若需要结合信号做频谱验证，应将本章计算结果与第二章的 FFT/包络谱方法联合使用。
