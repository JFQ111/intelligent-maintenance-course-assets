# CWRU 数据读取与处理示例

## 实验名称

第 1 章 CWRU 数据读取与处理示例

## 实验目标

本示例对应教材第一章《数据读取与处理》，目标不是训练模型，而是打通从 CWRU 原始 `.mat` 文件到可建模样本的完整数据链条。示例默认使用 CWRU 12 kHz 驱动端 10 类子集，依次完成文件检查、格式转换、滑动窗口切片和最小 `Dataset` 封装。

## 依赖环境

- Python 3.9 或以上
- `numpy`
- `pandas`
- `scipy`
- `torch` 为可选依赖，仅在希望直接继承 `torch.utils.data.Dataset` 时需要

可以使用如下命令安装基础依赖：

```bash
python -m pip install numpy pandas scipy
```

## 数据准备

请在 `--data-root` 指定的目录中放置以下 10 个 CWRU 原始文件：

- `097.mat`
- `105.mat`
- `118.mat`
- `130.mat`
- `169.mat`
- `185.mat`
- `197.mat`
- `209.mat`
- `222.mat`
- `234.mat`

这些文件对应正常状态以及不同故障部位、不同故障尺度的 10 个类别。示例统一从驱动端变量中读取时间序列。

## 目录说明

- `common.py`：共享常量、文件映射、读取函数和切片函数
- `step1_inspect_cwru_mat.py`：读取 `.mat` 文件并保存摘要表
- `step2_convert_cwru_formats.py`：生成宽表 `.csv` 和长序列 `.npz`
- `step3_slice_cwru_windows.py`：生成窗口级 `.npz`
- `step4_minimal_dataset.py`：实例化最小 `Dataset`

## 运行顺序

### 第一步：检查原始 `.mat` 文件

```bash
python step1_inspect_cwru_mat.py --data-root C:\data\cwru --output-dir .\outputs
```

该脚本会读取全部 10 个文件，检查驱动端变量名，并把采样点数、均值、标准差和转速信息保存为 `cwru_mat_summary.csv`。

### 第二步：转换为统一格式

```bash
python step2_convert_cwru_formats.py --data-root C:\data\cwru --output-dir .\outputs
```

该脚本会把每类长序列统一截断为 119808 点，并生成：

- `cwru_12k_drive_end_10class_aligned.csv`
- `cwru_12k_drive_end_10class_aligned.npz`

其中宽表 `.csv` 的组织方式为“行 = 采样点，列 = 类别信号”。

### 第三步：滑动窗口切片

```bash
python step3_slice_cwru_windows.py --data-root C:\data\cwru --output-dir .\outputs --window-size 1024 --step-size 512
```

该脚本会读取上一步生成的长序列 `.npz`，按固定窗口切片并生成窗口级样本：

- `cwru_12k_drive_end_10class_windows_w1024_s512.npz`

该 `.npz` 文件固定保存三个键：

- `signals`
- `labels`
- `label_names`

### 第四步：验证最小 `Dataset`

```bash
python step4_minimal_dataset.py --data-root C:\data\cwru --output-dir .\outputs --window-size 1024 --step-size 512
```

该脚本会读取窗口级 `.npz`，实例化最小 `Dataset`，并打印：

- 数据集长度
- 单条样本的形状
- 标签编号与标签名称映射
- 前几个样本的标签与基本统计量

## 预期结果

若全部步骤成功完成，在默认参数下应得到如下结果：

- 原始长序列 `.npz` 中，`signals` 形状为 `(10, 119808)`
- 窗口级 `.npz` 中，`signals` 形状为 `(2330, 1024)`
- 窗口级 `.npz` 中，`labels` 形状为 `(2330,)`
- 每一类长序列可切出 233 个窗口样本

## 说明

本示例只负责建立统一数据接口，不负责信号预处理、特征提取或模型训练。后续章节会直接复用本目录生成的窗口级样本文件。
