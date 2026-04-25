# 第 8 章 元学习核心思想与小样本诊断示例

## 实验名称

第 8 章 元学习核心思想与小样本诊断实验。

## 实验目标

本示例对应教材第八章《元学习核心思想与小样本诊断》，依次完成：标签重映射机制演示、ProtoNet 元训练与元测试、MAML（FOMAML）元训练与元测试、两种方法的对比评估。

## 依赖环境

- Python 3.9 或以上
- `numpy`、`torch`

```bash
python -m pip install numpy torch
```

`torch` 为必须依赖。如无 GPU，脚本会自动使用 CPU。

## 数据准备

本章代码依赖第一章生成的窗口级 npz 文件：

```
cwru_12k_drive_end_10class_windows_w1024_s512.npz
```

通过 `--npz-path` 参数传入。若不传或传入 `--demo`，则使用随机模拟信号，用于验证代码逻辑（不产生有诊断意义的准确率）。

## 目录说明

- `episode_sampler.py`：核心模块，提供 `FewShotDataset`、`EpisodeSampler`、`Episode` 三个类，负责 episode 采样与标签重映射。
- `embed_net.py`：共享 1D-CNN 编码器 `EmbedNet` 与带分类头版本 `EmbedNetWithHead`，两种方法均复用此模块。
- `step1_label_remapping_demo.py`：标签重映射机制演示，无需 GPU，无需真实数据。
- `step2_protonet_train_eval.py`：ProtoNet 元训练与元测试。
- `step3_maml_train_eval.py`：MAML（FOMAML）元训练与元测试。
- `step4_compare_protonet_maml.py`：在相同 episode 集合上对比两种方法的准确率。

## 运行顺序

### 任务一：标签重映射演示（无需数据/GPU）

```bash
python step1_label_remapping_demo.py
```

### 任务二：ProtoNet 元训练与元测试

```bash
# 使用模拟数据（验证代码）
python step2_protonet_train_eval.py --demo

# 使用真实 CWRU 数据
python step2_protonet_train_eval.py --npz-path ./outputs/cwru_12k_drive_end_10class_windows_w1024_s512.npz
```

### 任务三：MAML 元训练与元测试

```bash
python step3_maml_train_eval.py --demo
python step3_maml_train_eval.py --npz-path ./outputs/cwru_12k_drive_end_10class_windows_w1024_s512.npz
```

### 任务四：ProtoNet vs MAML 对比

```bash
python step4_compare_protonet_maml.py --demo
python step4_compare_protonet_maml.py --npz-path ./outputs/cwru_12k_drive_end_10class_windows_w1024_s512.npz
```

## 说明

本示例使用 FOMAML（一阶近似）降低计算开销。完整二阶 MAML 需要安装 `higher` 库，可参考教材注释修改 `step3_maml_train_eval.py` 中的 `inner_loop` 函数。
