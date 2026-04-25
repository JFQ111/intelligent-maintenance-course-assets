# 第 7 章迁移学习与领域泛化实验代码库

本目录对应教材第 7 章“迁移学习基本原理与跨工况诊断”的配套实验代码。实验数据只使用 `datasets/JNU/` 下的 JNU 轴承数据，不使用 CWRU 数据。

当前项目中原始数据实际也存在于 `datatsets/JNU/`。为保证后续复用，代码会优先查找 `datasets/JNU/`，若不存在再回退到 `datatsets/JNU/`。现在根目录下已经补了 `datasets/JNU` 联接，因此从项目根目录运行脚本时可以直接找到数据。

## 任务说明

本章代码围绕两个任务设定展开：

- `DA`（领域自适应）：训练阶段可见目标域数据，但不使用目标域故障标签。
- `DG`（领域泛化）：训练阶段不可见目标域数据，只用多个源域训练，目标域仅在最终测试时出现。

本实验将 600、800、1000 r/min 作为 3 个域，类别映射如下：

- `n`：正常，对应标签 `0`
- `ib`：内圈故障，对应标签 `1`
- `ob`：外圈故障，对应标签 `2`
- `tb`：滚动体故障，对应标签 `3`

域标签映射如下：

- `600 -> 0`
- `800 -> 1`
- `1000 -> 2`

## 目录结构

```text
examples/ch07_transfer_jnu/
├── README.md
├── 01_data_pipeline/
├── 02_common/
├── 03_domain_adaptation/
├── 04_domain_generalization/
└── 05_results/
```

## 环境依赖

- Python 3.9 或以上
- `numpy`
- `pandas`
- `torch`
- `scikit-learn`
- `matplotlib`
- `openpyxl`

`visualize.py` 中的绘图函数依赖 `scikit-learn` 与 `matplotlib`。

## 运行顺序

请在项目根目录执行以下命令：

1. `python examples/ch07_transfer_jnu/01_data_pipeline/step1_inspect_jnu_excel.py`
2. `python examples/ch07_transfer_jnu/01_data_pipeline/step2_convert_jnu_npz.py`
3. `python examples/ch07_transfer_jnu/01_data_pipeline/step3_slice_jnu_windows.py`
4. `python examples/ch07_transfer_jnu/01_data_pipeline/step4_build_da_dg_splits.py`
5. `python examples/ch07_transfer_jnu/01_data_pipeline/step5_check_dataset.py`
6. `python examples/ch07_transfer_jnu/03_domain_adaptation/dan/train_dan.py`
7. `python examples/ch07_transfer_jnu/03_domain_adaptation/dann/train_dann.py`
8. `python examples/ch07_transfer_jnu/03_domain_adaptation/adda/step1_pretrain_source.py`
9. `python examples/ch07_transfer_jnu/03_domain_adaptation/adda/step2_train_adda.py`
10. `python examples/ch07_transfer_jnu/04_domain_generalization/mixup/train_mixup.py`
11. `python examples/ch07_transfer_jnu/04_domain_generalization/coral/train_coral.py`
12. `python examples/ch07_transfer_jnu/04_domain_generalization/dg_dann/train_dg_dann.py`

## 数据流水线输出

数据流水线会在本目录下自动创建 `processed/`：

- `processed/jnu_long_signals.npz`
- `processed/jnu_windows.npz`
- `processed/da_600_to_1000.npz`
- `processed/dg_600_800_to_1000.npz`

结果会写入：

- `05_results/logs/`
- `05_results/checkpoints/`
- `05_results/metrics/`
- `05_results/figures/`

## 教学说明

本代码库不追求最高精度，重点是：

- 数据处理过程完整
- 目录结构清晰
- 训练逻辑可读
- 所有脚本能从项目根目录直接运行
- 能清楚体现 `DA` 与 `DG` 的任务边界
