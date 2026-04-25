# 第 4 章 JNU 600 rpm 浅层模型故障诊断示例

本示例对应教材第 4 章“使用浅层模型实现轴承故障诊断”，使用 JNU 轴承数据集的 600 rpm 单一工况，演示“人工特征 + 浅层模型”的完整故障诊断流程。

本示例的教学目标不是追求最高分类精度，而是帮助读者看清传统机器学习故障诊断的完整链条：

`振动信号 -> 25 维特征向量 -> 特征矩阵 X 与标签 y -> 决策树 / SVM / 随机森林 -> 混淆矩阵与分类指标`

## 1. 数据位置

请将 JNU 数据放在以下任一目录：

- `datasets/JNU/`
- `datatsets/JNU/`

当前脚本会优先查找 `datasets/JNU/`，若不存在则回退到 `datatsets/JNU/`。

本示例只读取 600 rpm 工况下的四类文件：

- `n600_3_2.csv`
- `ib600_2.csv`
- `ob600_2.csv`
- `tb600_2.csv`

类别含义如下：

- `n`：正常
- `ib`：内圈故障
- `ob`：外圈故障
- `tb`：滚动体故障

## 2. 程序结构

```text
examples/ch04_jnu_600rpm_shallow_diagnosis/
├── README.md
├── feature_utils.py
├── step1_inspect_jnu_600rpm.py
├── step2_extract_25_features.py
├── step3_split_dataset.py
├── step4_train_decision_tree.py
├── step5_train_svm.py
├── step6_train_random_forest.py
├── step7_compare_models.py
└── outputs/
```

## 3. 运行顺序

请在项目根目录执行：

1. `python examples/ch04_jnu_600rpm_shallow_diagnosis/step1_inspect_jnu_600rpm.py`
2. `python examples/ch04_jnu_600rpm_shallow_diagnosis/step2_extract_25_features.py`
3. `python examples/ch04_jnu_600rpm_shallow_diagnosis/step3_split_dataset.py`
4. `python examples/ch04_jnu_600rpm_shallow_diagnosis/step4_train_decision_tree.py`
5. `python examples/ch04_jnu_600rpm_shallow_diagnosis/step5_train_svm.py`
6. `python examples/ch04_jnu_600rpm_shallow_diagnosis/step6_train_random_forest.py`
7. `python examples/ch04_jnu_600rpm_shallow_diagnosis/step7_compare_models.py`

## 4. 流程说明

本示例按如下顺序组织：

1. 读取 JNU 600 rpm 原始振动信号，并检查类别、标签和信号长度。
2. 将每条长信号切分为固定长度窗口样本。
3. 对每个窗口样本提取 25 个诊断特征，形成特征矩阵 `X_features`。
4. 将数据划分为训练集、验证集和测试集。
5. 分别训练决策树、支持向量机和随机森林。
6. 使用准确率、分类报告和混淆矩阵评价模型表现。
7. 汇总三种模型的指标，形成统一对比表。

为了让模型比较更公平，本示例在特征提取阶段会对四类样本进行窗口级平衡抽样，使各类样本数量一致。

## 5. 25 个特征组成

### 5.1 13 个时域特征

1. 均值
2. 绝对均值
3. 均方根值
4. 方差
5. 标准差
6. 峰值
7. 峰峰值
8. 偏度
9. 峭度
10. 波形因子
11. 峰值因子
12. 脉冲因子
13. 裕度因子

### 5.2 12 个频域特征

14. 重心频率
15. 均方频率
16. 均方根频率
17. 频率方差
18. 频率标准差
19. 主频位置
20. 主频幅值
21. 谱均值
22. 谱方差
23. 谱偏度
24. 谱峭度
25. 谱熵

## 6. 三种模型代表的建模思想

- 决策树：通过一系列特征阈值规则划分特征空间。
- SVM：通过最大间隔思想寻找更稳健的分类边界。
- 随机森林：通过多棵决策树投票提高分类稳定性，并可输出特征重要性。

## 7. 输出结果

运行完成后，`outputs/` 目录至少包含：

```text
jnu_600rpm_summary.csv
jnu_600rpm_features.npz
jnu_600rpm_features.csv
jnu_600rpm_split.npz
decision_tree_metrics.csv
decision_tree_confusion_matrix.csv
svm_metrics.csv
svm_confusion_matrix.csv
random_forest_metrics.csv
random_forest_confusion_matrix.csv
random_forest_feature_importance.csv
model_comparison.csv
```

最终模型对比结果保存在：

- `outputs/model_comparison.csv`

若希望查看随机森林最重视哪些特征，可查看：

- `outputs/random_forest_feature_importance.csv`

## 8. 教学说明

本示例只做同工况监督分类，不涉及跨工况迁移学习、深度学习、异常检测或元学习。程序重点是让学生理解浅层模型故障诊断的完整流程，而不是展示复杂网络结构或追求极限精度。
