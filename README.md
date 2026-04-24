# 人工智能驱动的智能故障检测与诊断实践数字资产

本项目是面向“人工智能驱动的智能故障检测与诊断实践”课程建设的数字资产库，聚焦旋转机械、滚动轴承、齿轮箱等典型装备的故障检测、故障诊断、特征工程、迁移学习、小样本学习、异常检测和大模型诊断案例。

仓库内容包含可运行案例、实验教程、课程章节素材和数据处理脚本，适合用于课堂演示、实验教学、课程设计、科研入门和智能运维原型验证。

## 项目内容

| 目录 | 内容 |
| --- | --- |
| `ch01_cwru_data_pipeline` | CWRU 轴承数据读取、格式转换、滑动窗口切片和最小 `Dataset` 封装案例。 |
| `ch02_signal_preprocessing` | 信号预处理章节资源目录。 |
| `ch03_fault_mechanism_basics` | 故障机理基础章节资源目录。 |
| `ch04_shallow_model_bearing_diagnosis` | 浅层机器学习轴承故障诊断章节资源目录。 |
| `ch05_deep_learning_same_condition_diagnosis` | 同工况深度学习诊断章节资源目录。 |
| `ch06_feature_fusion` | 多源/多域特征融合章节资源目录。 |
| `ch07_transfer_learning_cross_condition` | 跨工况迁移学习章节资源目录。 |
| `ch08_meta_learning_few_shot_diagnosis` | 小样本元学习诊断章节资源目录。 |
| `ch09_contrastive_learning_diagnosis` | 对比学习诊断章节资源目录。 |
| `ch10_anomaly_detection` | 异常检测章节资源目录。 |
| `ch11_large_model_diagnosis_cases` | 大模型故障诊断案例实验教程，包含“特征文本化 + ChatGLM”和“序列 patching + GPT-2”两类路线说明。 |

## 快速开始

建议使用 Python 3.9 或更高版本。基础实验依赖如下：

```bash
python -m pip install numpy pandas scipy
```

如果需要运行深度学习或大模型相关案例，可按实验需要继续安装：

```bash
python -m pip install torch transformers peft datasets scikit-learn matplotlib
```

## 运行案例

### CWRU 数据流水线

进入第 1 章目录：

```bash
cd ch01_cwru_data_pipeline
```

准备 CWRU 12 kHz drive-end 10 类 `.mat` 文件后，按顺序执行：

```bash
python step1_inspect_cwru_mat.py --data-root C:\data\cwru --output-dir .\outputs
python step2_convert_cwru_formats.py --data-root C:\data\cwru --output-dir .\outputs
python step3_slice_cwru_windows.py --data-root C:\data\cwru --output-dir .\outputs --window-size 1024 --step-size 512
python step4_minimal_dataset.py --data-root C:\data\cwru --output-dir .\outputs --window-size 1024 --step-size 512
```

该流程会依次完成：检查 CWRU `.mat` 原始文件，生成统一长度的长序列 `.csv` 和 `.npz`，按固定窗口切分为可建模样本，并验证最小 `Dataset` 封装是否可用于后续训练。

### 大模型诊断案例

进入第 11 章目录：

```bash
cd ch11_large_model_diagnosis_cases
```

| 文件 | 说明 |
| --- | --- |
| `exp01_feature_textualized_chatglm.md` | 将统计特征文本化，接入 ChatGLM 类大语言模型进行诊断任务组织。 |
| `exp02_patched_gpt2.md` | 将振动序列切分为 patch，接入 GPT-2 类序列模型进行故障分类任务设计。 |

## 实验教程使用建议

| 阶段 | 建议 |
| --- | --- |
| 1 | 先运行 `ch01_cwru_data_pipeline`，建立从原始 `.mat` 到窗口样本的完整数据接口。 |
| 2 | 围绕第 2 至第 6 章补充信号预处理、故障特征、浅层模型、深度模型和特征融合实验。 |
| 3 | 进入第 7 至第 10 章，设计跨工况迁移、小样本诊断、对比学习和异常检测任务。 |
| 4 | 最后阅读第 11 章，把传统故障诊断样本组织方式扩展到大模型输入范式。 |

## 常用轴承故障诊断数据集链接

以下链接优先列出官方主页、论文配套仓库或主流数据平台入口。部分数据集可能需要注册、登录、申请权限或遵守非商业/署名许可，使用前请核对原始页面的授权条款和引用要求。

| 数据集 | 主要用途 | 链接 |
| --- | --- | --- |
| Case Western Reserve University Bearing Data Center, CWRU | 轴承分类、基准对比、入门教学 | https://engineering.case.edu/bearingdatacenter/download-data-file |
| CWRU Bearing Data Center overview | CWRU 数据说明、试验条件 | https://engineering.case.edu/bearingdatacenter |
| Paderborn University Bearing Data Center, PU/KAt | 多工况、真实/人工损伤、迁移学习 | https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter |
| Paderborn data sets and download | PU 数据下载页 | https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download |
| NASA IMS Bearing Dataset | 轴承退化、RUL、预测性维护 | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ |
| NASA IMS Bearings direct zip | IMS 轴承数据下载 | https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip |
| FEMTO-ST / PRONOSTIA / IEEE PHM 2012 Bearing | 轴承加速寿命、RUL 预测 | https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip |
| NASA PCoE Data Repository | PHM 多类公开数据集总入口 | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ |
| MFPT Fault Data Sets | 轴承与齿轮诊断、CBM 算法测试 | https://www.mfpt.org/fault-data-sets/ |
| XJTU-SY Bearing Datasets | 轴承全寿命退化、RUL 预测 | https://biaowang.tech/xjtu-sy-bearing-datasets/ |
| Jiangnan University Bearing Dataset, JNU | 不同转速下轴承故障分类 | https://github.com/ClarkGableWang/JNU-Bearing-Dataset |
| Southeast University Mechanical Datasets, SEU | DDS 平台轴承/齿轮故障诊断 | https://github.com/cathysiyu/Mechanical-datasets |
| HUST Bearing Dataset | 多轴承、多缺陷、多工况球轴承诊断 | https://data.mendeley.com/datasets/cbv7jyx4p9/3 |
| University of Ottawa Rolling-element Dataset, UORED-VAFCLS | 振动+声学轴承故障分类 | https://data.mendeley.com/datasets/y2px5tg92h/5 |
| University of Ottawa Electric Motor Dataset, UOEMD-VAFCVS | 电机多故障，含轴承故障，变速/恒速 | https://data.mendeley.com/datasets/msxs4vj48g/2 |
| MAFAULDA Machinery Fault Database | 旋转机械多传感器诊断，含轴承内/外圈故障 | https://www02.smt.ufrj.br/~offshore/mfs/page_01.html |
| Ball Bearing Vibration and Temperature Run-to-Failure Dataset | 振动+温度轴承全寿命退化 | https://data.mendeley.com/datasets/5hcdd3tdvb/6 |
| SCA Bearing Dataset | 真实纸浆厂轴承健康状态与故障数据 | https://data.mendeley.com/datasets/tdn96mkkpt/2 |
| Bearing Database, UPM/CITEF | 球面滚子轴承诱导故障振动数据 | https://zenodo.org/records/3898942 |
| Politecnico di Torino spherical roller bearing dataset | 中大型球面滚子轴承局部缺陷、多速度/载荷 | https://zenodo.org/records/13913254 |
| Politecnico di Torino single/dual/multi-bearing defects | 中大型轴承单/双/多缺陷数据，部分文件可能受限 | https://doi.org/10.5281/zenodo.14856937 |
| Vibration of rolling bearings under widely varying speed conditions | 变速工况轴承振动分析 | https://data.mendeley.com/datasets/6k6fbzc6vv/1 |
| Rolling Bearing Datasets for Transfer Learning Fault Diagnosis | 跨工况迁移学习轴承诊断 | https://data.mendeley.com/datasets/ykbc8hntzx/1 |
| Bearing and gearbox data for fault diagnostics application | 迁移学习故障诊断，含轴承/齿轮数据 | https://data.mendeley.com/datasets/fkp3nn4tp7/1 |
| Multi-mode Fault Diagnosis Datasets of Gearbox Under Variable Working Conditions | 变工况齿轮箱多模态故障，包含轴承/齿轮相关故障 | https://data.mendeley.com/datasets/p92gj2732w/2 |
| HIT-SM Bearing Datasets | 多源跨域轴承故障诊断 | https://github.com/hitwzc/Bearing-datasets |
| Awesome Bearing Dataset | 轴承故障检测公开数据集索引 | https://github.com/VictorBauler/awesome-bearing-dataset |
| IEEE IES Industrial AI Hub benchmark datasets | 工业 AI 基准数据集索引，含 IMS、PU 等 | https://ieee-ies-industrial-ai-lab.github.io/industrial-ai-hub/datasets/ |
| Kaggle CWRU MAT Full Dataset | CWRU 镜像，便于快速下载和教学演示 | https://www.kaggle.com/datasets/sufian79/cwru-mat-full-dataset |
| Kaggle NASA Bearing Dataset | NASA/IMS 衍生或镜像数据 | https://www.kaggle.com/datasets/jawadulkarim117/nasa-bearing-dataset |
| Kaggle XJTU-SY Bearing Dataset | XJTU-SY 镜像数据 | https://www.kaggle.com/datasets/zhenxinchen/xjtu-sy |
| Kaggle Machinery Fault Dataset | 第三方电机/机械故障数据入口，使用前需核验来源和标签 | https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset |

## 数据集选型建议

| 任务 | 推荐数据集 |
| --- | --- |
| 入门分类实验 | CWRU、JNU、HUST |
| 跨工况迁移学习 | PU、JNU、HUST、Rolling Bearing Transfer Learning 数据集 |
| RUL 与退化预测 | IMS、FEMTO/PRONOSTIA、XJTU-SY、Ball Bearing Run-to-Failure |
| 多传感器/声学诊断 | University of Ottawa、MAFAULDA、SCA Bearing |
| 更接近工业实际的大尺寸轴承 | Politecnico di Torino、UPM/CITEF 球面滚子轴承数据 |

## 说明

本仓库不直接分发第三方数据集原始文件。请从数据集发布方下载数据，并按照各数据集的许可协议、引用格式和使用限制开展实验。
