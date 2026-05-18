# 默认项目档案：fall-risk-project

## 项目定位

- 默认仓库：`yanzhaoing/fall-risk-project`
- 项目目标：老人跌倒风险前置预警，而不是事后跌倒检测
- 当前主线：骨架时序建模 + 连续风险评分 + 萤石视频流接入
- 当前支持的任务模式：
  - `classification`：跌倒 / 非跌倒二分类
  - `regression`：0-100 连续风险评分
  - `multimodal`：步态特征与环境/轨迹特征融合评分

## 当前真实仓库地图

### 根目录

- `README.md`
  仓库总说明与使用入口。
- `requirements.txt`
  当前依赖清单。
- `config/settings.py`
  全局配置中心，路径、模型超参数、训练参数、风险阈值、萤石 API 配置都集中在这里。
- `scripts/`
  训练、评估、演示、数据准备入口。
- `src/`
  核心实现代码。
- `docs/project-collaboration/`
  长期协作、项目档案、论文映射和审计记录。
- `tests/`
  仓库内存在测试目录，但本轮未通过目录列表接口完整展开具体测试文件。

### scripts

- `scripts/train.py`
  训练入口。支持 `gait_lstm`、`gait_transformer`、`stgcn` 三种骨干，以及三种任务模式。
- `scripts/evaluate.py`
  评估入口。当前与训练入口对齐，支持分类、回归和多模态任务评估。
- `scripts/demo.py`
  实时演示入口。支持本地摄像头、视频文件和萤石视频流。
- `scripts/download_data.py`
  数据集准备辅助脚本。目前主要提供数据集信息、下载链接和本地放置路径提示。

### src/data

- `src/data/dataset.py`
  数据集定义。包含通用 `FallDetectionDataset`、`UPFallDataset`、`Le2iDataset`。
- `src/data/dataloader.py`
  DataLoader 工厂，包含类别不平衡下的加权采样逻辑。

### src/features

- `src/features/gait_features.py`
  16 维步态特征工程实现，是当前推理链里很关键的一条支线。

### src/models

- `src/models/gait_analysis.py`
  `GaitLSTM`、`GaitTransformer`、`GaitRiskScorer`。
- `src/models/stgcn.py`
  ST-GCN 图时空骨架建模实现。
- `src/models/risk_scoring.py`
  连续风险评分与风险等级划分、个性化基线、时间平滑。
- `src/models/multimodal_risk.py`
  步态骨干 + 环境/轨迹特征融合评分。
- `src/models/fusion.py`
  concat / attention / gated 三种多模态融合策略。
- `src/models/backbone.py`
  人体检测与通用特征提取。
- `src/models/pose_estimation.py`
  MediaPipe / HRNet 姿态估计后端接口。
- `src/models/feature_mlp.py`
  基于 16 维步态特征的轻量 MLP 风险评分器。

### src/training

- `src/training/trainer.py`
  训练循环、验证、早停、检查点保存。
- `src/training/losses.py`
  `FocalLoss`、`RiskScoreLoss`、`ContrastiveLoss`。
- `src/training/metrics.py`
  分类指标与风险评分回归指标。

### src/inference

- `src/inference/pipeline.py`
  端到端推理流水线：人体检测 → 姿态估计 → 时序缓冲 → 风险预测。
- `src/inference/predictor.py`
  单次风险预测器。现在已经支持直接加载训练得到的回归 / 多模态 `.pt` checkpoint。

### src/ezviz

- `src/ezviz/api_client.py`
  萤石开放平台 API 封装。
- `src/ezviz/stream.py`
  实时视频流拉取。
- `src/ezviz/event_handler.py`
  风险事件记录与预警触发。

### src/utils

- `src/utils/helpers.py`
  设备、随机种子、目录创建等基础函数。
- `src/utils/visualization.py`
  关键点绘制与风险叠加层绘制。

## 当前真实数据流

### 训练流

1. `scripts/train.py` 解析命令行参数。
2. 从 `config/settings.py` 读取默认路径和超参数。
3. 从 `src/data/dataset.py` 组织 NTU / UP-Fall 数据样本。
4. 通过 `src/data/dataloader.py` 建立 train/val/test loader。
5. 在 `src/models/` 中构建 LSTM / Transformer / ST-GCN / 多模态模型。
6. 通过 `src/training/trainer.py` 完成训练、验证、保存最佳 checkpoint。

### 评估流

1. `scripts/evaluate.py` 根据 `--task` 重建对应模型。
2. 加载测试集并恢复 checkpoint。
3. 分类任务走 `compute_metrics`；回归 / 多模态任务走 `compute_risk_metrics`。
4. 输出结果并保存到 `results/`。

### 推理流

1. `scripts/demo.py` 建立视频源。
2. `src/inference/pipeline.py` 调用 `HumanDetector` + `PoseEstimator`。
3. 将关键点序列放入时序缓冲区。
4. `src/inference/predictor.py` 对关键点序列做风险预测。
5. `src/ezviz/event_handler.py` 按风险等级记录或触发通知。

## 当前仓库的实际状态判断

### 已经成型的部分

- 骨架时序建模主线已经具备训练、评估、演示三类入口。
- 分类、回归、多模态三种任务模式都已经在训练代码中成型。
- ST-GCN、LSTM、Transformer 三类骨架建模路线都已经进入代码层。
- 萤石接入链路已经有 API、视频流、事件处理器三块基础组件。
- 风险评分不只是二分类，还实现了等级划分、时间平滑、个性化基线。

### 仍处于研究原型阶段的部分

- 风险评分标签目前大量依赖动作语义规则映射，而不是真实临床风险标注。
- 多模态任务中环境 / 轨迹特征在当前数据流里仍以占位向量为主。
- `HRNetPose` 仍未完成实际推理实现。
- `download_data.py` 目前主要是手动下载引导，不是自动下载脚本。

## 本轮审计后确认的关键问题

- 旧版 `README` 中存在过时命令，误指向不存在的 `config/risk_model.yaml`。
- `train.py` 在分类任务的测试集阶段，曾把原始 logits 直接当预测类别使用；本轮已修正。
- 推理端 `predictor.py` 之前不能直接加载训练得到的回归 / 多模态 `.pt` checkpoint；本轮已修正。
- 训练与评估入口之前存在任务模式不完全对齐的问题；本轮已补齐。

## 当前最值得继续补齐的方向

1. 把真实环境 / 轨迹特征正式接入多模态训练与推理。
2. 给回归任务补更可信的风险标注来源，而不是只依赖动作语义映射。
3. 为演示链路补一套稳定可复现实验配置和默认 checkpoint 规范。
4. 扩充自动化测试，让训练、评估、推理三条主线各自至少有一条可回归验证路径。
