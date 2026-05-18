# 🏥 老年人跌倒风险前置预警系统

> 挑战杯"揭榜挂帅"擂台赛 XH-202617  
> 基于多模态 AI 监测的老年人跌倒风险、心理健康、诈骗识别及预警研究  
> 发榜单位：海康威视 / 萤石

## 📌 项目概述

本项目聚焦**跌倒风险前置预判**，不是事后检测跌倒事件，而是在跌倒发生**之前**评估风险等级，实现提前预警。

当前仓库已经形成三条并行能力：

1. **二分类基线**：正常 / 跌倒
2. **连续风险评分**：输出 0-100 风险分数
3. **多模态风险评分**：步态表征与环境 / 轨迹特征融合

## 🧠 当前技术路线

```text
视频流 / 摄像头 / 萤石设备
    -> 人体检测（YOLOv8）
    -> 姿态估计（MediaPipe；HRNet 接口预留）
    -> 骨架时序建模（LSTM / Transformer / ST-GCN）
    -> 风险评分（分类 / 回归 / 多模态融合）
    -> 风险等级划分、时间平滑、事件处理
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+（推荐）
- PyTorch 2.0+

### 安装

```bash
pip install -r requirements.txt
```

### 配置来源

当前仓库默认配置入口是：

- `config/settings.py`

仓库中**没有** `config/risk_model.yaml`，训练与评估脚本主要通过：

- `config/settings.py` 中的默认值
- 命令行参数覆盖

来控制运行行为。

## 📦 数据准备

### 数据集辅助脚本

```bash
python scripts/download_data.py --dataset upfall
python scripts/download_data.py --dataset le2i
python scripts/download_data.py --dataset all
```

说明：

- 当前 `download_data.py` 主要提供数据集说明、下载地址和建议放置路径
- 还不是自动下载脚本

### 当前代码里用到的数据入口

- `UP-Fall`：由 `src/data/dataset.py::UPFallDataset` 读取
- `NTU`：训练 / 评估脚本默认从 `data/processed/ntu_coco` 读取 `keypoints.json`
- `Le2i`：代码层已有数据集类与下载说明，但当前主训练入口仍主要围绕 NTU / UP-Fall

## 🏋️ 训练

### 1. 二分类基线

```bash
python scripts/train.py \
  --model gait_lstm \
  --task classification \
  --dataset ntu \
  --data-dir data/processed/ntu_coco
```

### 2. 连续风险评分

```bash
python scripts/train.py \
  --model gait_lstm \
  --task regression \
  --dataset ntu \
  --data-dir data/processed/ntu_coco
```

### 3. 多模态风险评分

```bash
python scripts/train.py \
  --model gait_lstm \
  --task multimodal \
  --dataset ntu \
  --data-dir data/processed/ntu_coco
```

说明：

- `--model` 当前支持：`gait_lstm`、`gait_transformer`、`stgcn`
- `--task` 当前支持：`classification`、`regression`、`multimodal`
- NTU 路径默认是 `data/processed/ntu_coco`
- 当前 NTU 的多模态训练里，环境特征仍主要是占位向量，适合先作为结构验证而不是最终实验结论

## 📈 评估

### 1. 分类评估

```bash
python scripts/evaluate.py \
  --model gait_lstm \
  --task classification \
  --dataset ntu \
  --checkpoint checkpoints/best_model.pt
```

### 2. 风险评分评估

```bash
python scripts/evaluate.py \
  --model gait_lstm \
  --task regression \
  --dataset ntu \
  --checkpoint checkpoints/best_model.pt
```

### 3. 多模态评分评估

```bash
python scripts/evaluate.py \
  --model gait_lstm \
  --task multimodal \
  --dataset ntu \
  --checkpoint checkpoints/best_model.pt
```

## 🎥 演示

### 本地摄像头

```bash
python scripts/demo.py --source camera --camera-id 0 --checkpoint checkpoints/best_model.pt
```

### 视频文件

```bash
python scripts/demo.py --source video --path input.mp4 --checkpoint checkpoints/best_model.pt
```

### 萤石设备

```bash
python scripts/demo.py --source ezviz --device-serial YOUR_DEVICE_SERIAL --checkpoint checkpoints/best_model.pt
```

说明：

- 演示链路当前最适合使用风险评分类 checkpoint
- `predictor.py` 已支持直接加载训练得到的回归 / 多模态 `.pt` checkpoint
- 若未提供可用 checkpoint，演示链路不会得到真实风险输出

## 📁 当前项目结构

```text
config/
├── settings.py                    # 全局配置中心

docs/
└── project-collaboration/         # 长期协作、项目档案、论文映射、审计记录

scripts/
├── download_data.py               # 数据集说明与准备入口
├── train.py                       # 训练入口
├── evaluate.py                    # 评估入口
└── demo.py                        # 演示入口

src/
├── data/                          # 数据集与 DataLoader
├── features/                      # 步态特征工程
├── models/                        # LSTM / Transformer / ST-GCN / 融合 / 风险评分
├── training/                      # loss / metrics / trainer
├── inference/                     # predictor / pipeline
├── ezviz/                         # 萤石 API / 视频流 / 事件处理
└── utils/                         # 工具函数与可视化

tests/                             # 测试目录
```

## 📊 当前真实状态

### 已经具备的能力

- 支持 LSTM、Transformer、ST-GCN 三类骨架建模路线
- 支持分类、回归、多模态三种任务模式
- 已有风险等级划分、时间平滑、事件处理逻辑
- 已有萤石 API、视频流、演示链路骨架

### 当前仍在研究推进中的部分

- 风险评分标签当前大量依赖动作语义规则映射，而不是真实临床评分标注
- 多模态环境特征仍未完全接入真实训练流
- HRNet 姿态估计后端还未完整实现
- 数据下载脚本仍以说明性功能为主

## 🗂️ 长期协作与记录

为了支持后续持续迭代、论文对照、实验追踪和仓库修改，仓库内维护了以下长期文档：

- `docs/project-collaboration/long-term-workflow.md`
  默认协作流程与长期记录方式。
- `docs/project-collaboration/project-profile.md`
  当前仓库的真实项目地图、主流程和状态判断。
- `docs/project-collaboration/paper-code-mapping.md`
  论文方法与当前代码模块的映射关系。
- `docs/project-collaboration/code-audit.md`
  当前已修复问题与仍待继续处理的问题。
- `docs/project-collaboration/session-template.md`
  每次任务结束后的记录模板。

## 📅 当前阶段建议

当前最值得继续投入的方向是：

1. 补齐多模态环境 / 轨迹真实特征输入。
2. 建立更可信的风险评分标注方案。
3. 给训练、评估、演示三条主线补稳定的实验规范与默认 checkpoint 约定。
4. 把自动化测试和回归检查链路整理清楚。
