# 🏥 老年人跌倒风险前置预警系统

> 挑战杯"揭榜挂帅"擂台赛 XH-202617
> 基于多模态AI监测的老年人跌倒风险、心理健康、诈骗识别及预警研究
> 发榜单位：海康威视 / 萤石

## 📌 项目概述

本项目聚焦**跌倒风险前置预判**，不是事后检测跌倒事件，而是在跌倒发生**之前**评估风险等级，实现提前预警。

### 核心创新点

1. **连续性风险评分模型** — 输出 0-100 分的风险评分，而非二分类
2. **多维度融合预警** — 融合萤石多个API输出为综合风险评估
3. **个性化基线** — 每位老人建立独立的正常状态基线，检测偏离
4. **分级预警+联动响应** — 低/中/高风险对应不同响应策略

### 技术架构

```
萤石摄像头取流 → 人体检测（萤石API / YOLOv8）
                → 姿态估计（MediaPipe / HRNet）
                → 步态分析（LSTM / Transformer）
                → 风险评分（多特征融合）
                → 分级预警 → 推送通知
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+（推荐）
- RTX 3060 12GB 或更高

### 安装

```bash
cd fall-risk-project
pip install -r requirements.txt
```

### 数据集准备

```bash
python scripts/download_data.py --dataset upfall
python scripts/download_data.py --dataset le2i
```

### 训练

```bash
python scripts/train.py --config config/risk_model.yaml
```

### 评估

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset upfall
```

### 演示

```bash
python scripts/demo.py --source camera --camera-id 0
python scripts/demo.py --source video --path input.mp4
```

## 📁 项目结构

```
src/
├── data/           # 数据集加载与预处理
├── models/         # 模型定义
├── training/       # 训练逻辑
├── inference/      # 推理流水线
├── ezviz/          # 萤石API对接
└── utils/          # 工具函数
```

## 📊 数据集

| 数据集 | 传感器 | 规模 | 用途 |
|--------|--------|------|------|
| UP-Fall | RGB+Depth+IR+IMU | 17人, 5活动+6跌倒 | 主训练集 |
| Le2i | RGB | 4场景 | 补充验证 |
| NTU RGB+D | RGB+Depth+IR | 56880视频 | 泛化测试 |

## 📅 时间线

- **5月**: 项目搭建 + 萤石接入 + 基础人体检测
- **6月**: 姿态估计 + 步态分析模型训练
- **7月**: 风险评分模型 + 分级预警
- **8月**: 整合 + 测试 + 文档
- **9月5日前**: 提交
