# 老人跌倒风险预测项目

本项目面向 XH-202617 赛题中的“跌倒风险”方向，目标是在居家养老场景下实现：

- 跌倒前风险预判
- 过程识别与持续监测
- 分级预警与响应闭环
- 面向比赛提交的可运行演示与验证材料

## 当前项目定位

当前仓库包含两条并行能力链路：

1. 模型链路
- 骨架时序模型：`scripts/train.py`
- 步态特征模型：`scripts/train_feature_model.py`
- 测试评估：`scripts/evaluate.py`、`scripts/evaluate_public_ntu.py`

2. 系统演示链路
- 比赛演示服务：`scripts/serve_demo.py`
- 内置系统评测：`scripts/evaluate_competition_demo.py`
- 视频/摄像头演示入口：`scripts/demo.py`

## 推荐参赛主线

面向初审，优先采用“可解释步态特征 + 风险评分 + 分级预警 + 演示系统”的路线：

- 公开骨架数据用于离线验证
- 特征模型用于稳定输出风险分数
- 预警模块负责四级分级与响应动作
- 演示系统展示前置预警、环境风险、视频分析与闭环逻辑

这条路线的优势是更容易形成完整系统，也更容易产出可解释的报告与测试材料。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

启动比赛演示服务：

```bash
python -B scripts/serve_demo.py
```

浏览器打开：

```text
http://127.0.0.1:7860
```

运行内置演示评估：

```bash
python -B scripts/evaluate_competition_demo.py --repeats 5
```

训练步态特征模型：

```bash
python -B scripts/train_feature_model.py --data-dir data/processed/ntu_coco
```

训练骨架时序模型：

```bash
python -B scripts/train.py --model gait_lstm --task regression --dataset ntu
```

## 目录说明

- `config/`：全局配置
- `scripts/`：训练、评估、演示、打包脚本
- `src/data/`：数据集与 DataLoader
- `src/features/`：步态特征、环境特征、个性化基线
- `src/models/`：时序模型、特征模型、多模态融合、风险评分
- `src/inference/`：推理、演示、视频分析
- `src/alerts/`：分级预警逻辑
- `web/`：比赛演示前端页面
- `docs/`：提交与实测准备材料

## 重要说明

当前仓库支持“公开数据集验证 + 比赛系统演示”，但真正冲击高分仍需要补齐：

- 萤石开放平台的实际调用证据
- 真实居家场景的实测视频或设备联动记录
- 专项研究报告、功能测试报告、部署说明
- 与最终演示一致的测试数据与结果截图

详见 [SUBMISSION_CHECKLIST.md](./SUBMISSION_CHECKLIST.md)。
