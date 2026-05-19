# 跌倒风险前置预警实验口径

## 1. 本轮统一原则

- 任务目标: 预测“跌倒前置风险”，不是只做跌倒后二分类。
- 标签来源: 使用骨架动态、姿态异常、垂向坍塌趋势、房间先验生成连续伪标签。
- 多模态定义: 步态 backbone + 18 维 scene/context 特征，不再使用全零占位向量。
- 主对比数据: 当前优先使用 `NTU` 主线复现实验。

## 2. 标签口径

`RiskAwareFallDataset` 为每个时间窗生成 0-100 连续风险分数，核心组成:

- 运动不稳定性: 平均速度、峰值速度、平均加速度、垂向位移范围
- 姿态异常: 躯干倾斜、躯干倾斜波动、支撑宽度、外接框形态波动
- 坍塌趋势: 质心下坠、人体高度收缩
- 场景/动作先验: 卫生间、厨房、弯腰拾物、坐下/起立等风险更高

解释原则:

- ADL 不再一律映射到固定低分，而是允许出现中低风险波动。
- Fall 样本不直接等于极高分，而是根据失稳强度拉到更高区间。
- 该标签仍是伪标签，不应表述为临床金标准。

## 3. scene/context 特征口径

18 维特征组成:

- 4 维房间先验: living room / bedroom / kitchen / bathroom
- 4 维轨迹离散度: center x/y std, x/y range
- 3 维运动强度: mean speed, max speed, mean accel
- 2 维坍塌趋势: vertical drop, height drop
- 2 维外接框形态: aspect mean, aspect std
- 2 维躯干稳定性: torso lean mean, torso lean std
- 1 维支撑稳定性: support width mean

## 4. 推荐实验顺序

### A. 分类基线

```bash
python scripts/train.py \
  --model gait_lstm \
  --task classification \
  --dataset ntu \
  --data-dir data/processed/ntu_coco
```

### B. 前置风险回归

```bash
python scripts/train.py \
  --model gait_lstm \
  --task regression \
  --dataset ntu \
  --data-dir data/processed/ntu_coco
```

### C. 多模态前置风险

```bash
python scripts/train.py \
  --model gait_lstm \
  --task multimodal \
  --dataset ntu \
  --data-dir data/processed/ntu_coco
```

### D. 评估

```bash
python scripts/evaluate.py \
  --model gait_lstm \
  --task regression \
  --dataset ntu \
  --checkpoint checkpoints/best_model.pt
```

```bash
python scripts/evaluate.py \
  --model gait_lstm \
  --task multimodal \
  --dataset ntu \
  --checkpoint checkpoints/best_model.pt
```

## 5. 统一报告指标

### 分类任务

- Accuracy
- Precision
- Recall
- F1
- Specificity
- AUC-ROC
- AUC-PR

### 风险评分任务

- MAE
- RMSE
- Spearman
- Pearson Correlation
- `within_10`: 预测分数与标签分数相差不超过 10 分的比例
- `within_20`: 预测分数与标签分数相差不超过 20 分的比例
- Level Accuracy: 预测风险等级与标签风险等级一致的比例

## 6. 研究报告结果表模板

### 表 1. 分类基线对比

| Model | Task | Accuracy | Precision | Recall | F1 | Specificity | AUC-ROC | AUC-PR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GaitLSTM | classification |  |  |  |  |  |  |  |
| Transformer | classification |  |  |  |  |  |  |  |
| ST-GCN | classification |  |  |  |  |  |  |  |

### 表 2. 风险评分主结果

| Model | Label Scheme | scene/context | MAE | RMSE | Spearman | Corr | within_10 | within_20 | Level Acc |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GaitLSTM | legacy action prior | zero vector |  |  |  |  |  |  |  |
| GaitLSTM | pre-fall pseudo label | real 18-dim |  |  |  |  |  |  |  |
| Transformer | pre-fall pseudo label | real 18-dim |  |  |  |  |  |  |  |
| ST-GCN | pre-fall pseudo label | real 18-dim |  |  |  |  |  |  |  |

### 表 3. 消融实验

| Setting | Pseudo Label | scene/context | MAE | RMSE | Spearman | Level Acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Regression baseline | old | zero |  |  |  |  |
| Label only | new | zero |  |  |  |  |
| Label + multimodal | new | real |  |  |  |  |

## 7. 写报告时需要明确说明的限制

- 当前连续风险分数属于基于骨架序列的工程化伪标签，不是医生标注的临床风险评分。
- 当前房间先验主要来自数据集元信息和骨架轨迹，而不是完整的环境语义分割结果。
- 如果后续接入萤石真实家庭视频、区域划分和设备联动，可进一步把 scene/context 特征升级为真实居家环境特征。
