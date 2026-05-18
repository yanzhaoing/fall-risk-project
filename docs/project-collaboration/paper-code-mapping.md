# 论文方法到代码模块映射

## 说明

当前附带的三篇论文里，两篇更适合作为“骨架方法分类与技术选型综述”，一篇更适合作为“未来增强方向”的参考，而不是当前仓库的逐行复现来源。

- `Deep learning for 3D skeleton-based action recognition: a comprehensive review of methods, datasets, and future directions`
  适合用于建立当前仓库中 RNN / Transformer / GCN / 数据集 / 训练范式的定位框架。
- `Graph network learning for human skeleton modeling: a survey`
  适合用于建立图结构骨架建模、拓扑设计、姿态估计与下游任务之间的关系。
- `Adaptive Multi-scale Lagrange Dynamics Spatial-Temporal Network for 3D Skeleton-based Human Motion Prediction`
  适合用于判断当前仓库距离更强的物理约束时空建模还差哪些模块。

## 一、骨架动作识别综述 → 当前代码

### 论文中的主方法族

综述将深度学习方法大致分为：

- RNN / LSTM 类时序模型
- CNN 类表征模型
- GCN / 图网络模型
- Transformer 类模型
- 自监督、半监督、无监督等训练范式

### 在仓库中的对应实现

#### 1. RNN / LSTM 路线

- 代码位置：`src/models/gait_analysis.py`
- 对应类：`GaitLSTM`
- 训练入口：`scripts/train.py --model gait_lstm`
- 对应任务：
  - `classification`
  - `regression`
  - `multimodal`（作为 gait backbone）

这条线是当前仓库里最直接的“骨架序列时序建模”实现。

#### 2. Transformer 路线

- 代码位置：`src/models/gait_analysis.py`
- 对应类：`GaitTransformer`
- 训练入口：`scripts/train.py --model gait_transformer`

这部分对应综述中“基于自注意力的骨架序列建模”方法族。

#### 3. GCN / 图网络路线

- 代码位置：`src/models/stgcn.py`
- 对应类：`STGCN`、`STGCNClassifier`
- 核心图结构：`COCO_EDGES`、`build_adjacency`
- 训练入口：`scripts/train.py --model stgcn`

这部分对应综述里最典型的骨架图建模路线，也和 Yan et al. 的 ST-GCN 思路一致。

#### 4. 姿态估计前端

- 代码位置：`src/models/pose_estimation.py`
- 对应实现：`MediaPipePose`、`HRNetPose`
- 上游检测：`src/models/backbone.py::HumanDetector`

综述中“骨架数据获取与表示”这一层，在仓库里对应的是“人体检测 + 姿态估计”前端，而不是训练骨干本身。

#### 5. 数据集层

- 代码位置：`src/data/dataset.py`
- 当前显式支持：`UPFallDataset`、`Le2iDataset`、NTU 目录扫描
- 数据准备入口：`scripts/download_data.py`

综述里关于 NTU、通用骨架动作数据集的讨论，对当前仓库的数据选择最有参考价值。

## 二、图网络骨架建模综述 → 当前代码

### 论文中的主观点

图网络综述强调：

- 人体骨架天然适合图建模
- 节点是关节，边是骨骼连接或可学习关系
- 关键问题包括拓扑建模、长程依赖、时空耦合、任务迁移
- 应用域覆盖姿态估计、动作识别、运动预测等

### 在仓库中的对应实现

#### 1. 静态骨架拓扑建模

- 代码位置：`src/models/stgcn.py`
- 对应实现：`COCO_EDGES` 与 `build_adjacency`

这部分是仓库里最明确的“骨架图”定义层。

#### 2. 图时空耦合建模

- 代码位置：`src/models/stgcn.py`
- 对应实现：`GraphConvolution` + `TemporalConv` + `STGCNBlock`

这部分对应综述中最核心的“空间图卷积 + 时间卷积”骨架学习范式。

#### 3. 图模型到风险任务的迁移

- 代码位置：
  - `src/models/stgcn.py`
  - `src/models/risk_scoring.py`
  - `src/models/multimodal_risk.py`

当前仓库并不是做标准动作分类，而是把骨架建模结果继续送入风险评分头或多模态融合头，因此属于“骨架图表示向下游风险任务迁移”的实现。

#### 4. 图网络综述与当前仓库的差距

当前仓库的图路线仍偏经典 ST-GCN 形态，尚未实现：

- 可学习动态图拓扑
- 多尺度图结构重建
- 更强的长程依赖建模
- 图对比学习或图自监督预训练

## 三、AMLD-STNet 论文 → 当前代码

### 论文里的关键模块

AMLD-STNet 重点强调：

- Lagrange dynamics 建模
- 基于关节受力关系的 adjacency 设计
- 多尺度动态建模
- 旋转运动与 Euler angle 约束
- 物理规律与神经网络的耦合

### 当前仓库已接近的部分

- `src/models/stgcn.py`
  已有骨架图拓扑和时空图卷积基础。
- `src/models/gait_analysis.py`
  已有骨架序列的时序建模骨干。
- `src/training/losses.py`
  已有可扩展的损失函数位置，适合加入新的物理一致性约束。

### 当前仓库尚未实现的部分

- Lagrange dynamics 网络
- 基于动力学变量的关节力邻接矩阵
- 多尺度动力学分支
- Euler angle consistency loss
- 面向未来姿态预测的物理建模流程

### 如果以后要往这篇论文靠拢，最自然的落点

1. 在 `src/models/stgcn.py` 中把固定邻接矩阵扩展成可学习 / 动态 / 多尺度邻接。
2. 在 `src/models/gait_analysis.py` 或新增模块中加入速度、加速度、角度等动力学变量分支。
3. 在 `src/training/losses.py` 中新增旋转一致性或物理约束损失。
4. 在当前跌倒风险任务中，把“动力学异常”作为风险评分输入，而不是只看动作类别或粗粒度步态特征。

## 四、仓库当前方法与论文关系的结论

### 已经实现

- 骨架序列的 RNN / LSTM 建模
- 骨架序列的 Transformer 建模
- 经典 ST-GCN 图时空建模
- 多模态特征融合框架
- 风险评分头、风险等级划分与时间平滑

### 部分实现

- 多模态风险评分
  当前模型框架有了，但环境特征仍主要是占位向量。
- 姿态估计双后端
  MediaPipe 可用，HRNet 仍未真正完成。

### 尚未实现

- 物理约束时空建模
- 动态图 / 多尺度图 / 力学邻接建模
- 真正基于临床或连续监测标注的风险评分学习
- 自监督 / 半监督骨架预训练

## 五、对当前项目最实际的论文使用方式

最适合当前仓库的用法不是“照着论文逐层抄”，而是分层吸收：

1. 先把两篇综述当作当前仓库方法定位图谱，明确现在属于哪几条骨架建模路线。
2. 把 AMLD-STNet 当作下一阶段增强方向，用来指导我们未来在哪些模块上升级时空建模和物理约束。
3. 在正式引入新模型前，先确认它服务的是“跌倒风险前置预警”还是“骨架动作识别 / 运动预测”本身，避免论文任务和项目任务错位。
