"""
ST-GCN（时空图卷积网络）

骨架动作识别的 SOTA 架构。
把骨架数据建模为图结构：
- 节点 = 关节（COCO 17 keypoints）
- 边 = 骨骼连接
- 空间图卷积：捕捉关节间关系（哪些关节同时运动）
- 时间卷积：捕捉时序变化（关节运动轨迹）

输入: (batch, 30, 51) → reshape → (batch, 3, 30, 17)
输出: (batch, 256) 特征向量

参考: Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── COCO 17 关键点骨架拓扑 ──────────────────────────────────
# 0:nose 1:L_eye 2:R_eye 3:L_ear 4:R_ear
# 5:L_shoulder 6:R_shoulder 7:L_elbow 8:R_elbow 9:L_wrist 10:R_wrist
# 11:L_hip 12:R_hip 13:L_knee 14:R_knee 15:L_ankle 16:R_ankle

COCO_EDGES = [
    # 头部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 躯干
    (0, 5), (0, 6), (5, 6),
    (5, 11), (6, 12), (11, 12),
    # 左臂
    (5, 7), (7, 9),
    # 右臂
    (6, 8), (8, 10),
    # 左腿
    (11, 13), (13, 15),
    # 右腿
    (12, 14), (14, 16),
]


def build_adjacency(num_nodes=17, edges=None, self_loop=True):
    """
    构建归一化邻接矩阵

    Args:
        num_nodes: 节点数
        edges: 边列表 [(i,j), ...]
        self_loop: 是否加自连接

    Returns:
        归一化邻接矩阵 (num_nodes, num_nodes)
    """
    if edges is None:
        edges = COCO_EDGES

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0  # 无向图

    if self_loop:
        np.fill_diagonal(A, 1.0)

    # D^{-1/2} A D^{-1/2} 归一化
    D = A.sum(axis=1)
    D_inv_sqrt = np.power(D, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat

    return torch.tensor(A_norm, dtype=torch.float32)


class GraphConvolution(nn.Module):
    """
    空间图卷积层

    公式: Z = A_norm @ X @ W
    其中 A_norm 是归一化邻接矩阵，X 是节点特征，W 是可学习权重

    这是最简单的图卷积实现（Kipf & Welling, 2017）。
    ST-GCN 原文用的是空间分区策略（K=3），但简单 GCN 已经能工作。
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 卷积等价于对每个节点做线性变换
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, A):
        """
        Args:
            x: (batch, channels, frames, joints)
            A: (joints, joints) 归一化邻接矩阵
        Returns:
            (batch, out_channels, frames, joints)
        """
        # 先图卷积：聚合邻居信息
        # x: (B, C, T, V) → (B, C, T, V)
        x = torch.einsum('bctv,vw->bctw', x, A)

        # 再线性变换
        x = self.conv(x)  # (B, C', T, V)

        x = self.bn(x)
        return x


class TemporalConv(nn.Module):
    """
    时间卷积层

    用 1D 卷积在时间维度上提取特征。
    类似于 TCN（Temporal Convolutional Network）的设计：
    - 大 kernel 捕捉长时间依赖
    - causal padding 保证时间因果性
    """

    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        # causal padding: 左边填充 kernel_size-1，右边不填充
        self.padding = (kernel_size - 1, 0)
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),  # 只在时间维度卷积
            stride=(stride, 1),
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, frames, joints)
        Returns:
            (batch, channels, frames', joints)
        """
        # causal padding
        x = F.pad(x, (0, 0) + self.padding)  # 只 pad 时间维度
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class STGCNBlock(nn.Module):
    """
    ST-GCN 基本块

    结构: Spatial Graph Conv → ReLU → Temporal Conv → Residual → ReLU
    """

    def __init__(self, in_channels, out_channels, temporal_kernel=9,
                 stride=1, dropout=0.0, residual=True):
        super().__init__()

        # 空间图卷积
        self.spatial_conv = GraphConvolution(in_channels, out_channels)

        # 时间卷积
        self.temporal_conv = TemporalConv(
            out_channels, kernel_size=temporal_kernel,
            stride=stride, dropout=dropout,
        )

        # 残差连接
        self.residual = residual
        if residual:
            if in_channels != out_channels or stride != 1:
                self.residual_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual_conv = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        """
        Args:
            x: (batch, in_channels, frames, joints)
            A: (joints, joints)
        Returns:
            (batch, out_channels, frames', joints)
        """
        residual = x

        # 空间图卷积
        x = self.spatial_conv(x, A)
        x = self.relu(x)

        # 时间卷积
        x = self.temporal_conv(x)

        # 残差连接
        if self.residual:
            residual = self.residual_conv(residual)
            # 如果时间维度不同（stride>1），用平均池化正确降采样
            if residual.shape[2] != x.shape[2]:
                residual = F.adaptive_avg_pool2d(residual, (x.shape[2], x.shape[3]))
            x = x + residual

        x = self.relu(x)
        return x


class STGCN(nn.Module):
    """
    ST-GCN 完整模型

    架构:
        输入 (batch, 3, 30, 17)
        → STGCNBlock (64)
        → STGCNBlock (64)
        → STGCNBlock (128, stride=2)
        → STGCNBlock (128)
        → STGCNBlock (256, stride=2)
        → STGCNBlock (256)
        → Global Average Pooling
        → FC (256 → num_classes)

    参数量: ~800K（与 GaitLSTM 的 666K 相当）
    """

    def __init__(
        self,
        input_dim=3,           # x, y, confidence
        num_nodes=17,          # COCO 17 关键点
        num_classes=2,         # 跌倒/正常
        base_channels=64,      # 基础通道数
        temporal_kernel=9,     # 时间卷积核大小
        dropout=0.3,
        num_stages=3,          # ST-GCN 阶段数
    ):
        super().__init__()
        self.num_nodes = num_nodes

        # 注册邻接矩阵为 buffer（不参与梯度更新）
        A = build_adjacency(num_nodes)
        self.register_buffer('A', A)

        # 构建 ST-GCN blocks
        channels = [base_channels * (2 ** i) for i in range(num_stages)]
        # channels: [64, 128, 256] (num_stages=3)

        self.blocks = nn.ModuleList()
        in_ch = input_dim

        for i, out_ch in enumerate(channels):
            stride = 2 if i > 0 else 1  # 第一个 block 不降采样
            self.blocks.append(
                STGCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temporal_kernel=temporal_kernel,
                    stride=stride,
                    dropout=dropout,
                    residual=True,
                )
            )
            in_ch = out_ch
            # 每个通道阶段额外加一个 block（不降采样）
            self.blocks.append(
                STGCNBlock(
                    in_channels=out_ch,
                    out_channels=out_ch,
                    temporal_kernel=temporal_kernel,
                    stride=1,
                    dropout=dropout,
                    residual=True,
                )
            )

        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 输出维度
        self.output_dim = channels[-1]

    def forward(self, x):
        """
        Args:
            x: (batch, frames, features) 其中 features = num_nodes * 3
               或 (batch, channels, frames, joints)
        Returns:
            (batch, output_dim) 特征向量
        """
        # 如果输入是扁平格式，reshape
        if x.dim() == 3:
            batch, T, F = x.shape
            # F = num_nodes * 3
            x = x.view(batch, T, self.num_nodes, 3)  # (B, T, V, C)
            x = x.permute(0, 3, 1, 2)  # (B, C, T, V)

        # ST-GCN blocks
        for block in self.blocks:
            x = block(x, self.A)

        # 全局平均池化: (B, C, T, V) → (B, C, 1, 1) → (B, C)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x


class STGCNClassifier(nn.Module):
    """ST-GCN 分类器（包装 STGCN backbone + 分类头）"""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.output_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


def build_stgcn(num_classes=2, **kwargs):
    """构建 ST-GCN 模型"""
    backbone = STGCN(num_classes=num_classes, **kwargs)
    return STGCNClassifier(backbone, num_classes)


# ─── 测试 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # 测试模型
    model = build_stgcn(num_classes=2, base_channels=64, num_stages=3)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试输入
    x = torch.randn(4, 30, 51)  # (batch, frames, 17*3)
    out = model(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")

    # 测试不同输入格式
    x2 = torch.randn(4, 3, 30, 17)  # (batch, channels, frames, joints)
    out2 = model(x2)
    print(f"输入: {x2.shape} → 输出: {out2.shape}")
