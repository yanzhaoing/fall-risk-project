"""
多模态特征融合模块

将来自不同来源的特征（姿态、步态、外观等）融合为统一表示
支持三种融合策略:
1. Concat — 简单拼接
2. Attention — 注意力加权融合
3. Gated — 门控融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

from config.settings import CFG_MODEL


class MultiModalFusion(nn.Module):
    """
    多模态特征融合网络

    将多个特征源融合为统一的特征向量

    Args:
        feature_dims: 各模态特征维度列表
        output_dim: 融合后输出维度
        strategy: 融合策略 "concat" | "attention" | "gated"
    """

    def __init__(
        self,
        feature_dims: List[int],
        output_dim: int = 256,
        strategy: str = CFG_MODEL.FUSION_STRATEGY,
    ):
        super().__init__()
        self.strategy = strategy
        self.num_modalities = len(feature_dims)
        self.feature_dims = feature_dims

        if strategy == "concat":
            self._init_concat(output_dim)
        elif strategy == "attention":
            self._init_attention(output_dim)
        elif strategy == "gated":
            self._init_gated(output_dim)
        else:
            raise ValueError(f"不支持的融合策略: {strategy}")

    def _init_concat(self, output_dim: int):
        """拼接融合: 各模态直接拼接后投影"""
        total_dim = sum(self.feature_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim),
        )

    def _init_attention(self, output_dim: int):
        """注意力融合: 学习各模态的重要性权重"""
        # 统一维度
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in self.feature_dims
        ])
        # 注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.Tanh(),
            nn.Linear(output_dim // 4, 1),
        )
        self.output_proj = nn.Linear(output_dim, output_dim)

    def _init_gated(self, output_dim: int):
        """门控融合: 每个模态有独立的门控"""
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in self.feature_dims
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid(),
            )
            for _ in self.feature_dims
        ])
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 各模态特征列表, 每个 shape (B, dim_i)

        Returns:
            融合特征, shape (B, output_dim)
        """
        if self.strategy == "concat":
            return self._forward_concat(features)
        elif self.strategy == "attention":
            return self._forward_attention(features)
        elif self.strategy == "gated":
            return self._forward_gated(features)

    def _forward_concat(self, features: List[torch.Tensor]) -> torch.Tensor:
        """拼接融合前向传播"""
        x = torch.cat(features, dim=-1)  # (B, total_dim)
        return self.fusion(x)            # (B, output_dim)

    def _forward_attention(self, features: List[torch.Tensor]) -> torch.Tensor:
        """注意力融合前向传播"""
        # 统一维度
        projected = [
            proj(feat) for proj, feat in zip(self.projections, features)
        ]  # 各 (B, output_dim)

        # 计算注意力权重
        stacked = torch.stack(projected, dim=1)  # (B, N, output_dim)
        attn_scores = self.attention_net(stacked)  # (B, N, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, N, 1)

        # 加权求和
        fused = (stacked * attn_weights).sum(dim=1)  # (B, output_dim)
        return self.output_proj(fused)

    def _forward_gated(self, features: List[torch.Tensor]) -> torch.Tensor:
        """门控融合前向传播"""
        gated_features = []
        for proj, gate, feat in zip(
            self.projections, self.gates, features
        ):
            h = proj(feat)       # (B, output_dim)
            g = gate(h)          # (B, output_dim) 门控值
            gated_features.append(h * g)

        # 求和融合
        fused = sum(gated_features)  # (B, output_dim)
        return self.output_proj(fused)


class TemporalFusion(nn.Module):
    """
    时序融合模块

    在时间维度上融合多帧的特征
    使用 1D 卷积 + 注意力机制

    Args:
        feature_dim: 特征维度
        seq_len: 序列长度
    """

    def __init__(
        self,
        feature_dim: int = 256,
        seq_len: int = 30,
    ):
        super().__init__()

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 时序特征, shape (B, seq_len, feature_dim)
        Returns:
            融合后的特征, shape (B, feature_dim)
        """
        # 1D 卷积
        h = x.transpose(1, 2)           # (B, feature_dim, seq_len)
        h = self.temporal_conv(h)        # (B, feature_dim, seq_len)
        h = h.transpose(1, 2)           # (B, seq_len, feature_dim)

        # 时序注意力池化
        attn = self.temporal_attn(h)     # (B, seq_len, 1)
        out = (h * attn).sum(dim=1)      # (B, feature_dim)

        return out
