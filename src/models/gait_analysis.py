"""
步态分析模块

从关键点时间序列中提取步态特征，判断步态异常程度
支持两种架构:
1. LSTM — 经典时序模型，适合小数据集
2. Transformer — 注意力机制，适合大数据集和长序列

输入: 关键点序列 (batch, seq_len, input_dim)
输出: 步态特征向量 (batch, hidden_dim)
"""
import torch
import torch.nn as nn
import math
from typing import Optional

from config.settings import CFG_MODEL


class GaitLSTM(nn.Module):
    """
    LSTM 步态分析模型

    用双向 LSTM 捕捉步态的时间依赖关系
    适合数据量较小的场景（如 UP-Fall）

    Args:
        input_dim: 输入维度（num_kpts × 坐标维度）
        hidden_dim: LSTM 隐藏层维度
        num_layers: LSTM 层数
        dropout: Dropout 比率
        bidirectional: 是否双向
    """

    def __init__(
        self,
        input_dim: int = CFG_MODEL.GAIT_INPUT_DIM,
        hidden_dim: int = CFG_MODEL.GAIT_HIDDEN_DIM,
        num_layers: int = CFG_MODEL.GAIT_NUM_LAYERS,
        dropout: float = CFG_MODEL.GAIT_DROPOUT,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 注意力池化（替代简单的取最后时刻）
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 1),
            nn.Softmax(dim=1),
        )

        # 输出投影
        self.output_dim = hidden_dim * self.num_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 关键点序列, shape (B, seq_len, input_dim)
        Returns:
            步态特征, shape (B, hidden_dim * num_directions)
        """
        # 输入投影
        x = self.input_proj(x)           # (B, seq_len, hidden_dim)

        # LSTM 编码
        lstm_out, _ = self.lstm(x)        # (B, seq_len, hidden_dim*2)

        # 注意力池化
        attn_weights = self.attention(lstm_out)  # (B, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, hidden_dim*2)

        return context


class GaitTransformer(nn.Module):
    """
    Transformer 步态分析模型

    用自注意力机制捕捉步态中的长程依赖
    适合数据量较大的场景

    Args:
        input_dim: 输入维度
        d_model: Transformer 维度
        nhead: 注意力头数
        num_layers: Transformer 层数
        dropout: Dropout 比率
        max_seq_len: 最大序列长度
    """

    def __init__(
        self,
        input_dim: int = CFG_MODEL.GAIT_INPUT_DIM,
        d_model: int = CFG_MODEL.TRANSFORMER_D_MODEL,
        nhead: int = CFG_MODEL.TRANSFORMER_NHEAD,
        num_layers: int = CFG_MODEL.TRANSFORMER_NUM_LAYERS,
        dropout: float = 0.1,
        max_seq_len: int = 300,
    ):
        super().__init__()

        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = self._create_positional_encoding(
            max_seq_len, d_model
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # CLS token（用于聚合全局信息）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 输出投影
        self.output_dim = d_model

    def _create_positional_encoding(
        self, max_len: int, d_model: int
    ) -> nn.Parameter:
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 关键点序列, shape (B, seq_len, input_dim)
        Returns:
            步态特征, shape (B, d_model)
        """
        B, L, _ = x.shape

        # 输入投影 + 位置编码
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :L, :]

        # 添加 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, L+1, d_model)

        # Transformer 编码
        x = self.transformer(x)

        # 取 CLS token 的输出作为全局特征
        return x[:, 0, :]  # (B, d_model)


class GaitRiskScorer(nn.Module):
    """
    步态风险评分器

    将 GaitLSTM/GaitTransformer 提取的步态特征映射到 [0, 100] 连续风险分数
    用于跌倒风险前置预判（非二分类）

    架构:
        backbone → MLP → sigmoid × 100

    Args:
        backbone: GaitLSTM 或 GaitTransformer
        feat_dim: backbone 输出维度
    """

    def __init__(self, backbone: nn.Module, feat_dim: int):
        super().__init__()
        self.backbone = backbone
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, input_dim)
        Returns:
            risk scores: (B,) 范围 [0, 100]
        """
        feat = self.backbone(x)             # (B, feat_dim)
        score = self.regressor(feat).squeeze(-1) * 100  # (B,) → [0, 100]
        return score


class GaitClassifier(nn.Module):
    """
    步态分类器

    在步态特征之上添加分类头
    用于判断当前步态是正常还是异常

    Args:
        feature_dim: 步态特征维度
        num_classes: 类别数
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, gait_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gait_features: 步态特征, shape (B, feature_dim)
        Returns:
            分类 logits, shape (B, num_classes)
        """
        return self.classifier(gait_features)
