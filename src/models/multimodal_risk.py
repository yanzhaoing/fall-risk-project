"""
多模态风险评分模型

将步态特征（行为模态）和环境/轨迹特征（环境模态）融合，
输出统一的 0-100 风险分数。

架构:
    skeleton → GaitLSTM backbone → gait_feat (256)
    scene features (env+traj) → scene_feat (18)
    [gait_feat, scene_feat] → GatedFusion → fused (128)
    fused → risk_head → sigmoid × 100 → score (0-100)

关键设计:
    - 复用已有的 MultiModalFusion（gated 策略）
    - GaitLSTM backbone 的梯度可以通过融合层反传
    - Scene features 作为额外输入（可以是预计算的）
    - 当 scene features 全零时，模型自动退化为 gait-only

用法:
    model = MultiModalRiskScorer(gait_backbone, gait_dim=256, scene_dim=18)
    score = model(skeleton, scene_features)  # (B,)
"""
import torch
import torch.nn as nn
from typing import Optional

from .gait_analysis import GaitLSTM
from .fusion import MultiModalFusion
from config.settings import CFG_MODEL


class MultiModalRiskScorer(nn.Module):
    """
    多模态风险评分器

    融合步态特征和环境/轨迹特征，输出 0-100 风险分数。

    Args:
        gait_backbone: 步态特征提取器（GaitLSTM 或 GaitTransformer）
        gait_dim: 步态特征维度（backbone.output_dim）
        scene_dim: 环境/轨迹特征维度
        fusion_dim: 融合后特征维度
        fusion_strategy: 融合策略 "gated" | "attention" | "concat"
    """

    def __init__(
        self,
        gait_backbone: nn.Module,
        gait_dim: int = CFG_MODEL.GAIT_HIDDEN_DIM * 2,  # bidirectional
        scene_dim: int = 18,  # 12 env + 6 traj
        fusion_dim: int = 128,
        fusion_strategy: str = "gated",
    ):
        super().__init__()

        self.gait_backbone = gait_backbone
        self.scene_dim = scene_dim

        # 多模态融合层
        self.fusion = MultiModalFusion(
            feature_dims=[gait_dim, scene_dim],
            output_dim=fusion_dim,
            strategy=fusion_strategy,
        )

        # 风险评分头
        self.risk_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.output_dim = fusion_dim

    def forward(
        self,
        skeleton: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            skeleton: 骨架序列 (B, seq_len, input_dim)
            scene_features: 环境/轨迹特征 (B, scene_dim)，可选
                如果为 None，使用零向量（退化为 gait-only）

        Returns:
            风险评分 (B,) 范围 [0, 100]
        """
        B = skeleton.shape[0]

        # 步态特征提取
        gait_feat = self.gait_backbone(skeleton)  # (B, gait_dim)

        # 环境特征（默认零向量）
        if scene_features is None:
            scene_features = torch.zeros(
                B, self.scene_dim, device=skeleton.device
            )

        # 多模态融合
        fused = self.fusion([gait_feat, scene_features])  # (B, fusion_dim)

        # 风险评分
        score = self.risk_head(fused).squeeze(-1) * 100  # (B,) → [0, 100]

        return score

    def get_gait_features(self, skeleton: torch.Tensor) -> torch.Tensor:
        """提取步态特征（用于分析/可视化）"""
        return self.gait_backbone(skeleton)

    def get_fused_features(
        self,
        skeleton: torch.Tensor,
        scene_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """提取融合特征（用于分析/可视化）"""
        B = skeleton.shape[0]
        gait_feat = self.gait_backbone(skeleton)
        if scene_features is None:
            scene_features = torch.zeros(B, self.scene_dim, device=skeleton.device)
        return self.fusion([gait_feat, scene_features])


def build_multimodal_model(
    input_dim: int = CFG_MODEL.GAIT_INPUT_DIM,
    scene_dim: int = 18,
    fusion_dim: int = 128,
    fusion_strategy: str = "gated",
) -> MultiModalRiskScorer:
    """
    构建多模态风险评分模型

    Args:
        input_dim: 骨架输入维度（17 * 3 = 51）
        scene_dim: 环境/轨迹特征维度
        fusion_dim: 融合特征维度
        fusion_strategy: 融合策略

    Returns:
        MultiModalRiskScorer 实例
    """
    # 构建 GaitLSTM backbone
    backbone = GaitLSTM(
        input_dim=input_dim,
        hidden_dim=CFG_MODEL.GAIT_HIDDEN_DIM,
        num_layers=CFG_MODEL.GAIT_NUM_LAYERS,
        dropout=CFG_MODEL.GAIT_DROPOUT,
    )

    # 构建多模态模型
    model = MultiModalRiskScorer(
        gait_backbone=backbone,
        gait_dim=backbone.output_dim,
        scene_dim=scene_dim,
        fusion_dim=fusion_dim,
        fusion_strategy=fusion_strategy,
    )

    return model
