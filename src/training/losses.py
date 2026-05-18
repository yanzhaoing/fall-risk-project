"""
损失函数模块

包含针对跌倒检测场景的定制损失:
1. Focal Loss — 处理严重的类别不平衡
2. Risk Score Loss — 用于连续风险评分回归
3. Contrastive Loss — 用于学习区分正常/异常步态
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config.settings import CFG_TRAIN


class FocalLoss(nn.Module):
    """
    Focal Loss — 解决类别不平衡问题

    跌倒事件在数据集中占比极低（<5%），
    标准交叉熵会被大量正常样本主导。
    Focal Loss 降低易分类样本的权重，聚焦于难分类样本。

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: 类别权重（正类权重通常更高）
        gamma: 聚焦参数（越大越聚焦于难样本）
        reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        alpha: float = CFG_TRAIN.FOCAL_ALPHA,
        gamma: float = CFG_TRAIN.FOCAL_GAMMA,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: 模型输出, shape (B, C)
            targets: 真实标签, shape (B,)
        Returns:
            标量损失
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(
            targets, num_classes=logits.size(1)
        ).float()

        # 计算 p_t
        p_t = (probs * targets_one_hot).sum(dim=1)  # (B,)

        # alpha 权重
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())

        # Focal Loss
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class RiskScoreLoss(nn.Module):
    """
    风险评分回归损失

    结合 MSE 和排序损失:
    - MSE: 保证评分准确
    - 排序损失: 保证高风险样本的评分高于低风险样本

    Args:
        mse_weight: MSE 损失权重
        ranking_weight: 排序损失权重
        margin: 排序损失的 margin
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        ranking_weight: float = 0.5,
        margin: float = 10.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_scores: 预测风险评分, shape (B,)
            target_scores: 真实风险评分, shape (B,)
            labels: 分类标签（可选，用于排序损失）, shape (B,)
        """
        # MSE 损失
        mse_loss = self.mse(pred_scores, target_scores)

        # 排序损失（可选）
        ranking_loss = torch.tensor(0.0, device=pred_scores.device)
        if labels is not None and self.ranking_weight > 0:
            # 正样本（跌倒）的风险评分应高于负样本（正常）
            pos_mask = labels == 1
            neg_mask = labels == 0

            if pos_mask.any() and neg_mask.any():
                pos_scores = pred_scores[pos_mask].mean()
                neg_scores = pred_scores[neg_mask].mean()
                ranking_loss = F.relu(self.margin - (pos_scores - neg_scores))

        total = self.mse_weight * mse_loss + self.ranking_weight * ranking_loss
        return total


class ContrastiveLoss(nn.Module):
    """
    对比损失 — 学习正常/异常步态的区分性表示

    拉近同类样本、推远异类样本

    Args:
        margin: 正负样本对的间隔
    """

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: 特征向量, shape (B, D)
            labels: 标签, shape (B,)
        """
        # 计算两两距离
        dist_matrix = torch.cdist(features, features, p=2)  # (B, B)

        # 同类对: 距离应小
        same_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        same_mask.fill_diagonal_(0)  # 排除自身

        # 异类对: 距离应大于 margin
        diff_mask = 1 - same_mask
        diff_mask.fill_diagonal_(0)

        # 损失
        pos_loss = (same_mask * dist_matrix).sum() / (same_mask.sum() + 1e-8)
        neg_loss = (diff_mask * F.relu(self.margin - dist_matrix)).sum() / (diff_mask.sum() + 1e-8)

        return pos_loss + neg_loss
