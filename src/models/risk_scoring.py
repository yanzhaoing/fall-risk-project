"""
风险评分模型

核心创新点: 输出连续性风险评分（0-100），而非二分类

将多模态特征融合后，通过回归网络预测风险分数
同时支持个性化基线和时间平滑

输入: 融合特征向量 (batch, input_dim)
输出: 风险评分 (batch,) 范围 [0, 100]
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from collections import deque

from config.settings import CFG_RISK, CFG_MODEL


class RiskScoringModel(nn.Module):
    """
    连续性风险评分网络

    将多模态特征映射到 [0, 100] 的风险分数
    使用回归而非分类，保留细粒度信息

    架构:
        输入特征 → MLP → 风险分数 (sigmoid × 100)

    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
    """

    def __init__(
        self,
        input_dim: int = CFG_MODEL.RISK_INPUT_DIM,
        hidden_dim: int = CFG_MODEL.RISK_HIDDEN_DIM,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 融合特征, shape (B, input_dim)
        Returns:
            风险评分, shape (B,) 范围 [0, 100]
        """
        logit = self.network(x).squeeze(-1)  # (B,)
        score = torch.sigmoid(logit) * 100   # 映射到 [0, 100]
        return score


class RiskScoreCalibrator:
    """
    风险评分校准器

    功能:
    1. 个性化基线 — 为每位老人建立正常状态基线
    2. 偏离度计算 — 检测当前状态偏离正常基线的程度
    3. 时间平滑 — 避免评分剧烈波动

    Args:
        baseline_window: 基线计算窗口大小
        deviation_weight: 偏离度在最终评分中的权重
        smoothing_window: 时间平滑窗口
        temporal_weight: 时序权重
    """

    def __init__(
        self,
        baseline_window: int = CFG_RISK.BASELINE_WINDOW,
        deviation_weight: float = CFG_RISK.DEVIATION_WEIGHT,
        smoothing_window: int = CFG_RISK.SMOOTHING_WINDOW,
        temporal_weight: float = CFG_RISK.TEMPORAL_WEIGHT,
    ):
        self.baseline_window = baseline_window
        self.deviation_weight = deviation_weight
        self.smoothing_window = smoothing_window
        self.temporal_weight = temporal_weight

        # 存储每个用户的基线
        self.baselines = {}  # user_id → feature_mean, feature_std
        # 评分历史（用于时间平滑）
        self.score_history = deque(maxlen=smoothing_window)

    def update_baseline(
        self,
        user_id: str,
        features: np.ndarray,
    ):
        """
        更新用户基线

        Args:
            user_id: 用户 ID
            features: 正常状态下的特征序列, shape (N, feature_dim)
        """
        self.baselines[user_id] = {
            "mean": features.mean(axis=0),
            "std": features.std(axis=0) + 1e-6,
        }

    def calibrate(
        self,
        raw_score: float,
        features: np.ndarray,
        user_id: Optional[str] = None,
    ) -> float:
        """
        校准风险评分

        1. 如果有用户基线，计算偏离度并融合
        2. 应用时间平滑

        Args:
            raw_score: 模型输出的原始评分 [0, 100]
            features: 当前特征向量
            user_id: 用户 ID（可选）

        Returns:
            校准后的评分 [0, 100]
        """
        final_score = raw_score

        # 个性化基线校准
        if user_id and user_id in self.baselines:
            baseline = self.baselines[user_id]
            deviation = np.abs(features - baseline["mean"]) / baseline["std"]
            deviation_score = np.clip(deviation.mean() * 20, 0, 100)

            # 融合原始评分和偏离度
            final_score = (
                (1 - self.deviation_weight) * raw_score
                + self.deviation_weight * deviation_score
            )

        # 时间平滑
        self.score_history.append(final_score)
        if len(self.score_history) > 1:
            weights = np.array([
                self.temporal_weight ** i
                for i in range(len(self.score_history))
            ])
            weights = weights[::-1]  # 最新的权重最大
            weights /= weights.sum()
            final_score = np.average(
                list(self.score_history), weights=weights
            )

        return float(np.clip(final_score, 0, 100))


def get_risk_level(score: float) -> str:
    """
    根据风险分数返回风险等级

    Args:
        score: 风险评分 [0, 100]

    Returns:
        风险等级字符串
    """
    if score <= CFG_RISK.LOW_RISK_THRESHOLD:
        return "low"       # 低风险 — 绿色
    elif score <= CFG_RISK.MEDIUM_RISK_THRESHOLD:
        return "medium"    # 中风险 — 黄色
    elif score <= CFG_RISK.HIGH_RISK_THRESHOLD:
        return "high"      # 高风险 — 橙色
    else:
        return "critical"  # 极高风险 — 红色


def get_response_strategy(risk_level: str) -> dict:
    """
    根据风险等级返回响应策略

    Args:
        risk_level: "low" | "medium" | "high" | "critical"

    Returns:
        响应策略字典
    """
    strategies = {
        "low": {
            "action": "continue_monitoring",
            "alert": False,
            "description": "继续监控，无需干预",
        },
        "medium": {
            "action": "increased_monitoring",
            "alert": False,
            "description": "提高监控频率，记录异常",
        },
        "high": {
            "action": "alert_family",
            "alert": True,
            "notify": ["family"],
            "description": "通知家属，准备干预",
        },
        "critical": {
            "action": "family_emergency_decision",
            "alert": True,
            "notify": ["family", "app"],
            "description": "紧急通知家属，由家属决定是否叫急救",
        },
    }
    return strategies.get(risk_level, strategies["low"])
