"""
可视化工具

在视频帧上绘制:
- 人体关键点和骨架
- 风险评分和等级
- 警报状态
"""
import cv2
import numpy as np
from typing import Optional, Dict, Tuple, List

from src.models.pose_estimation import SKELETON, COCO_KEYPOINTS

# 风险等级颜色 (BGR)
RISK_COLORS = {
    "low": (0, 255, 0),       # 绿色
    "medium": (0, 255, 255),   # 黄色
    "high": (0, 165, 255),     # 橙色
    "critical": (0, 0, 255),   # 红色
}


def draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """
    在帧上绘制关键点和骨架

    Args:
        frame: BGR 图像
        keypoints: shape (17, 3) — x, y, confidence
        confidence_threshold: 置信度阈值

    Returns:
        绘制后的图像
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    for i, (x, y, conf) in enumerate(keypoints):
        if conf < confidence_threshold:
            continue
        # 关键点
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

    # 骨架
    for idx1, idx2 in SKELETON:
        x1, y1, c1 = keypoints[idx1]
        x2, y2, c2 = keypoints[idx2]
        if c1 < confidence_threshold or c2 < confidence_threshold:
            continue
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return vis


def draw_risk_overlay(
    frame: np.ndarray,
    result: Dict,
) -> np.ndarray:
    """
    在帧上叠加风险评分信息

    Args:
        frame: BGR 图像
        result: risk_result dict with score, risk_level, bbox

    Returns:
        绘制后的图像
    """
    vis = frame.copy()
    score = result.get("score", 0)
    risk_level = result.get("risk_level", "low")
    bbox = result.get("bbox")
    color = RISK_COLORS.get(risk_level, (255, 255, 255))

    # 绘制人体检测框
    if bbox:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # 风险评分仪表盘
    bar_x, bar_y = 20, 20
    bar_w, bar_h = 200, 30
    # 背景
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    # 填充
    fill_w = int(bar_w * score / 100)
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    # 边框
    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
    # 文字
    text = f"Risk: {score:.0f}/100 [{risk_level.upper()}]"
    cv2.putText(vis, text, (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis
