"""
基于骨架序列提取上下文特征，并构造更像“前置风险”的伪标签。

说明:
- 不再把风险分数简单等同于动作类别。
- 结合运动不稳定性、姿态异常、垂向坍塌趋势和场景先验，
  为每个时间窗生成 0-100 的连续风险分数。
- scene/context features 也可直接供多模态模型使用。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

SCENE_FEATURE_DIM = 18

IDX_LIVING_ROOM = 0
IDX_BEDROOM = 1
IDX_KITCHEN = 2
IDX_BATHROOM = 3
IDX_CENTER_X_STD = 4
IDX_CENTER_Y_STD = 5
IDX_X_RANGE = 6
IDX_Y_RANGE = 7
IDX_MEAN_SPEED = 8
IDX_MAX_SPEED = 9
IDX_MEAN_ACCEL = 10
IDX_VERTICAL_DROP = 11
IDX_HEIGHT_DROP = 12
IDX_ASPECT_MEAN = 13
IDX_ASPECT_STD = 14
IDX_TORSO_LEAN_MEAN = 15
IDX_TORSO_LEAN_STD = 16
IDX_SUPPORT_WIDTH_MEAN = 17

ACTION_PRIORS = {
    "pickup": 8.0,
    "pick_up": 8.0,
    "drop": 6.0,
    "sit": 5.0,
    "stand": 5.0,
    "stair": 8.0,
    "stairs": 8.0,
    "walk": 3.0,
    "transfer": 6.0,
    "turn": 4.0,
    "bath": 8.0,
    "toilet": 8.0,
    "syncope": 10.0,
    "stumble": 10.0,
}


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _safe_stats(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(values.mean()), float(values.std()), float(values.max())


def _mean_joint(coords: np.ndarray, valid: np.ndarray, joint_indices) -> np.ndarray:
    joint_values = coords[:, joint_indices, :]
    joint_valid = valid[:, joint_indices]
    out = np.full((coords.shape[0], 2), np.nan, dtype=np.float32)

    for i in range(coords.shape[0]):
        if joint_valid[i].any():
            out[i] = joint_values[i][joint_valid[i]].mean(axis=0)
    return out


def _infer_scene_one_hot(metadata: Optional[Dict]) -> np.ndarray:
    one_hot = np.zeros(4, dtype=np.float32)
    if not metadata:
        return one_hot

    scene_text = str(metadata.get("scene", "")).lower()
    video_text = str(metadata.get("video", "")).lower()
    merged = f"{scene_text} {video_text}"

    if any(token in merged for token in ["living", "salon", "salle"]):
        one_hot[IDX_LIVING_ROOM] = 1.0
    elif any(token in merged for token in ["bed", "bedroom", "chambre"]):
        one_hot[IDX_BEDROOM] = 1.0
    elif any(token in merged for token in ["kitchen", "cuisine"]):
        one_hot[IDX_KITCHEN] = 1.0
    elif any(token in merged for token in ["bath", "toilet", "wc", "washroom"]):
        one_hot[IDX_BATHROOM] = 1.0

    return one_hot


def extract_scene_features(keypoints: np.ndarray, metadata: Optional[Dict] = None) -> np.ndarray:
    """从骨架序列中提取 18 维上下文/轨迹特征。"""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 3 or keypoints.shape[1] == 0:
        return np.zeros(SCENE_FEATURE_DIM, dtype=np.float32)

    coords = keypoints[..., :2]
    conf = keypoints[..., 2] if keypoints.shape[-1] > 2 else np.ones(keypoints.shape[:2], dtype=np.float32)
    valid = np.isfinite(coords).all(axis=-1) & (conf > 0)

    if not valid.any():
        return np.zeros(SCENE_FEATURE_DIM, dtype=np.float32)

    masked_min = np.where(valid[..., None], coords, np.inf)
    masked_max = np.where(valid[..., None], coords, -np.inf)
    frame_min = masked_min.min(axis=1)
    frame_max = masked_max.max(axis=1)

    invalid_frames = ~valid.any(axis=1)
    frame_min[invalid_frames] = 0.0
    frame_max[invalid_frames] = 0.0

    width = np.maximum(frame_max[:, 0] - frame_min[:, 0], 1e-6)
    height = np.maximum(frame_max[:, 1] - frame_min[:, 1], 1e-6)
    body_scale = float(np.median(height[~invalid_frames])) if (~invalid_frames).any() else 1.0
    body_scale = max(body_scale, 1e-6)

    center = (frame_min + frame_max) / 2.0
    center_x_std = float(np.std(center[:, 0]) / body_scale)
    center_y_std = float(np.std(center[:, 1]) / body_scale)
    x_range = float((center[:, 0].max() - center[:, 0].min()) / body_scale)
    y_range = float((center[:, 1].max() - center[:, 1].min()) / body_scale)

    deltas = np.diff(center, axis=0)
    if deltas.size == 0:
        speed = np.zeros(1, dtype=np.float32)
    else:
        speed = np.linalg.norm(deltas, axis=1) / body_scale
    accel = np.abs(np.diff(speed)) if speed.size > 1 else np.zeros(1, dtype=np.float32)
    mean_speed, _, max_speed = _safe_stats(speed)
    mean_accel, _, _ = _safe_stats(accel)

    vertical_drop = max(float(center[-1, 1] - center[0, 1]) / body_scale, 0.0)
    height_drop = max(float(height[0] - height[-1]) / max(height[0], 1e-6), 0.0)
    aspect_ratio = width / height
    aspect_mean, aspect_std, _ = _safe_stats(aspect_ratio)

    shoulders = _mean_joint(coords, valid, [5, 6])
    hips = _mean_joint(coords, valid, [11, 12])
    torso_vec = shoulders - hips
    torso_lean = np.abs(torso_vec[:, 0]) / (np.abs(torso_vec[:, 1]) + 1e-6)
    torso_lean = np.nan_to_num(torso_lean, nan=0.0, posinf=0.0, neginf=0.0)
    torso_lean_mean, torso_lean_std, _ = _safe_stats(torso_lean)

    ankles = _mean_joint(coords, valid, [15, 16])
    knees = _mean_joint(coords, valid, [13, 14])
    support_source = np.where(np.isfinite(ankles), ankles, knees)
    left_joint = coords[:, 15, :] if coords.shape[1] > 15 else support_source
    right_joint = coords[:, 16, :] if coords.shape[1] > 16 else support_source
    left_valid = valid[:, 15] if valid.shape[1] > 15 else np.isfinite(support_source).all(axis=1)
    right_valid = valid[:, 16] if valid.shape[1] > 16 else np.isfinite(support_source).all(axis=1)
    support_mask = left_valid & right_valid
    support_width = np.zeros(coords.shape[0], dtype=np.float32)
    if support_mask.any():
        support_width[support_mask] = np.abs(left_joint[support_mask, 0] - right_joint[support_mask, 0]) / body_scale
    support_width_mean, _, _ = _safe_stats(support_width)

    scene_one_hot = _infer_scene_one_hot(metadata)
    features = np.array([
        scene_one_hot[0],
        scene_one_hot[1],
        scene_one_hot[2],
        scene_one_hot[3],
        center_x_std,
        center_y_std,
        x_range,
        y_range,
        mean_speed,
        max_speed,
        mean_accel,
        vertical_drop,
        height_drop,
        aspect_mean,
        aspect_std,
        torso_lean_mean,
        torso_lean_std,
        support_width_mean,
    ], dtype=np.float32)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _action_prior(metadata: Optional[Dict]) -> float:
    if not metadata:
        return 0.0
    merged = " ".join(str(metadata.get(key, "")).lower() for key in ["action", "activity", "video", "scene"])
    prior = 0.0
    for token, score in ACTION_PRIORS.items():
        if token in merged:
            prior = max(prior, score)
    return prior


def estimate_pre_fall_risk(
    keypoints: np.ndarray,
    label: int,
    metadata: Optional[Dict] = None,
    scene_features: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """基于骨架动态和上下文特征生成更像“前置风险”的连续伪标签。"""
    if scene_features is None:
        scene_features = extract_scene_features(keypoints, metadata=metadata)

    scene_features = np.asarray(scene_features, dtype=np.float32)
    motion_score = _clip01(
        0.30 * (scene_features[IDX_MEAN_SPEED] / 0.08)
        + 0.30 * (scene_features[IDX_MAX_SPEED] / 0.18)
        + 0.20 * (scene_features[IDX_MEAN_ACCEL] / 0.05)
        + 0.20 * (scene_features[IDX_Y_RANGE] / 0.60)
    )
    posture_score = _clip01(
        0.35 * (scene_features[IDX_TORSO_LEAN_MEAN] / 0.35)
        + 0.25 * (scene_features[IDX_TORSO_LEAN_STD] / 0.20)
        + 0.25 * max(0.55 - scene_features[IDX_SUPPORT_WIDTH_MEAN], 0.0) / 0.35
        + 0.15 * (scene_features[IDX_ASPECT_STD] / 0.25)
    )
    collapse_score = _clip01(
        0.60 * (scene_features[IDX_VERTICAL_DROP] / 0.60)
        + 0.40 * (scene_features[IDX_HEIGHT_DROP] / 0.40)
    )
    context_score = _clip01(
        1.00 * scene_features[IDX_BATHROOM]
        + 0.50 * scene_features[IDX_KITCHEN]
        + 0.25 * scene_features[IDX_BEDROOM]
    )

    prior = _action_prior(metadata) + 10.0 * context_score
    base_score = 12.0 + 22.0 * motion_score + 20.0 * posture_score + 18.0 * collapse_score + prior

    if int(label) == 1:
        boosted = 58.0 + 18.0 * collapse_score + 10.0 * motion_score + 6.0 * posture_score
        score = max(base_score + 12.0, boosted)
        score = min(score, 97.0)
    else:
        score = min(base_score, 68.0)

    score = float(np.clip(score, 5.0, 97.0))
    breakdown = {
        "motion_score": round(motion_score, 4),
        "posture_score": round(posture_score, 4),
        "collapse_score": round(collapse_score, 4),
        "context_score": round(context_score, 4),
        "action_prior": round(float(_action_prior(metadata)), 4),
        "risk_score": round(score, 2),
    }
    return score, breakdown
