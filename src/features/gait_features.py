"""
步态特征提取模块

从骨架序列中提取步态特征，用于跌倒风险评分。

输入: skeleton sequence (T, 17, 3) — T帧, 17个COCO关键点, (x, y, confidence)
输出: 步态特征向量

特征列表（16维）:
    1. mean_hip_speed      — 平均步速（髋中心移动速度）
    2. std_hip_speed       — 步速稳定性
    3. max_hip_speed       — 最大步速
    4. mean_step_length    — 平均步幅（双脚距离）
    5. std_step_length     — 步幅稳定性
    6. sway_x              — 左右摆动幅度（髋中心x标准差）
    7. sway_y              — 上下摆动幅度（髋中心y标准差）
    8. mean_knee_angle     — 平均膝关节角度
    9. std_knee_angle      — 膝关节角度稳定性
    10. mean_hip_angle     — 平均髋关节角度
    11. std_hip_angle      — 髋关节角度稳定性
    12. knee_angle_rate    — 膝关节角度变化率
    13. hip_angle_rate     — 髋关节角度变化率
    14. step_asymmetry     — 步态对称性（左右腿差异）
    15. gait_regularity    — 步态规律性（自相关峰值）
    16. balance_score      — 平衡得分（重心偏离程度）

参考:
    - Gait speed is the #1 predictor of fall risk in geriatric medicine
    - Step length variability correlates with fall risk
    - Center of mass sway indicates balance impairment
    - Gait cycle regularity reflects neuromuscular control
"""

import numpy as np
from typing import Dict, Optional, Tuple


# COCO 17 关键点索引
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# 特征名称
FEATURE_NAMES = [
    "mean_hip_speed", "std_hip_speed", "max_hip_speed",
    "mean_step_length", "std_step_length",
    "sway_x", "sway_y",
    "mean_knee_angle", "std_knee_angle",
    "mean_hip_angle", "std_hip_angle",
    "knee_angle_rate", "hip_angle_rate",
    "step_asymmetry", "gait_regularity", "balance_score",
]

NUM_FEATURES = len(FEATURE_NAMES)


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """安全归一化，避免除零"""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + eps)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    计算两个向量之间的角度（弧度）
    v1, v2: (..., 2)  — 二维向量
    返回: (...) — 角度（弧度）
    """
    cos = np.sum(v1 * v2, axis=-1)
    cos = np.clip(cos / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8), -1, 1)
    return np.arccos(cos)


def _confidence_weighted_mean(data: np.ndarray, conf: np.ndarray, axis: int = 0) -> np.ndarray:
    """置信度加权平均"""
    conf_sum = np.sum(conf, axis=axis, keepdims=True)
    return np.sum(data * conf, axis=axis) / (conf_sum + 1e-8)


class GaitFeatureExtractor:
    """
    步态特征提取器

    从骨架序列中提取16维步态特征向量。

    用法:
        extractor = GaitFeatureExtractor()
        features = extractor.extract(skeleton)  # (T, 17, 3) → dict of 16 features
        feature_vec = extractor.extract_vector(skeleton)  # → np.ndarray (16,)
    """

    def __init__(self, fps: float = 30.0, use_confidence: bool = True):
        """
        Args:
            fps: 帧率（用于计算速度、加速度等物理量）
            use_confidence: 是否使用置信度加权
        """
        self.fps = fps
        self.use_confidence = use_confidence

    def extract(self, skeleton: np.ndarray) -> Dict[str, float]:
        """
        提取所有步态特征

        Args:
            skeleton: (T, 17, 3) — T帧, 17关键点, (x, y, conf)

        Returns:
            dict: 16个特征的字典
        """
        T = skeleton.shape[0]
        if T < 3:
            return {name: 0.0 for name in FEATURE_NAMES}

        # 提取坐标和置信度
        xy = skeleton[:, :, :2]  # (T, 17, 2)
        conf = skeleton[:, :, 2]  # (T, 17)

        # 关键关节
        hip_center = (xy[:, LEFT_HIP] + xy[:, RIGHT_HIP]) / 2  # (T, 2)
        left_ankle = xy[:, LEFT_ANKLE]  # (T, 2)
        right_ankle = xy[:, RIGHT_ANKLE]  # (T, 2)
        left_knee = xy[:, LEFT_KNEE]  # (T, 2)
        right_knee = xy[:, RIGHT_KNEE]  # (T, 2)
        left_shoulder = xy[:, LEFT_SHOULDER]  # (T, 2)
        right_shoulder = xy[:, RIGHT_SHOULDER]  # (T, 2)

        features = {}

        # ── 1. 步速特征 ──
        hip_vel = np.diff(hip_center, axis=0) * self.fps  # (T-1, 2) 像素/秒
        hip_speed = np.linalg.norm(hip_vel, axis=-1)  # (T-1,)
        features["mean_hip_speed"] = float(np.mean(hip_speed))
        features["std_hip_speed"] = float(np.std(hip_speed))
        features["max_hip_speed"] = float(np.max(hip_speed))

        # ── 2. 步幅特征 ──
        step_length = np.linalg.norm(left_ankle - right_ankle, axis=-1)  # (T,)
        features["mean_step_length"] = float(np.mean(step_length))
        features["std_step_length"] = float(np.std(step_length))

        # ── 3. 重心摆动 ──
        features["sway_x"] = float(np.std(hip_center[:, 0]))
        features["sway_y"] = float(np.std(hip_center[:, 1]))

        # ── 4. 关节角度 ──
        # 膝关节角度: hip-knee-ankle
        left_thigh = left_knee - xy[:, LEFT_HIP]  # (T, 2)
        left_shin = left_ankle - left_knee  # (T, 2)
        left_knee_angles = _angle_between(left_thigh, left_shin)  # (T,)

        right_thigh = right_knee - xy[:, RIGHT_HIP]
        right_shin = right_ankle - right_knee
        right_knee_angles = _angle_between(right_thigh, right_shin)

        knee_angles = (left_knee_angles + right_knee_angles) / 2  # (T,)
        features["mean_knee_angle"] = float(np.mean(knee_angles))
        features["std_knee_angle"] = float(np.std(knee_angles))

        # 髋关节角度: shoulder-hip-knee
        left_torso = left_shoulder - xy[:, LEFT_HIP]
        left_thigh_from_hip = left_knee - xy[:, LEFT_HIP]
        left_hip_angles = _angle_between(left_torso, left_thigh_from_hip)

        right_torso = right_shoulder - xy[:, RIGHT_HIP]
        right_thigh_from_hip = right_knee - xy[:, RIGHT_HIP]
        right_hip_angles = _angle_between(right_torso, right_thigh_from_hip)

        hip_angles = (left_hip_angles + right_hip_angles) / 2  # (T,)
        features["mean_hip_angle"] = float(np.mean(hip_angles))
        features["std_hip_angle"] = float(np.std(hip_angles))

        # ── 5. 角度变化率 ──
        knee_angle_rate = np.abs(np.diff(knee_angles)) * self.fps  # (T-1,) rad/s
        hip_angle_rate = np.abs(np.diff(hip_angles)) * self.fps
        features["knee_angle_rate"] = float(np.mean(knee_angle_rate))
        features["hip_angle_rate"] = float(np.mean(hip_angle_rate))

        # ── 6. 步态对称性 ──
        # 左右腿运动轨迹的差异
        left_leg_motion = np.linalg.norm(np.diff(left_ankle, axis=0), axis=-1)
        right_leg_motion = np.linalg.norm(np.diff(right_ankle, axis=0), axis=-1)
        # 对称性 = 左右差异的均值（越小越对称）
        asymmetry = np.abs(left_leg_motion - right_leg_motion)
        features["step_asymmetry"] = float(np.mean(asymmetry))

        # ── 7. 步态规律性 ──
        # 用髋中心y坐标自相关来衡量步态周期规律性
        hip_y = hip_center[:, 1]
        if T > 6:
            hip_y_centered = hip_y - np.mean(hip_y)
            autocorr = np.correlate(hip_y_centered, hip_y_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # 只取正半部分
            autocorr = autocorr / (autocorr[0] + 1e-8)  # 归一化
            # 找第一个峰值（排除lag=0）
            if len(autocorr) > 3:
                peaks = autocorr[1:-1]
                features["gait_regularity"] = float(np.max(peaks))
            else:
                features["gait_regularity"] = 0.0
        else:
            features["gait_regularity"] = 0.0

        # ── 8. 平衡得分 ──
        # 重心投影到双脚中心的距离
        feet_center = (left_ankle + right_ankle) / 2  # (T, 2)
        balance_dist = np.linalg.norm(hip_center - feet_center, axis=-1)  # (T,)
        # 归一化：用身高（肩到脚的距离）来归一化
        body_height = np.linalg.norm(
            (left_shoulder + right_shoulder) / 2 - (left_ankle + right_ankle) / 2,
            axis=-1
        )
        mean_height = np.mean(body_height)
        if mean_height > 1e-3:
            balance_normalized = balance_dist / mean_height
        else:
            balance_normalized = balance_dist
        features["balance_score"] = float(np.mean(balance_normalized))

        return features

    def extract_vector(self, skeleton: np.ndarray) -> np.ndarray:
        """
        提取特征向量（numpy数组）

        Args:
            skeleton: (T, 17, 3)

        Returns:
            np.ndarray (16,)
        """
        features = self.extract(skeleton)
        return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)

    def extract_batch(self, skeletons: np.ndarray) -> np.ndarray:
        """
        批量提取特征

        Args:
            skeletons: (B, T, 17, 3)

        Returns:
            np.ndarray (B, 16)
        """
        B = skeletons.shape[0]
        features = np.zeros((B, NUM_FEATURES), dtype=np.float32)
        for i in range(B):
            features[i] = self.extract_vector(skeletons[i])
        return features

    def extract_per_frame(self, skeleton: np.ndarray) -> np.ndarray:
        """
        提取逐帧特征（用于LSTM等时序模型）

        每帧提取以下特征:
            - hip_speed (1): 髋中心速度
            - step_length (1): 步幅
            - hip_xy (2): 髋中心位置
            - left_ankle_xy (2): 左踝位置
            - right_ankle_xy (2): 右踝位置
            - knee_angle (1): 膝关节角度
            - hip_angle (1): 髋关节角度
            - balance (1): 平衡指标

        Args:
            skeleton: (T, 17, 3)

        Returns:
            np.ndarray (T-1, 11): 逐帧特征（比输入少1帧因为需要diff）
        """
        T = skeleton.shape[0]
        if T < 2:
            return np.zeros((1, 10), dtype=np.float32)

        xy = skeleton[:, :, :2]  # (T, 17, 2)

        # 关键关节
        hip_center = (xy[:, LEFT_HIP] + xy[:, RIGHT_HIP]) / 2
        left_ankle = xy[:, LEFT_ANKLE]
        right_ankle = xy[:, RIGHT_ANKLE]
        left_knee = xy[:, LEFT_KNEE]
        right_knee = xy[:, RIGHT_KNEE]
        left_shoulder = xy[:, LEFT_SHOULDER]
        right_shoulder = xy[:, RIGHT_SHOULDER]

        # 髋中心速度
        hip_vel = np.diff(hip_center, axis=0) * self.fps
        hip_speed = np.linalg.norm(hip_vel, axis=-1, keepdims=True)  # (T-1, 1)

        # 步幅
        step_length = np.linalg.norm(left_ankle - right_ankle, axis=-1, keepdims=True)  # (T, 1)
        step_length = step_length[1:]  # (T-1, 1)

        # 髋中心位置（归一化）
        hip_xy = hip_center[1:]  # (T-1, 2)

        # 踝部位置
        left_ankle_xy = left_ankle[1:]  # (T-1, 2)
        right_ankle_xy = right_ankle[1:]  # (T-1, 2)

        # 膝关节角度
        left_thigh = left_knee - xy[:, LEFT_HIP]
        left_shin = left_ankle - left_knee
        left_knee_angle = _angle_between(left_thigh, left_shin)
        right_thigh = right_knee - xy[:, RIGHT_HIP]
        right_shin = right_ankle - right_knee
        right_knee_angle = _angle_between(right_thigh, right_shin)
        knee_angle = ((left_knee_angle + right_knee_angle) / 2)[1:, np.newaxis]  # (T-1, 1)

        # 髋关节角度
        left_torso = left_shoulder - xy[:, LEFT_HIP]
        left_thigh_hip = left_knee - xy[:, LEFT_HIP]
        left_hip_angle = _angle_between(left_torso, left_thigh_hip)
        right_torso = right_shoulder - xy[:, RIGHT_HIP]
        right_thigh_hip = right_knee - xy[:, RIGHT_HIP]
        right_hip_angle = _angle_between(right_torso, right_thigh_hip)
        hip_angle = ((left_hip_angle + right_hip_angle) / 2)[1:, np.newaxis]  # (T-1, 1)

        # 平衡指标
        feet_center = (left_ankle + right_ankle) / 2
        balance = np.linalg.norm(hip_center - feet_center, axis=-1, keepdims=True)  # (T, 1)
        balance = balance[1:]  # (T-1, 1)

        # 拼接: (T-1, 10)
        per_frame = np.concatenate([
            hip_speed,       # (T-1, 1)
            step_length,     # (T-1, 1)
            hip_xy,          # (T-1, 2)
            left_ankle_xy,   # (T-1, 2)
            right_ankle_xy,  # (T-1, 2)
            knee_angle,      # (T-1, 1)
            hip_angle,       # (T-1, 1)
            balance,         # (T-1, 1)
        ], axis=-1)

        return per_frame.astype(np.float32)


# ── 测试 ──
if __name__ == "__main__":
    # 用随机数据测试
    np.random.seed(42)
    skeleton = np.random.randn(30, 17, 3).astype(np.float32)
    skeleton[:, :, 2] = 1.0  # 置信度全设为1

    extractor = GaitFeatureExtractor(fps=30.0)

    # 测试 extract
    features = extractor.extract(skeleton)
    print("=== 步态特征 ===")
    for name, val in features.items():
        print(f"  {name}: {val:.4f}")

    # 测试 extract_vector
    vec = extractor.extract_vector(skeleton)
    print(f"\n特征向量 shape: {vec.shape}")
    print(f"特征向量: {vec}")

    # 测试 extract_per_frame
    per_frame = extractor.extract_per_frame(skeleton)
    print(f"\n逐帧特征 shape: {per_frame.shape}")
