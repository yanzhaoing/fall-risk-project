"""
姿态估计模块

支持两种后端:
1. MediaPipe — 轻量、实时、适合边缘设备
2. HRNet — 高精度、适合离线分析

输出: COCO 格式 17 关键点 (x, y, confidence)
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

from config.settings import CFG_MODEL

# COCO 17 关键点定义
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 骨骼连接（用于可视化）
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # 头部
    (5, 6),                                     # 肩膀
    (5, 7), (7, 9),                             # 左臂
    (6, 8), (8, 10),                            # 右臂
    (5, 11), (6, 12),                           # 躯干
    (11, 12),                                   # 髋部
    (11, 13), (13, 15),                         # 左腿
    (12, 14), (14, 16),                         # 右腿
]


class PoseEstimator(ABC):
    """姿态估计器基类"""

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        估计单帧姿态

        Args:
            frame: BGR 图像, shape (H, W, 3)

        Returns:
            关键点数组, shape (17, 3) — [x, y, confidence]
            如果未检测到人体则返回 None
        """
        pass

    def estimate_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[Optional[np.ndarray]]:
        """批量估计"""
        return [self.estimate(frame) for frame in frames]


class MediaPipePose(PoseEstimator):
    """
    MediaPipe 姿态估计器

    特点:
    - 实时性能好（CPU 也能跑）
    - 33 个关键点（需要映射到 COCO 17 点）
    - 支持多人（取置信度最高的）

    Args:
        confidence: 检测置信度阈值
        tracking_confidence: 追踪置信度阈值
    """

    # MediaPipe → COCO 关键点映射
    MP_TO_COCO = {
        0: 0,     # nose
        2: 1,     # left_eye (inner)
        5: 2,     # right_eye (inner)
        7: 3,     # left_ear
        8: 4,     # right_ear
        11: 5,    # left_shoulder
        12: 6,    # right_shoulder
        13: 7,    # left_elbow
        14: 8,    # right_elbow
        15: 9,    # left_wrist
        16: 10,   # right_wrist
        23: 11,   # left_hip
        24: 12,   # right_hip
        25: 13,   # left_knee
        26: 14,   # right_knee
        27: 15,   # left_ankle
        28: 16,   # right_ankle
    }

    def __init__(
        self,
        confidence: float = CFG_MODEL.POSE_CONFIDENCE,
        tracking_confidence: float = 0.5,
    ):
        self.confidence = confidence
        self.tracking_confidence = tracking_confidence
        self.mp_pose = None

    def _init_mediapipe(self):
        """延迟初始化 MediaPipe"""
        if self.mp_pose is None:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=self.confidence,
                min_tracking_confidence=self.tracking_confidence,
            )

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """使用 MediaPipe 估计姿态"""
        self._init_mediapipe()
        import mediapipe as mp

        rgb = frame[:, :, ::-1]  # BGR → RGB
        results = self.mp_pose.process(rgb)

        if not results.pose_landmarks:
            return None

        # 提取关键点并映射到 COCO 格式
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        coco_kpts = np.zeros((17, 3), dtype=np.float32)
        for mp_idx, coco_idx in self.MP_TO_COCO.items():
            lm = landmarks[mp_idx]
            coco_kpts[coco_idx] = [
                lm.x * w,   # 像素坐标 x
                lm.y * h,   # 像素坐标 y
                lm.visibility,  # 置信度
            ]

        return coco_kpts


class HRNetPose(PoseEstimator):
    """
    HRNet 姿态估计器

    特点:
    - 高精度
    - 需要 GPU
    - 需要预训练模型权重

    Args:
        model_path: HRNet 权重路径
        confidence: 检测置信度阈值
        device: 推理设备
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = CFG_MODEL.POSE_CONFIDENCE,
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None

    def load(self):
        """加载 HRNet 模型"""
        if self.model is None:
            # TODO: 实现 HRNet 模型加载
            # 需要安装 mmpose 或 hrnet 库
            print("[HRNetPose] 模型加载待实现，请使用 MediaPipePose")
            pass

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """使用 HRNet 估计姿态"""
        self.load()
        if self.model is None:
            raise NotImplementedError("HRNet 模型尚未实现，请使用 MediaPipePose")
        # TODO: 实现 HRNet 推理
        return None


def create_pose_estimator(
    backend: str = CFG_MODEL.POSE_MODEL,
    **kwargs,
) -> PoseEstimator:
    """
    姿态估计器工厂函数

    Args:
        backend: "mediapipe" | "hrnet"
        **kwargs: 传给具体估计器的参数

    Returns:
        PoseEstimator 实例
    """
    estimators = {
        "mediapipe": MediaPipePose,
        "hrnet": HRNetPose,
    }

    if backend not in estimators:
        raise ValueError(f"不支持的姿态估计后端: {backend}")

    return estimators[backend](**kwargs)
