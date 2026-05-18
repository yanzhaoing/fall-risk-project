"""
端到端推理流水线

将所有模块串联:
视频帧 → 人体检测 → 姿态估计 → 步态特征提取 → 风险评分

支持:
1. 特征工程模式（推荐）: 关键点 → GaitFeatureExtractor → MLP/GBDT
2. LSTM 模式: 关键点 → LSTM → 风险分数

支持实时摄像头和离线视频文件两种模式
"""
import numpy as np
import time
from typing import Optional, Dict, Generator, Callable
from pathlib import Path

from config.settings import CFG_MODEL, CFG_DATA, CFG_RISK
from src.models.backbone import HumanDetector
from src.models.pose_estimation import create_pose_estimator
from src.inference.predictor import FallRiskPredictor
from src.models.risk_scoring import get_risk_level


class InferencePipeline:
    """
    端到端推理流水线

    组装各子模块，提供统一的推理接口

    Args:
        checkpoint_path: 风险评分模型权重路径
        pose_backend: 姿态估计后端
        device: 推理设备
        user_id: 用户 ID
        sequence_length: 时序窗口长度
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        pose_backend: str = "mediapipe",
        device: str = "cuda:0",
        user_id: Optional[str] = None,
        sequence_length: int = CFG_DATA.SEQUENCE_LENGTH,
    ):
        self.device = device
        self.sequence_length = sequence_length

        # 子模块
        self.detector = HumanDetector(device=device)
        self.pose_estimator = create_pose_estimator(pose_backend)
        self.predictor = FallRiskPredictor(
            checkpoint_path=checkpoint_path,
            device=device,
            user_id=user_id,
        )

        # 关键点序列缓冲区
        self.keypoint_buffer: list = []

        # 回调函数
        self.on_risk_update: Optional[Callable] = None

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        处理单帧

        Args:
            frame: BGR 图像

        Returns:
            风险评估结果（如果缓冲区满），否则 None
        """
        # 1. 人体检测
        detections = self.detector.detect(frame)
        if not detections:
            return None

        # 取置信度最高的检测框
        best_det = max(detections, key=lambda d: d["confidence"])
        bbox = best_det["bbox"]

        # 2. 裁剪人体区域再做姿态估计（提高精度，排除其他人）
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 扩大裁剪区域 10%，避免截断
        h, w = frame.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        # 姿态估计（在裁剪区域上）
        keypoints = self.pose_estimator.estimate(cropped)
        if keypoints is None:
            return None

        # 将裁剪区域坐标还原到原图坐标
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1

        # 3. 缓冲关键点序列
        self.keypoint_buffer.append(keypoints)
        if len(self.keypoint_buffer) > self.sequence_length:
            self.keypoint_buffer.pop(0)

        # 4. 缓冲区满时进行风险评估
        if len(self.keypoint_buffer) >= self.sequence_length:
            sequence = np.array(self.keypoint_buffer[-self.sequence_length:])  # (T, 17, 3)

            # 特征提取
            result = self.predictor.predict_from_keypoints(sequence)
            result["bbox"] = bbox
            result["keypoints"] = keypoints.tolist()
            result["buffer_size"] = len(self.keypoint_buffer)

            # 触发回调
            if self.on_risk_update:
                self.on_risk_update(result)

            return result

        return None

    def process_video(
        self,
        source: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Generator[Dict, None, None]:
        """
        处理视频（生成器模式）

        Args:
            source: "camera" 或视频文件路径
            output_path: 输出视频路径（可选）
            max_frames: 最大帧数（可选）

        Yields:
            每帧的处理结果
        """
        import cv2

        if source == "camera":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")

        frame_count = 0
        fps_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.process_frame(frame)
                frame_count += 1

                # 计算 FPS
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_time)
                    fps_time = time.time()
                    print(f"  Frame {frame_count} | FPS: {fps:.1f}")

                yield {
                    "frame": frame,
                    "frame_id": frame_count,
                    "result": result,
                }

                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()

    def reset(self):
        """重置序列缓冲区"""
        self.keypoint_buffer.clear()
