"""
数据变换模块

用于数据预处理和后处理的变换函数
"""
import numpy as np
from typing import Tuple


class NormalizeKeypoints:
    """
    关键点归一化

    将关键点坐标归一化到 [0, 1] 范围，基于图像尺寸
    """

    def __init__(self, image_size: Tuple[int, int] = (640, 480)):
        self.w, self.h = image_size

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Args:
            keypoints: (seq_len, num_kpts, 3) — x, y, confidence
        Returns:
            归一化后的关键点
        """
        kpts = keypoints.copy()
        kpts[:, :, 0] /= self.w  # x 归一化
        kpts[:, :, 1] /= self.h  # y 归一化
        return kpts


class CenterNormalize:
    """
    中心归一化

    以人体中心（关键点均值）为原点，用身体尺度归一化
    对不同体型的老人更鲁棒
    """

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Args:
            keypoints: (seq_len, num_kpts, 3)
        Returns:
            中心归一化后的关键点
        """
        kpts = keypoints.copy()
        xy = kpts[:, :, :2]  # (seq_len, num_kpts, 2)

        # 计算每帧的中心（忽略置信度为0的关键点）
        conf = kpts[:, :, 2:3]  # (seq_len, num_kpts, 1)
        mask = (conf > 0).squeeze(-1)  # (seq_len, num_kpts)

        for i in range(len(kpts)):
            valid = mask[i]
            if valid.sum() > 0:
                center = xy[i, valid].mean(axis=0)
                kpts[i, :, :2] -= center

                # 用身体尺度归一化（肩宽或身高）
                valid_xy = xy[i, valid]
                if len(valid_xy) > 1:
                    body_scale = np.ptp(valid_xy, axis=0).max()  # 身体尺度
                    if body_scale > 0:
                        kpts[i, :, :2] /= body_scale

        return kpts


class VelocityTransform:
    """
    计算速度特征

    在原始关键点基础上添加速度（帧间差分）
    """

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Args:
            keypoints: (seq_len, num_kpts, 3)
        Returns:
            带速度特征的关键点: (seq_len, num_kpts, 5) — x, y, conf, vx, vy
        """
        seq_len, num_kpts, _ = keypoints.shape
        result = np.zeros((seq_len, num_kpts, 5))
        result[:, :, :3] = keypoints

        # 计算速度（帧间差分）
        for i in range(1, seq_len):
            result[i, :, 3] = keypoints[i, :, 0] - keypoints[i - 1, :, 0]  # vx
            result[i, :, 4] = keypoints[i, :, 1] - keypoints[i - 1, :, 1]  # vy

        return result
