"""
数据增强模块

针对人体关键点序列的增强方法：
- 随机旋转
- 随机缩放
- 随机平移
- 随机遮挡（模拟摄像头遮挡）
- 时间扭曲（模拟不同步速）
- 添加噪声
"""
import numpy as np
from typing import Optional


class KeypointAugmentor:
    """
    关键点序列增强器

    所有增强操作直接作用在关键点坐标上（不需要原始图像）

    Args:
        rotation_range: 旋转角度范围（度）
        scale_range: 缩放范围
        translate_range: 平移范围（像素）
        noise_std: 高斯噪声标准差
        occlusion_prob: 随机遮挡概率
        temporal_warp_range: 时间扭曲范围
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        scale_range: tuple = (0.8, 1.2),
        translate_range: float = 20.0,
        noise_std: float = 2.0,
        occlusion_prob: float = 0.1,
        temporal_warp_range: tuple = (0.8, 1.2),
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.noise_std = noise_std
        self.occlusion_prob = occlusion_prob
        self.temporal_warp_range = temporal_warp_range

    def __call__(self, keypoints: np.ndarray) -> np.ndarray:
        """
        对关键点序列应用随机增强

        Args:
            keypoints: shape (seq_len, num_kpts, 3) — x, y, confidence

        Returns:
            增强后的关键点，same shape
        """
        # 随机旋转
        if np.random.random() < 0.5:
            keypoints = self._random_rotate(keypoints)

        # 随机缩放
        if np.random.random() < 0.5:
            keypoints = self._random_scale(keypoints)

        # 随机平移
        if np.random.random() < 0.5:
            keypoints = self._random_translate(keypoints)

        # 添加噪声
        if np.random.random() < 0.5:
            keypoints = self._add_noise(keypoints)

        # 随机遮挡
        if np.random.random() < self.occlusion_prob:
            keypoints = self._random_occlusion(keypoints)

        # 时间扭曲
        if np.random.random() < 0.3:
            keypoints = self._temporal_warp(keypoints)

        return keypoints

    def _random_rotate(self, kpts: np.ndarray) -> np.ndarray:
        """随机旋转（绕图像中心）"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # 取 x, y 坐标
        xy = kpts[:, :, :2].copy()
        # 旋转矩阵
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # 应用旋转
        kpts[:, :, :2] = np.einsum("ijk,lk->ijl", xy, rotation_matrix)

        return kpts

    def _random_scale(self, kpts: np.ndarray) -> np.ndarray:
        """随机缩放"""
        scale = np.random.uniform(*self.scale_range)
        kpts[:, :, :2] *= scale
        return kpts

    def _random_translate(self, kpts: np.ndarray) -> np.ndarray:
        """随机平移"""
        tx = np.random.uniform(-self.translate_range, self.translate_range)
        ty = np.random.uniform(-self.translate_range, self.translate_range)
        kpts[:, :, 0] += tx
        kpts[:, :, 1] += ty
        return kpts

    def _add_noise(self, kpts: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, kpts[:, :, :2].shape)
        kpts[:, :, :2] += noise
        return kpts

    def _random_occlusion(self, kpts: np.ndarray) -> np.ndarray:
        """随机遮挡某些关键点（只设 confidence=0，保留坐标避免产生虚假特征）"""
        num_kpts = kpts.shape[1]
        num_occluded = np.random.randint(1, max(2, num_kpts // 3))
        occluded_indices = np.random.choice(num_kpts, num_occluded, replace=False)
        kpts[:, occluded_indices, 2] = 0.0  # 只设 confidence=0，保留坐标
        return kpts

    def _temporal_warp(self, kpts: np.ndarray) -> np.ndarray:
        """时间扭曲（随机加速/减速）"""
        seq_len = kpts.shape[0]
        if seq_len < 2:
            return kpts

        # 随机选择一个缩放因子
        warp_factor = np.random.uniform(*self.temporal_warp_range)
        new_len = int(seq_len * warp_factor)
        if new_len < 2:
            return kpts

        # 线性插值到新长度
        indices = np.linspace(0, seq_len - 1, new_len)
        indices = np.clip(indices, 0, seq_len - 1).astype(int)
        kpts = kpts[indices]

        # 填充或截断回原始长度
        if len(kpts) > seq_len:
            kpts = kpts[:seq_len]
        elif len(kpts) < seq_len:
            pad = np.zeros((seq_len - len(kpts), *kpts.shape[1:]))
            kpts = np.concatenate([kpts, pad], axis=0)

        return kpts


def get_train_augmentor() -> KeypointAugmentor:
    """获取训练集增强器"""
    return KeypointAugmentor(
        rotation_range=15.0,
        scale_range=(0.8, 1.2),
        translate_range=20.0,
        noise_std=2.0,
        occlusion_prob=0.1,
        temporal_warp_range=(0.8, 1.2),
    )


def get_val_augmentor() -> None:
    """验证集不做增强"""
    return None
