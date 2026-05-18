"""数据集测试"""
import pytest
import numpy as np
import torch
from src.data.dataset import FallDetectionDataset
from src.data.augmentation import KeypointAugmentor


class TestFallDetectionDataset:
    """FallDetectionDataset 测试"""

    def test_create_dataset(self):
        """测试创建数据集"""
        samples = [
            {"keypoints": np.random.randn(30, 17, 3).astype(np.float32), "label": 0},
            {"keypoints": np.random.randn(30, 17, 3).astype(np.float32), "label": 1},
        ]
        dataset = FallDetectionDataset(samples)
        assert len(dataset) == 2

    def test_getitem(self):
        """测试获取样本"""
        samples = [
            {"keypoints": np.random.randn(30, 17, 3).astype(np.float32), "label": 0},
        ]
        dataset = FallDetectionDataset(samples)
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (30, 51)  # 30帧, 17关键点*3
        assert y.item() == 0

    def test_padding(self):
        """测试序列填充"""
        samples = [
            {"keypoints": np.random.randn(10, 17, 3).astype(np.float32), "label": 1},
        ]
        dataset = FallDetectionDataset(samples, sequence_length=30)
        x, y = dataset[0]
        assert x.shape[0] == 30  # 填充到 30 帧


class TestKeypointAugmentor:
    """KeypointAugmentor 测试"""

    def test_augmentation_preserves_shape(self):
        """测试增强不改变形状"""
        augmentor = KeypointAugmentor()
        kpts = np.random.randn(30, 17, 3).astype(np.float32)
        augmented = augmentor(kpts)
        assert augmented.shape == kpts.shape

    def test_augmentation_changes_values(self):
        """测试增强确实改变了数值"""
        augmentor = KeypointAugmentor()
        changed = False
        for seed in range(100):
            np.random.seed(seed)
            kpts = np.random.randn(30, 17, 3).astype(np.float32)
            np.random.seed(seed)
            augmented = augmentor(kpts)
            if not np.allclose(kpts, augmented):
                changed = True
                break
        assert changed, "增强器在100个种子下均未改变数据"
