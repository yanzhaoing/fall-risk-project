"""
数据加载器工厂

根据配置创建 train/val/test DataLoader
"""
import platform
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Optional
import numpy as np

from config.settings import CFG_TRAIN, CFG_DATA
from .dataset import FallDetectionDataset

# Windows 上 num_workers=0（spawn 模式下多进程有隐患）
DEFAULT_NUM_WORKERS = 0 if platform.system() == "Windows" else CFG_TRAIN.NUM_WORKERS


def create_dataloaders(
    train_dataset: FallDetectionDataset,
    val_dataset: FallDetectionDataset,
    test_dataset: Optional[FallDetectionDataset] = None,
    batch_size: int = CFG_TRAIN.BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    use_weighted_sampling: bool = True,
) -> Dict[str, DataLoader]:
    """
    创建 train/val/test DataLoader

    Args:
        train_dataset: 训练集
        val_dataset: 验证集
        test_dataset: 测试集（可选）
        batch_size: 批次大小
        num_workers: 数据加载线程数
        use_weighted_sampling: 是否使用加权采样（处理类别不平衡）

    Returns:
        Dict 包含 "train", "val", "test"(可选) 的 DataLoader
    """
    loaders = {}

    # ─── 训练集: 使用加权采样处理类别不平衡 ───────────────────
    if use_weighted_sampling and len(train_dataset) > 0:
        # 计算类别权重
        labels = [s["label"] for s in train_dataset.samples]
        class_counts = np.bincount(labels, minlength=2)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    # ─── 验证集 ──────────────────────────────────────────────
    loaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ─── 测试集（可选）──────────────────────────────────────
    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders
