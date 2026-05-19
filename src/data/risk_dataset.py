"""
风险标签与真实 scene/context 特征增强版数据集。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from config.settings import CFG_DATA
from src.data.dataset import FallDetectionDataset
from src.features.context_features import (
    SCENE_FEATURE_DIM,
    extract_scene_features,
    estimate_pre_fall_risk,
)


class RiskAwareFallDataset(FallDetectionDataset):
    """为回归和多模态任务提供更合理的伪标签与非空上下文特征。"""

    def __init__(
        self,
        samples: List[Dict],
        transform=None,
        sequence_length: int = CFG_DATA.SEQUENCE_LENGTH,
        risk_mode: bool = False,
        multimodal: bool = False,
    ):
        super().__init__(
            samples=samples,
            transform=transform,
            sequence_length=sequence_length,
            risk_mode=False,
            multimodal=False,
        )
        self.risk_mode = risk_mode
        self.multimodal = multimodal
        self.scene_dim = SCENE_FEATURE_DIM

        if self.risk_mode or self.multimodal:
            self._precompute_targets()

    def _precompute_targets(self):
        for sample in self.samples:
            keypoints = np.asarray(sample["keypoints"], dtype=np.float32)
            keypoints = self._pad_or_truncate(keypoints)
            metadata = sample.get("metadata", {})
            scene_features = extract_scene_features(keypoints, metadata=metadata)
            risk_score, breakdown = estimate_pre_fall_risk(
                keypoints,
                int(sample.get("label", 0)),
                metadata=metadata,
                scene_features=scene_features,
            )
            sample["scene_features"] = scene_features.astype(np.float32)
            sample["risk_score"] = float(risk_score)
            sample.setdefault("metadata", {})
            sample["metadata"]["risk_breakdown"] = breakdown

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        keypoints = self._pad_or_truncate(sample["keypoints"])

        if self.transform:
            keypoints = self.transform(keypoints)

        seq_len, _, _ = keypoints.shape
        x = torch.tensor(keypoints.reshape(seq_len, -1), dtype=torch.float32)

        if self.multimodal:
            scene_feat = torch.tensor(sample["scene_features"], dtype=torch.float32)
            y = torch.tensor(sample["risk_score"], dtype=torch.float32)
            label = torch.tensor(sample["label"], dtype=torch.long)
            return x, scene_feat, y, label

        if self.risk_mode:
            y = torch.tensor(sample["risk_score"], dtype=torch.float32)
            label = torch.tensor(sample["label"], dtype=torch.long)
            return x, y, label

        y = torch.tensor(sample["label"], dtype=torch.long)
        return x, y
