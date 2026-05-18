"""
人体检测与特征提取骨干网络

职责:
1. 从视频帧中检测人体（YOLOv8）
2. 裁剪人体区域并提取特征

本模块输出人体检测框和裁剪后的图像区域，
姿态估计由 pose_estimation.py 负责
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from config.settings import CFG_MODEL


class HumanDetector:
    """
    人体检测器（基于 YOLOv8）

    从视频帧中检测人体，返回边界框和置信度

    Args:
        model_name: YOLOv8 模型名（yolov8n/s/m/l/x）
        confidence: 检测置信度阈值
        iou: NMS IoU 阈值
        device: 推理设备
    """

    def __init__(
        self,
        model_name: str = CFG_MODEL.DET_MODEL,
        confidence: float = CFG_MODEL.DET_CONFIDENCE,
        iou: float = CFG_MODEL.DET_IOU,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.model = None

    def load(self):
        """加载模型（延迟加载）"""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"[HumanDetector] 已加载 {self.model_name}")

    def detect(
        self,
        frame: np.ndarray,
        target_class: int = 0,  # 0 = person in COCO
    ) -> List[Dict]:
        """
        检测人体

        Args:
            frame: BGR 图像, shape (H, W, 3)
            target_class: 目标类别（0=person）

        Returns:
            List[Dict], 每个 dict:
                - "bbox": [x1, y1, x2, y2]
                - "confidence": float
                - "class_id": int
        """
        self.load()

        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[target_class],
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "bbox": box.xyxy[0].cpu().numpy().tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    })

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[List[Dict]]:
        """批量检测"""
        return [self.detect(frame) for frame in frames]


class FeatureExtractor(nn.Module):
    """
    CNN 特征提取器

    从人体裁剪区域提取空间特征，供下游任务使用
    支持多种 backbone（ResNet, EfficientNet, ConvNeXt 等）

    Args:
        backbone_name: 骨干网络名
        pretrained: 是否使用预训练权重
        output_dim: 输出特征维度
    """

    BACKBONES = {
        "resnet18": ("resnet18", 512),
        "resnet50": ("resnet50", 2048),
        "efficientnet_b0": ("efficientnet_b0", 1280),
        "convnext_tiny": ("convnext_tiny", 768),
    }

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        output_dim: int = 256,
    ):
        super().__init__()

        if backbone_name not in self.BACKBONES:
            raise ValueError(f"不支持的骨干网络: {backbone_name}")

        timm_name, feat_dim = self.BACKBONES[backbone_name]

        import timm
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0
        )
        self.proj = nn.Linear(feat_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像 tensor, shape (B, 3, H, W)
        Returns:
            特征向量, shape (B, output_dim)
        """
        feat = self.backbone(x)     # (B, feat_dim)
        return self.proj(feat)       # (B, output_dim)


class PoseFeatureExtractor(nn.Module):
    """
    从关键点序列提取高级特征

    使用 1D-CNN 捕捉局部时序模式，
    输出固定维度的特征向量供下游使用

    Args:
        input_dim: 输入维度（num_kpts * 3）
        hidden_dim: 隐藏层维度
        output_dim: 输出特征维度
    """

    def __init__(
        self,
        input_dim: int = 51,  # 17 * 3
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            # 1D 卷积捕捉局部时序模式
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化 → (B, hidden_dim, 1)
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 关键点序列, shape (B, seq_len, input_dim)
        Returns:
            特征向量, shape (B, output_dim)
        """
        # Conv1d 需要 (B, C, L)
        x = x.transpose(1, 2)    # (B, input_dim, seq_len)
        feat = self.encoder(x)    # (B, hidden_dim, 1)
        feat = feat.squeeze(-1)   # (B, hidden_dim)
        return self.proj(feat)    # (B, output_dim)
