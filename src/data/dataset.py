"""
数据集加载模块

支持的数据集:
- UP-Fall: 多模态（RGB + Depth + IR + IMU），17人，5类活动 + 6类跌倒
- Le2i: RGB 室内场景，4个场景，跌倒 + ADL
- 通用 FallDetectionDataset: 自定义数据集

数据格式约定:
- 每个样本是一个时间窗口（SEQUENCE_LENGTH 帧）
- 每帧包含: 关键点坐标 (17×3) + 可选的额外特征
- 标签: 0=正常(ADL), 1=跌倒
"""
import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from config.settings import CFG_DATA, CFG_PATHS


def infer_ntu_action_key(
    action_name: str = "",
    action_id: Optional[int] = None,
    label: int = 0,
) -> str:
    """
    将 NTU 样本元信息映射到统一动作键。

    优先使用 action_id，因为处理后的目录名可能是 ADL_10000 这类
    非语义编号，直接解析目录名会把大量 ADL 样本错误压缩成同一个默认分数。
    """
    action_name = action_name or ""

    if label == 1 or "Fall" in action_name or "fall" in action_name:
        if action_id is not None:
            if 43 <= action_id <= 50:
                return f"A{action_id}"
            if 42 <= action_id <= 49:
                return f"A{action_id + 1}"
        return "Fall"

    if action_id is not None:
        if 1 <= action_id <= 12:
            return f"A{action_id}"
        if 0 <= action_id <= 11:
            return f"A{action_id + 1}"

    if "ADL" in action_name:
        try:
            act_num = int(action_name.split("_")[-1])
            if 1 <= act_num <= 12:
                return f"A{act_num}"
        except (ValueError, IndexError):
            pass

    return "ADL"


def infer_ntu_risk_score(
    action_name: str,
    label: int,
    risk_mapping: Dict[str, float],
    action_id: Optional[int] = None,
) -> float:
    """根据动作信息返回可复用的 NTU 风险代理标签。"""
    action_key = infer_ntu_action_key(
        action_name=action_name,
        action_id=action_id,
        label=label,
    )
    return float(risk_mapping.get(action_key, risk_mapping.get("ADL", 15)))


class FallDetectionDataset(Dataset):
    """
    通用跌倒检测数据集基类

    数据格式:
        samples: List[Dict]，每个 dict 包含:
            - "keypoints": np.ndarray, shape (seq_len, num_kpts, 3)
            - "label": int (0=正常, 1=跌倒)
            - "metadata": dict (可选，包含场景、被试等信息)

    Args:
        samples: 样本列表
        transform: 数据增强/变换
        sequence_length: 时间窗口长度
    """

    # NTU 动作 → 风险分数映射（确定性，非随机）
    # NTU Action IDs: A1-A12 = ADL, A43-A50 = Fall
    ACTION_RISK_SCORES = {
        # ADL 动作（低风险 5-25）
        "A1": 10,   # drink water
        "A2": 8,    # eat meal
        "A3": 5,    # brush teeth
        "A4": 12,   # brush hair
        "A5": 15,   # drop
        "A6": 18,   # pickup
        "A7": 8,    # throw
        "A8": 10,   # sit down
        "A9": 10,   # stand up
        "A10": 5,   # applaud
        "A11": 12,  # read
        "A12": 8,   # write
        "ADL": 15,  # 默认 ADL
        # Fall 动作（高风险 75-95）
        "A43": 85,  # fall forward
        "A44": 85,  # fall backward
        "A45": 85,  # fall left
        "A46": 85,  # fall right
        "A47": 90,  # fall syncope
        "A48": 85,  # fall stumble
        "A49": 80,  # fall chair
        "A50": 85,  # fall pick up
        "Fall": 85, # 默认 Fall
    }

    def __init__(
        self,
        samples: List[Dict],
        transform=None,
        sequence_length: int = CFG_DATA.SEQUENCE_LENGTH,
        risk_mode: bool = False,
        multimodal: bool = False,
    ):
        self.samples = samples
        self.transform = transform
        self.sequence_length = sequence_length
        self.risk_mode = risk_mode
        self.multimodal = multimodal
        self.scene_dim = 18  # 12 env + 6 traj

        # 风险模式下预计算风险分数（确定性，基于动作语义）
        if (self.risk_mode or self.multimodal) and len(self.samples) > 0:
            self._precompute_risk_scores()

    def __len__(self) -> int:
        return len(self.samples)

    def _precompute_risk_scores(self):
        """为每个样本预计算确定性风险分数（基于动作语义）"""
        for sample in self.samples:
            metadata = sample.get("metadata", {})
            action = metadata.get("action", "")
            action_id = metadata.get("action_id")
            label = sample.get("label", 0)
            sample["risk_score"] = infer_ntu_risk_score(
                action_name=action,
                action_id=action_id,
                label=label,
                risk_mapping=self.ACTION_RISK_SCORES,
            )

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        keypoints = sample["keypoints"]  # (seq_len, num_kpts, 3)

        # 确保序列长度一致
        keypoints = self._pad_or_truncate(keypoints)

        # 应用变换
        if self.transform:
            keypoints = self.transform(keypoints)

        # 转换为 tensor
        # 展平: (seq_len, num_kpts, 3) → (seq_len, num_kpts*3)
        seq_len, num_kpts, _ = keypoints.shape
        keypoints_flat = keypoints.reshape(seq_len, -1)

        x = torch.tensor(keypoints_flat, dtype=torch.float32)

        if self.multimodal:
            # 多模态模式：返回 (skeleton, scene_feat, risk_score, label)
            scene_feat = torch.zeros(self.scene_dim, dtype=torch.float32)
            y = torch.tensor(sample["risk_score"], dtype=torch.float32)
            label = torch.tensor(sample["label"], dtype=torch.long)
            return x, scene_feat, y, label
        elif self.risk_mode:
            # 回归模式：返回 (skeleton, risk_score, label)
            y = torch.tensor(sample["risk_score"], dtype=torch.float32)
            label = torch.tensor(sample["label"], dtype=torch.long)
            return x, y, label
        else:
            # 分类模式：返回二分类标签
            y = torch.tensor(sample["label"], dtype=torch.long)
            return x, y

    def _pad_or_truncate(self, keypoints: np.ndarray) -> np.ndarray:
        """将序列填充或截断到固定长度"""
        seq_len = keypoints.shape[0]
        if seq_len >= self.sequence_length:
            return keypoints[:self.sequence_length]
        else:
            # 零填充
            pad_len = self.sequence_length - seq_len
            padding = np.zeros((pad_len, *keypoints.shape[1:]))
            return np.concatenate([keypoints, padding], axis=0)


class UPFallDataset(FallDetectionDataset):
    """
    UP-Fall 数据集加载器

    数据集特点:
    - 17名被试（年轻人模拟）
    - 5类日常活动 + 6类跌倒方式
    - 多模态: RGB + Depth + IR + 加速度计
    - 30 FPS

    目录结构（预期）:
        UPFALL_DIR/
        ├── Subject_01/
        │   ├── ADL_01/
        │   │   ├── rgb/
        │   │   ├── depth/
        │   │   └── keypoints.json
        │   └── Fall_01/
        │       ├── rgb/
        │       ├── depth/
        │       └── keypoints.json
        └── ...

    Args:
        root_dir: UP-Fall 数据集根目录
        split: "train" | "val" | "test"
        use_modalities: 使用的模态列表 ["keypoints", "depth", "imu"]
        transform: 数据变换
    """

    ACTIVITIES = [
        "Walking", "Sitting", "Standing", "Lying",
        "Picking_up_object",
        "Fall_forward", "Fall_backward", "Fall_left",
        "Fall_right", "Fall_syncope", "Fall_walk",
    ]

    def __init__(
        self,
        root_dir: str = str(CFG_DATA.UPFALL_DIR),
        split: str = "train",
        use_modalities: List[str] = ["keypoints"],
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.use_modalities = use_modalities

        # 加载样本列表
        samples = self._load_samples()
        super().__init__(samples, transform)

    def _load_samples(self) -> List[Dict]:
        """扫描目录，构建样本列表"""
        samples = []
        if not self.root_dir.exists():
            print(f"[WARNING] UP-Fall 数据目录不存在: {self.root_dir}")
            return samples

        # 遍历每个被试
        for subject_dir in sorted(self.root_dir.glob("Subject_*")):
            # 简单的 train/val/test 划分: 按被试编号
            subject_id = int(subject_dir.name.split("_")[-1])
            if self.split == "train" and subject_id > 12:
                continue
            elif self.split == "val" and not (13 <= subject_id <= 15):
                continue
            elif self.split == "test" and subject_id < 16:
                continue

            # 遍历每个活动/跌倒
            for activity_dir in sorted(subject_dir.iterdir()):
                if not activity_dir.is_dir():
                    continue

                # 确定标签: 跌倒类=1, ADL=0
                activity_name = activity_dir.name
                is_fall = "Fall" in activity_name
                label = 1 if is_fall else 0

                # 加载关键点
                kpt_file = activity_dir / "keypoints.json"
                if kpt_file.exists():
                    with open(kpt_file) as f:
                        kpt_data = json.load(f)
                    keypoints = np.array(kpt_data["keypoints"])  # (N, 17, 3)
                else:
                    continue

                samples.append({
                    "keypoints": keypoints,
                    "label": label,
                    "metadata": {
                        "subject_id": subject_id,
                        "activity": activity_name,
                    },
                })

        return samples


class Le2iDataset(FallDetectionDataset):
    """
    Le2i Fall Detection 数据集加载器

    数据集特点:
    - 4个室内场景（客厅、厨房等）
    - RGB 视频
    - 25 FPS
    - 年轻人模拟跌倒

    Args:
        root_dir: Le2i 数据集根目录
        split: "train" | "val" | "test"
        transform: 数据变换
    """

    def __init__(
        self,
        root_dir: str = str(CFG_DATA.LE2I_DIR),
        split: str = "train",
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split

        samples = self._load_samples()
        super().__init__(samples, transform)

    def _load_samples(self) -> List[Dict]:
        """加载 Le2i 样本"""
        samples = []
        if not self.root_dir.exists():
            print(f"[WARNING] Le2i 数据目录不存在: {self.root_dir}")
            return samples

        # 遍历场景目录
        for scene_dir in sorted(self.root_dir.iterdir()):
            if not scene_dir.is_dir():
                continue

            for video_dir in sorted(scene_dir.iterdir()):
                if not video_dir.is_dir():
                    continue

                # 检查是否有关键点文件
                kpt_file = video_dir / "keypoints.json"
                if not kpt_file.exists():
                    continue

                with open(kpt_file) as f:
                    kpt_data = json.load(f)

                keypoints = np.array(kpt_data["keypoints"])
                label = kpt_data.get("label", 0)

                samples.append({
                    "keypoints": keypoints,
                    "label": label,
                    "metadata": {"scene": scene_dir.name, "video": video_dir.name},
                })

        return samples


class RiskScoreDataset(Dataset):
    """
    风险评分专用数据集

    输入: 多模态特征序列
    输出: 连续风险评分 (0-100)

    用于训练回归模型而非分类模型
    """

    def __init__(
        self,
        features: np.ndarray,
        risk_scores: np.ndarray,
        sequence_length: int = CFG_DATA.SEQUENCE_LENGTH,
    ):
        """
        Args:
            features: 特征数组, shape (N, feature_dim)
            risk_scores: 风险评分数组, shape (N,) 范围 [0, 100]
            sequence_length: 时间窗口长度
        """
        self.features = features
        self.risk_scores = risk_scores
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.sequence_length]
        y = self.risk_scores[idx + self.sequence_length - 1]  # 预测窗口末尾的风险

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y
