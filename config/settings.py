"""
全局配置 — 所有路径、超参数、常量集中管理

使用方式:
    from config.settings import CFG
    print(CFG.DATA_DIR)
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# ─── 项目根目录 ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class PathConfig:
    """路径配置"""
    # 数据
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DATASETS_DIR: Path = PROJECT_ROOT / "data" / "datasets"
    RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    # 模型
    CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
    # 日志
    LOGS_DIR: Path = PROJECT_ROOT / "logs"


@dataclass
class DatasetConfig:
    """数据集配置"""
    # UP-Fall
    UPFALL_DIR: Path = PROJECT_ROOT / "data" / "datasets" / "UP-Fall"
    UPFALL_FPS: int = 30
    UPFALL_FRAME_SIZE: Tuple[int, int] = (640, 480)

    # Le2i
    LE2I_DIR: Path = PROJECT_ROOT / "data" / "datasets" / "Le2i"
    LE2I_FPS: int = 25

    # NTU RGB+D
    NTU_DIR: Path = PROJECT_ROOT / "data" / "datasets" / "NTU-RGBD"
    NTU_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed" / "ntu_coco"

    # UR Fall
    URFALL_DIR: Path = PROJECT_ROOT / "data" / "datasets" / "UR-Fall"

    # ETRI-Activity3D
    ETRI_DIR: Path = PROJECT_ROOT / "data" / "datasets" / "ETRI-Activity3D"

    # 通用
    NUM_KEYPOINTS: int = 17           # COCO 关键点数
    SEQUENCE_LENGTH: int = 30         # 时间窗口长度（帧数）
    STRIDE: int = 5                   # 滑动窗口步长
    NUM_CLASSES: int = 2              # 跌倒/非跌倒（二分类基线）


@dataclass
class ModelConfig:
    """模型配置"""
    # 姿态估计
    POSE_MODEL: str = "mediapipe"     # "mediapipe" | "hrnet"
    POSE_CONFIDENCE: float = 0.5

    # 人体检测
    DET_MODEL: str = "yolov8n.pt"     # YOLOv8 模型
    DET_CONFIDENCE: float = 0.5
    DET_IOU: float = 0.45

    # 步态分析 (LSTM)
    GAIT_INPUT_DIM: int = 51          # 17关键点 × 3 (x, y, confidence)
    GAIT_HIDDEN_DIM: int = 128
    GAIT_NUM_LAYERS: int = 2
    GAIT_DROPOUT: float = 0.3

    # Transformer 替代方案
    TRANSFORMER_D_MODEL: int = 128
    TRANSFORMER_NHEAD: int = 8
    TRANSFORMER_NUM_LAYERS: int = 4

    # ST-GCN
    STGCN_BASE_CHANNELS: int = 64
    STGCN_NUM_STAGES: int = 3
    STGCN_TEMPORAL_KERNEL: int = 9
    STGCN_DROPOUT: float = 0.3

    # 风险评分
    RISK_INPUT_DIM: int = 256         # 融合特征维度
    RISK_HIDDEN_DIM: int = 128

    # 多模态融合
    FUSION_STRATEGY: str = "gated"   # "concat" | "attention" | "gated"


@dataclass
class TrainConfig:
    """训练配置"""
    # 基础
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4
    EPOCHS: int = 100
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    SCHEDULER: str = "cosine"         # "cosine" | "step" | "plateau"

    # 类别不平衡处理
    USE_FOCAL_LOSS: bool = True
    FOCAL_ALPHA: float = 0.75
    FOCAL_GAMMA: float = 2.0
    CLASS_WEIGHTS: List[float] = field(default_factory=lambda: [1.0, 5.0])

    # 早停
    EARLY_STOP_PATIENCE: int = 15
    MIN_DELTA: float = 0.001

    # 设备
    DEVICE: str = "cuda:0"
    MIXED_PRECISION: bool = True

    # 日志
    LOG_INTERVAL: int = 10
    SAVE_INTERVAL: int = 5
    USE_WANDB: bool = False


@dataclass
class RiskConfig:
    """风险评分配置"""
    # 预警阈值
    LOW_RISK_THRESHOLD: int = 30      # 0-30: 低风险（绿色）
    MEDIUM_RISK_THRESHOLD: int = 60   # 31-60: 中风险（黄色）
    HIGH_RISK_THRESHOLD: int = 80     # 61-80: 高风险（橙色）
    # 81-100: 极高风险（红色）

    # 个性化基线
    BASELINE_WINDOW: int = 300        # 基线计算窗口（帧数，约10秒@30fps）
    DEVIATION_WEIGHT: float = 0.6     # 偏离度权重

    # 时间平滑
    SMOOTHING_WINDOW: int = 15        # 评分平滑窗口
    TEMPORAL_WEIGHT: float = 0.3      # 时序权重


@dataclass
class EZVIZConfig:
    """萤石API配置"""
    APP_KEY: str = os.getenv("EZVIZ_APP_KEY", "")
    APP_SECRET: str = os.getenv("EZVIZ_APP_SECRET", "")
    ACCESS_TOKEN: str = os.getenv("EZVIZ_ACCESS_TOKEN", "")
    API_BASE: str = "https://open.ys7.com"
    # 人体检测API
    BODY_DETECT_URL: str = f"{API_BASE}/api/lapp/ai/body/detect"
    # 姿态分析API
    POSE_ANALYSIS_URL: str = f"{API_BASE}/api/lapp/ai/pose/analysis"


# ─── 全局配置实例 ─────────────────────────────────────────────
CFG_PATHS = PathConfig()
CFG_DATA = DatasetConfig()
CFG_MODEL = ModelConfig()
CFG_TRAIN = TrainConfig()
CFG_RISK = RiskConfig()
CFG_EZVIZ = EZVIZConfig()
