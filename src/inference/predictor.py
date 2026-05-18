"""
预测器模块

封装单次预测逻辑:
输入一帧/一段序列 → 输出风险评分

支持两种模式:
1. LSTM 模式: 原始关键点 → LSTM → 风险分数
2. 特征工程模式: 关键点 → GaitFeatureExtractor → MLP/GBDT → 风险分数
"""
import torch
import numpy as np
import pickle
from typing import Optional, Dict, Tuple
from pathlib import Path

from config.settings import CFG_MODEL, CFG_RISK, CFG_TRAIN
from src.models.risk_scoring import RiskScoreCalibrator, get_risk_level
from src.features.gait_features import GaitFeatureExtractor
from src.models.feature_mlp import FeatureMLP


class FallRiskPredictor:
    """
    跌倒风险预测器

    端到端的预测接口:
    输入: 关键点序列 (T, 17, 3) 或 特征向量 (16,)
    输出: 风险评分 + 风险等级 + 响应策略

    支持:
    - LSTM 模式: 从 checkpoint 加载 GaitRiskScorer
    - 特征工程模式: 从 checkpoint 加载 MLP/GBDT
    - 通用模式: 接受任意模型

    Args:
        checkpoint_path: 模型权重路径
        device: 推理设备
        user_id: 用户 ID（用于个性化基线）
        model_type: "auto" | "lstm" | "feature_mlp" | "feature_gbr"
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = CFG_TRAIN.DEVICE,
        user_id: Optional[str] = None,
        model_type: str = "auto",
    ):
        self.device = device
        self.user_id = user_id
        self.model = None
        self.model_type = model_type
        self._scaler = None  # 默认无 scaler
        self.feature_extractor = GaitFeatureExtractor(fps=30.0)

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_model(checkpoint_path)

        # 校准器
        self.calibrator = RiskScoreCalibrator()

    def _load_model(self, path: str):
        """自动检测模型类型并加载"""
        path = Path(path)

        if path.suffix == ".pt":
            # PyTorch checkpoint — 可能是 LSTM 或 MLP
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)

            # 检测是否是 FeatureMLP
            has_backbone = any("backbone" in k for k in state.keys())
            has_regressor = any("regressor" in k for k in state.keys())
            has_gait_backbone = any("gait_backbone" in k for k in state.keys())

            if has_gait_backbone:
                # MultiModalRiskScorer
                self.model_type = "multimodal"
                # TODO: 需要重建模型架构再加载
                print(f"[Predictor] 检测到多模态模型: {path}")
            elif has_backbone and has_regressor:
                # GaitRiskScorer (LSTM)
                self.model_type = "lstm"
                # TODO: 需要重建模型架构再加载
                print(f"[Predictor] 检测到 LSTM 模型: {path}")
            else:
                # 可能是 FeatureMLP
                self.model_type = "feature_mlp"
                self.model = FeatureMLP()
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                print(f"[Predictor] 加载特征 MLP: {path}")

        elif path.suffix == ".pkl":
            # GBR 模型
            self.model_type = "feature_gbr"
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            print(f"[Predictor] 加载 GBR 模型: {path}")

        # 加载 scaler（如果有）
        scaler_path = path.parent / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            print(f"[Predictor] 加载 scaler: {scaler_path}")
        else:
            self._scaler = None

    def set_model(self, model):
        """直接设置模型（不从文件加载）"""
        self.model = model
        if isinstance(model, torch.nn.Module):
            model.to(self.device)
            model.eval()

    @torch.no_grad()
    def predict_from_keypoints(self, skeleton: np.ndarray) -> Dict:
        """
        从关键点序列预测风险

        Args:
            skeleton: (T, 17, 3) — T帧, 17关键点, (x, y, conf)

        Returns:
            预测结果字典
        """
        if self.model is None:
            return {"score": 0.0, "risk_level": "low", "error": "no model"}

        # 提取特征
        features = self.feature_extractor.extract_vector(skeleton)  # (16,)
        features = np.nan_to_num(features, nan=0.0)

        # 标准化（如果有 scaler）
        if self._scaler is not None:
            features = self._scaler.transform(features.reshape(1, -1)).flatten()

        # 预测
        if isinstance(self.model, torch.nn.Module):
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            raw_score = self.model(x).item()
        else:
            # GBR
            raw_score = float(self.model.predict(features.reshape(1, -1))[0])

        raw_score = float(np.clip(raw_score, 0, 100))

        # 校准
        calibrated_score = self.calibrator.calibrate(
            raw_score, features, self.user_id
        )

        risk_level = get_risk_level(calibrated_score)

        return {
            "score": round(calibrated_score, 2),
            "risk_level": risk_level,
            "raw_score": round(raw_score, 2),
            "features": features.tolist(),
            "calibrated": self.user_id is not None,
        }

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Dict:
        """
        从特征向量预测风险（兼容旧接口）

        Args:
            features: 特征向量 (feature_dim,)

        Returns:
            预测结果字典
        """
        if self.model is None:
            return {"score": 0.0, "risk_level": "low", "error": "no model"}

        if isinstance(self.model, torch.nn.Module):
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            raw_score = self.model(x).item()
        else:
            raw_score = float(self.model.predict(features.reshape(1, -1))[0])

        raw_score = float(np.clip(raw_score, 0, 100))

        calibrated_score = self.calibrator.calibrate(
            raw_score, features, self.user_id
        )

        risk_level = get_risk_level(calibrated_score)

        return {
            "score": round(calibrated_score, 2),
            "risk_level": risk_level,
            "raw_score": round(raw_score, 2),
            "calibrated": self.user_id is not None,
        }

    def predict_sequence(self, feature_sequence: np.ndarray) -> Dict:
        """
        序列预测（取最后一个时间步的评分）

        Args:
            feature_sequence: 特征序列, shape (seq_len, feature_dim)

        Returns:
            同 predict
        """
        return self.predict(feature_sequence[-1])

    def update_baseline(self, normal_features: np.ndarray):
        """
        更新用户正常状态基线

        Args:
            normal_features: 正常状态特征, shape (N, feature_dim)
        """
        if self.user_id:
            self.calibrator.update_baseline(self.user_id, normal_features)
            print(f"[Predictor] 已更新用户 {self.user_id} 的基线")
