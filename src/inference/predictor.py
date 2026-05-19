"""
预测器模块

封装单次预测逻辑:
输入一帧/一段序列 → 输出风险评分

支持两种模式:
1. 时序骨架模式: 原始关键点序列 → GaitRiskScorer / MultiModalRiskScorer → 风险分数
2. 特征工程模式: 关键点 → GaitFeatureExtractor → MLP/GBDT → 风险分数
"""
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from config.settings import CFG_MODEL, CFG_TRAIN
from src.features.context_features import SCENE_FEATURE_DIM, extract_scene_features
from src.features.gait_features import GaitFeatureExtractor
from src.models.feature_mlp import FeatureMLP
from src.models.gait_analysis import GaitLSTM, GaitRiskScorer, GaitTransformer
from src.models.multimodal_risk import MultiModalRiskScorer
from src.models.risk_scoring import RiskScoreCalibrator, get_risk_level


class FallRiskPredictor:
    """
    跌倒风险预测器

    端到端的预测接口:
    输入: 关键点序列 (T, 17, 3) 或 特征向量
    输出: 风险评分 + 风险等级 + 响应策略

    支持:
    - 时序骨架回归模型（GaitRiskScorer）
    - 多模态风险模型（MultiModalRiskScorer）
    - 特征工程模型（FeatureMLP / GBR）
    - 通用模式：接受任意已设置模型

    Args:
        checkpoint_path: 模型权重路径
        device: 推理设备
        user_id: 用户 ID（用于个性化基线）
        model_type: "auto" | "sequence" | "multimodal" | "feature_mlp" | "feature_gbr"
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
        self._scaler = None
        self.feature_extractor = GaitFeatureExtractor(fps=30.0)

        if checkpoint_path and Path(checkpoint_path).exists():
            self._load_model(checkpoint_path)

        self.calibrator = RiskScoreCalibrator()

    def _load_model(self, path: str):
        """自动检测模型类型并加载"""
        path = Path(path)

        if path.suffix == ".pt":
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)

            if any(k.startswith("gait_backbone.") for k in state):
                self.model_type = "multimodal"
                self.model = self._build_multimodal_model_from_state(state)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                print(f"[Predictor] 加载多模态风险模型: {path}")
            elif any(k.startswith("backbone.") for k in state) and any(k.startswith("regressor.") for k in state):
                self.model_type = "sequence"
                self.model = self._build_gait_regressor_from_state(state)
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                print(f"[Predictor] 加载时序风险模型: {path}")
            elif any(k.startswith("classifier.") for k in state):
                self.model_type = "unsupported_classification"
                self.model = None
                print(f"[Predictor] 检测到分类 checkpoint，当前预测器仅支持风险评分类 checkpoint: {path}")
            else:
                self.model_type = "feature_mlp"
                self.model = FeatureMLP()
                self.model.load_state_dict(state)
                self.model.to(self.device)
                self.model.eval()
                print(f"[Predictor] 加载特征 MLP: {path}")

        elif path.suffix == ".pkl":
            self.model_type = "feature_gbr"
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            print(f"[Predictor] 加载 GBR 模型: {path}")

        scaler_path = path.parent / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            print(f"[Predictor] 加载 scaler: {scaler_path}")
        else:
            self._scaler = None

    def _build_backbone_from_state(self, state: Dict[str, torch.Tensor], prefix: str):
        """根据 state_dict 自动重建 gait backbone。"""
        if any(k.startswith(f"{prefix}transformer.") for k in state):
            input_dim = state[f"{prefix}input_proj.weight"].shape[1]
            d_model = state[f"{prefix}input_proj.weight"].shape[0]
            num_layers = len({
                int(k.split(".")[3])
                for k in state
                if k.startswith(f"{prefix}transformer.layers.")
            })
            max_seq_len = state[f"{prefix}pos_encoding"].shape[1]
            return GaitTransformer(
                input_dim=input_dim,
                d_model=d_model,
                nhead=CFG_MODEL.TRANSFORMER_NHEAD,
                num_layers=max(num_layers, 1),
                dropout=CFG_MODEL.GAIT_DROPOUT,
                max_seq_len=max_seq_len,
            )

        input_dim = state[f"{prefix}input_proj.0.weight"].shape[1]
        hidden_dim = state[f"{prefix}input_proj.0.weight"].shape[0]
        num_layers = len({
            int(k.split("l")[-1].split("_")[0])
            for k in state
            if k.startswith(f"{prefix}lstm.weight_ih_l") and "reverse" not in k
        })
        bidirectional = any("reverse" in k for k in state if k.startswith(f"{prefix}lstm."))
        return GaitLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=max(num_layers, 1),
            dropout=CFG_MODEL.GAIT_DROPOUT,
            bidirectional=bidirectional,
        )

    def _build_gait_regressor_from_state(self, state: Dict[str, torch.Tensor]) -> GaitRiskScorer:
        backbone = self._build_backbone_from_state(state, prefix="backbone.")
        feat_dim = state["regressor.0.weight"].shape[1]
        return GaitRiskScorer(backbone, feat_dim)

    def _build_multimodal_model_from_state(self, state: Dict[str, torch.Tensor]) -> MultiModalRiskScorer:
        backbone = self._build_backbone_from_state(state, prefix="gait_backbone.")
        scene_dim = state["fusion.projections.1.weight"].shape[1]
        fusion_dim = state["fusion.output_proj.weight"].shape[0]

        if any(k.startswith("fusion.gates.") for k in state):
            strategy = "gated"
        elif any(k.startswith("fusion.attention_net.") for k in state):
            strategy = "attention"
        else:
            strategy = "concat"

        return MultiModalRiskScorer(
            gait_backbone=backbone,
            gait_dim=backbone.output_dim,
            scene_dim=scene_dim,
            fusion_dim=fusion_dim,
            fusion_strategy=strategy,
        )

    def set_model(self, model):
        """直接设置模型（不从文件加载）"""
        self.model = model
        if isinstance(model, torch.nn.Module):
            model.to(self.device)
            model.eval()

    def _predict_raw_score_from_skeleton(self, skeleton: np.ndarray) -> float:
        seq = skeleton.reshape(skeleton.shape[0], -1)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.model_type == "multimodal":
            scene_dim = getattr(self.model, "scene_dim", SCENE_FEATURE_DIM)
            scene_features = extract_scene_features(skeleton)
            if scene_features.shape[0] != scene_dim:
                resized = np.zeros(scene_dim, dtype=np.float32)
                copy_dim = min(scene_dim, scene_features.shape[0])
                resized[:copy_dim] = scene_features[:copy_dim]
                scene_features = resized
            scene = torch.tensor(scene_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            return float(self.model(x, scene).item())

        return float(self.model(x).item())

    def _predict_raw_score_from_features(self, features: np.ndarray) -> float:
        if isinstance(self.model, torch.nn.Module):
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            return float(self.model(x).item())
        return float(self.model.predict(features.reshape(1, -1))[0])

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

        skeleton = np.asarray(skeleton, dtype=np.float32)
        features = self.feature_extractor.extract_vector(skeleton)
        features = np.nan_to_num(features, nan=0.0)
        scene_features = extract_scene_features(skeleton)

        if self.model_type in {"sequence", "multimodal"}:
            raw_score = self._predict_raw_score_from_skeleton(skeleton)
        else:
            model_features = features
            if self._scaler is not None:
                model_features = self._scaler.transform(model_features.reshape(1, -1)).flatten()
            raw_score = self._predict_raw_score_from_features(model_features)

        raw_score = float(np.clip(raw_score, 0, 100))
        calibrated_score = self.calibrator.calibrate(raw_score, features, self.user_id)
        risk_level = get_risk_level(calibrated_score)

        result = {
            "score": round(calibrated_score, 2),
            "risk_level": risk_level,
            "raw_score": round(raw_score, 2),
            "features": features.tolist(),
            "calibrated": self.user_id is not None,
        }
        if self.model_type == "multimodal":
            result["scene_features"] = scene_features.tolist()
        return result

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> Dict:
        """
        从特征向量预测风险（兼容旧接口）

        Args:
            features: 特征向量

        Returns:
            预测结果字典
        """
        if self.model is None:
            return {"score": 0.0, "risk_level": "low", "error": "no model"}

        if self.model_type in {"sequence", "multimodal"}:
            return {
                "score": 0.0,
                "risk_level": "low",
                "error": "sequence checkpoint requires keypoint sequence input",
            }

        features = np.asarray(features, dtype=np.float32)
        model_features = features
        if self._scaler is not None:
            model_features = self._scaler.transform(model_features.reshape(1, -1)).flatten()

        raw_score = self._predict_raw_score_from_features(model_features)
        raw_score = float(np.clip(raw_score, 0, 100))
        calibrated_score = self.calibrator.calibrate(raw_score, features, self.user_id)
        risk_level = get_risk_level(calibrated_score)

        return {
            "score": round(calibrated_score, 2),
            "risk_level": risk_level,
            "raw_score": round(raw_score, 2),
            "calibrated": self.user_id is not None,
        }

    def predict_sequence(self, feature_sequence: np.ndarray) -> Dict:
        """
        兼容旧接口：
        - 如果输入是骨架序列 (T, 17, 3)，直接调用 `predict_from_keypoints`
        - 否则按特征序列处理，取最后一个时间步
        """
        arr = np.asarray(feature_sequence)
        if arr.ndim == 3 and arr.shape[1] == 17:
            return self.predict_from_keypoints(arr)
        return self.predict(arr[-1])

    def update_baseline(self, normal_features: np.ndarray):
        """更新用户正常状态基线"""
        if self.user_id:
            self.calibrator.update_baseline(self.user_id, normal_features)
            print(f"[Predictor] 已更新用户 {self.user_id} 的基线")
