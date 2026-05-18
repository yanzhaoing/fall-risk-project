"""模型测试"""
import pytest
import torch
from src.models.gait_analysis import GaitLSTM, GaitTransformer
from src.models.risk_scoring import RiskScoringModel
from src.models.fusion import MultiModalFusion


class TestGaitLSTM:
    """GaitLSTM 测试"""

    def test_forward(self):
        """测试前向传播"""
        model = GaitLSTM(input_dim=51, hidden_dim=64, num_layers=2)
        x = torch.randn(4, 30, 51)  # batch=4, seq=30, dim=51
        out = model(x)
        assert out.shape == (4, 128)  # 64*2 (bidirectional)

    def test_output_dim(self):
        """测试输出维度属性"""
        model = GaitLSTM(hidden_dim=64)
        assert model.output_dim == 128


class TestGaitTransformer:
    """GaitTransformer 测试"""

    def test_forward(self):
        """测试前向传播"""
        model = GaitTransformer(input_dim=51, d_model=64, nhead=4, num_layers=2)
        x = torch.randn(4, 30, 51)
        out = model(x)
        assert out.shape == (4, 64)


class TestRiskScoringModel:
    """RiskScoringModel 测试"""

    def test_output_range(self):
        """测试输出范围在 [0, 100]"""
        model = RiskScoringModel(input_dim=128, hidden_dim=64)
        x = torch.randn(8, 128)
        scores = model(x)
        assert scores.shape == (8,)
        assert (scores >= 0).all()
        assert (scores <= 100).all()


class TestMultiModalFusion:
    """MultiModalFusion 测试"""

    def test_concat_fusion(self):
        """测试拼接融合"""
        model = MultiModalFusion(
            feature_dims=[64, 128], output_dim=256, strategy="concat"
        )
        f1 = torch.randn(4, 64)
        f2 = torch.randn(4, 128)
        out = model([f1, f2])
        assert out.shape == (4, 256)

    def test_attention_fusion(self):
        """测试注意力融合"""
        model = MultiModalFusion(
            feature_dims=[64, 128], output_dim=256, strategy="attention"
        )
        f1 = torch.randn(4, 64)
        f2 = torch.randn(4, 128)
        out = model([f1, f2])
        assert out.shape == (4, 256)

    def test_gated_fusion(self):
        """测试门控融合"""
        model = MultiModalFusion(
            feature_dims=[64, 128], output_dim=256, strategy="gated"
        )
        f1 = torch.randn(4, 64)
        f2 = torch.randn(4, 128)
        out = model([f1, f2])
        assert out.shape == (4, 256)
