"""
特征工程 MLP 模型

用于步态特征 (16维) → 风险分数 (0-100) 的映射

参数量: 3,201（比 LSTM 的 682,882 小 200 倍）
训练时间: 秒级
"""
import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    """
    基于特征的风险评分 MLP

    输入: 16 维步态特征
    输出: 0-100 风险分数

    架构:
        16 → 64 (ReLU) → Dropout → 32 (ReLU) → Dropout → 1 (Sigmoid) → ×100
    """

    def __init__(self, input_dim=16, hidden1=64, hidden2=32, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1) * 100
