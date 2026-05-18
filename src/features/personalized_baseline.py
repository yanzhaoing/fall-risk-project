"""
个性化基线模块

核心思想：不跟"标准老人"比，跟"他自己"比。

每位老人的正常状态不同：
- 张大爷正常步速 0.4m/s → 对他来说正常
- 李奶奶正常步速 0.8m/s → 对她来说正常
- 如果张大爷突然变成 0.8m/s → 异常（可能在慌张跑动）
- 如果李奶奶突然变成 0.4m/s → 异常（可能腿脚出了问题）

实现：
1. 采集期：记录老人正常活动数据，计算特征的均值和标准差
2. 实时期：新数据与基线对比，计算偏离度（z-score）
3. 更新期：基线随时间缓慢更新（指数移动平均），适应自然变化

输入: 步态特征向量 (16维) 或任意特征向量
输出: 偏离度分数 (0-100)

用法:
    baseline = PersonalizedBaseline(user_id="zhang_dage")
    baseline.calibrate(normal_features)  # 采集期
    deviation = baseline.compute_deviation(new_features)  # 实时期
    baseline.update(new_features)  # 更新期
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from .gait_features import FEATURE_NAMES as GAIT_FEATURE_NAMES


class PersonalizedBaseline:
    """
    个性化基线

    为每位老人建立正常状态特征分布（均值、方差），
    实时检测当前状态偏离正常基线的程度。

    Args:
        user_id: 用户唯一标识
        feature_names: 特征名称列表（默认使用步态特征的 16 维）
        alpha: 基线更新的指数移动平均系数（越大越快适应变化）
        min_samples: 最少校准样本数（低于此数不计算偏离度）
    """

    def __init__(
        self,
        user_id: str,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05,
        min_samples: int = 10,
    ):
        self.user_id = user_id
        self.feature_names = feature_names or GAIT_FEATURE_NAMES
        self.num_features = len(self.feature_names)
        self.alpha = alpha
        self.min_samples = min_samples

        # 基线统计
        self._mean: Optional[np.ndarray] = None  # (num_features,)
        self._std: Optional[np.ndarray] = None   # (num_features,)
        self._calibration_data: List[np.ndarray] = []  # 校准期原始数据
        self._is_calibrated: bool = False
        self._sample_count: int = 0

    @property
    def is_calibrated(self) -> bool:
        """是否已完成校准"""
        return self._is_calibrated

    @property
    def baseline_mean(self) -> Optional[np.ndarray]:
        """基线均值"""
        return self._mean

    @property
    def baseline_std(self) -> Optional[np.ndarray]:
        """基线标准差"""
        return self._std

    def calibrate(self, features: np.ndarray):
        """
        校准：用正常状态数据建立基线

        可以多次调用，累积数据。调用 compute_deviation 前
        至少需要 min_samples 个样本。

        Args:
            features: 正常状态特征，shape (N, num_features) 或 (num_features,)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 验证维度
        if features.shape[1] != self.num_features:
            raise ValueError(
                f"特征维度不匹配: 期望 {self.num_features}, 得到 {features.shape[1]}"
            )

        self._calibration_data.append(features)
        self._sample_count += features.shape[0]

        # 累积足够样本后计算基线
        if self._sample_count >= self.min_samples:
            all_data = np.concatenate(self._calibration_data, axis=0)
            self._mean = np.mean(all_data, axis=0)
            self._std = np.std(all_data, axis=0) + 1e-6  # 避免除零
            self._is_calibrated = True

    def compute_deviation(self, features: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        计算当前特征与基线的偏离度

        Args:
            features: 当前特征，shape (num_features,)

        Returns:
            (deviation_score, deviation_vector)
            - deviation_score: 综合偏离度分数 (0-100)
            - deviation_vector: 每个特征的偏离度 (num_features,)
        """
        if not self._is_calibrated:
            return 0.0, np.zeros(self.num_features)

        # z-score: 每个特征偏离几个标准差
        z_scores = np.abs(features - self._mean) / self._std

        # 综合偏离度：所有特征 z-score 的加权平均
        # 用均值，不用加权（避免特征重要性假设）
        deviation_score = float(np.mean(z_scores))

        # 映射到 0-100
        # z-score 1.0 = 轻微偏离 → 约 20 分
        # z-score 2.0 = 明显偏离 → 约 40 分
        # z-score 3.0 = 严重偏离 → 约 60 分
        # z-score 5.0+ = 极端偏离 → 100 分
        deviation_score = float(np.clip(deviation_score * 20, 0, 100))

        return deviation_score, z_scores

    def compute_deviation_per_feature(
        self, features: np.ndarray
    ) -> Dict[str, float]:
        """
        计算每个特征的偏离度（用于分析/可视化）

        Args:
            features: 当前特征，shape (num_features,)

        Returns:
            特征名 → z-score 的字典
        """
        if not self._is_calibrated:
            return {name: 0.0 for name in self.feature_names}

        z_scores = np.abs(features - self._mean) / self._std
        return {
            name: float(z)
            for name, z in zip(self.feature_names, z_scores)
        }

    def update(self, features: np.ndarray):
        """
        更新基线（指数移动平均）

        用于适应老人状态的缓慢变化（如逐渐衰老）。
        不应该在异常状态时调用。

        Args:
            features: 当前特征，shape (num_features,)
        """
        if not self._is_calibrated:
            # 还没校准完，加入校准数据
            self.calibrate(features)
            return

        # 指数移动平均
        self._mean = (1 - self.alpha) * self._mean + self.alpha * features
        # std 的更新更保守（用更大的 alpha）
        new_std = np.abs(features - self._mean)
        self._std = (1 - self.alpha * 0.5) * self._std + (self.alpha * 0.5) * new_std
        self._std = np.maximum(self._std, 1e-6)  # 避免除零

    def get_risk_adjustment(
        self,
        raw_risk_score: float,
        features: np.ndarray,
        deviation_weight: float = 0.6,
    ) -> float:
        """
        基于偏离度调整风险分数

        Args:
            raw_risk_score: 模型输出的原始风险分数 (0-100)
            features: 当前特征
            deviation_weight: 偏离度的权重 (0-1)

        Returns:
            调整后的风险分数 (0-100)
        """
        deviation_score, _ = self.compute_deviation(features)

        # 融合：偏离度权重越大，个性化越强
        adjusted = (
            (1 - deviation_weight) * raw_risk_score
            + deviation_weight * deviation_score
        )

        return float(np.clip(adjusted, 0, 100))

    def save(self, path: str):
        """
        保存基线到文件

        Args:
            path: 保存路径（JSON 格式）
        """
        data = {
            "user_id": self.user_id,
            "feature_names": self.feature_names,
            "alpha": self.alpha,
            "min_samples": self.min_samples,
            "is_calibrated": self._is_calibrated,
            "sample_count": self._sample_count,
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PersonalizedBaseline":
        """
        从文件加载基线

        Args:
            path: 文件路径

        Returns:
            PersonalizedBaseline 实例
        """
        with open(path) as f:
            data = json.load(f)

        baseline = cls(
            user_id=data["user_id"],
            feature_names=data.get("feature_names"),
            alpha=data.get("alpha", 0.05),
            min_samples=data.get("min_samples", 10),
        )

        baseline._is_calibrated = data.get("is_calibrated", False)
        baseline._sample_count = data.get("sample_count", 0)

        if data.get("mean") is not None:
            baseline._mean = np.array(data["mean"], dtype=np.float32)
        if data.get("std") is not None:
            baseline._std = np.array(data["std"], dtype=np.float32)

        return baseline


class BaselineManager:
    """
    基线管理器

    管理多个用户的个性化基线。支持：
    - 添加/删除用户
    - 加载/保存所有基线
    - 批量计算偏离度

    Args:
        baselines_dir: 基线文件存储目录
    """

    def __init__(self, baselines_dir: str = "baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.baselines: Dict[str, PersonalizedBaseline] = {}

    def add_user(
        self,
        user_id: str,
        feature_names: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> PersonalizedBaseline:
        """
        添加新用户

        Args:
            user_id: 用户 ID
            feature_names: 特征名称
            alpha: 更新系数

        Returns:
            新建的 PersonalizedBaseline 实例
        """
        baseline = PersonalizedBaseline(
            user_id=user_id,
            feature_names=feature_names,
            alpha=alpha,
        )
        self.baselines[user_id] = baseline
        return baseline

    def get_user(self, user_id: str) -> Optional[PersonalizedBaseline]:
        """获取用户的基线"""
        return self.baselines.get(user_id)

    def remove_user(self, user_id: str):
        """删除用户基线"""
        self.baselines.pop(user_id, None)

    def calibrate_user(self, user_id: str, features: np.ndarray):
        """校准指定用户的基线"""
        if user_id in self.baselines:
            self.baselines[user_id].calibrate(features)

    def compute_all_deviations(
        self, user_id: str, features: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """计算指定用户的偏离度"""
        if user_id not in self.baselines:
            return 0.0, np.zeros(len(GAIT_FEATURE_NAMES))
        return self.baselines[user_id].compute_deviation(features)

    def save_all(self):
        """保存所有用户基线"""
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        for user_id, baseline in self.baselines.items():
            path = self.baselines_dir / f"{user_id}.json"
            baseline.save(str(path))

    def load_all(self):
        """加载目录下所有用户基线"""
        if not self.baselines_dir.exists():
            return

        for path in self.baselines_dir.glob("*.json"):
            baseline = PersonalizedBaseline.load(str(path))
            self.baselines[baseline.user_id] = baseline

    def list_users(self) -> List[str]:
        """列出所有用户 ID"""
        return list(self.baselines.keys())


# ── 测试 ──
if __name__ == "__main__":
    np.random.seed(42)

    print("=== 个性化基线测试 ===\n")

    # 模拟张大爷的正常步态
    zhang_normal = np.random.randn(50, 16).astype(np.float32) * 0.5 + 2.0
    # 模拟李奶奶的正常步态（步速更快，摆动更小）
    li_normal = np.random.randn(50, 16).astype(np.float32) * 0.3 + 4.0

    # 创建基线
    zhang = PersonalizedBaseline("zhang_dage")
    li = PersonalizedBaseline("li_nainai")

    # 校准
    zhang.calibrate(zhang_normal)
    li.calibrate(li_normal)

    print(f"张大爷已校准: {zhang.is_calibrated}, 样本数: {zhang._sample_count}")
    print(f"李奶奶已校准: {li.is_calibrated}, 样本数: {li._sample_count}")

    # 测试正常状态
    zhang_test_normal = np.random.randn(16).astype(np.float32) * 0.5 + 2.0
    score, _ = zhang.compute_deviation(zhang_test_normal)
    print(f"\n张大爷正常测试: 偏离度 = {score:.2f}")

    # 测试异常状态（步速突变）
    zhang_test_abnormal = zhang_test_normal.copy()
    zhang_test_abnormal[0] = 8.0  # 步速突然变很快
    score, vec = zhang.compute_deviation(zhang_test_abnormal)
    print(f"张大爷异常测试（步速突变）: 偏离度 = {score:.2f}")
    deviations = zhang.compute_deviation_per_feature(zhang_test_abnormal)
    print(f"  步速偏离: {deviations['mean_hip_speed']:.2f} 个标准差")

    # 测试交叉（用张大爷的数据测李奶奶的基线）
    score, _ = li.compute_deviation(zhang_test_normal)
    print(f"\n用张大爷正常数据测李奶奶基线: 偏离度 = {score:.2f}")

    # 测试风险调整
    raw_score = 30.0  # 模型认为低风险
    adjusted = zhang.get_risk_adjustment(raw_score, zhang_test_abnormal)
    print(f"\n原始风险: {raw_score:.1f}, 偏离度调整后: {adjusted:.1f}")

    # 测试保存/加载
    zhang.save("/tmp/test_baseline.json")
    zhang_loaded = PersonalizedBaseline.load("/tmp/test_baseline.json")
    score2, _ = zhang_loaded.compute_deviation(zhang_test_abnormal)
    print(f"保存/加载后偏离度: {score2:.2f} (应与上面相同)")

    # 测试 BaselineManager
    print("\n=== BaselineManager 测试 ===")
    manager = BaselineManager("/tmp/test_baselines")
    manager.add_user("zhang_dage")
    manager.calibrate_user("zhang_dage", zhang_normal)
    manager.add_user("li_nainai")
    manager.calibrate_user("li_nainai", li_normal)

    print(f"用户列表: {manager.list_users()}")
    score, _ = manager.compute_all_deviations("zhang_dage", zhang_test_abnormal)
    print(f"张大爷偏离度: {score:.2f}")

    manager.save_all()
    manager2 = BaselineManager("/tmp/test_baselines")
    manager2.load_all()
    print(f"加载后用户列表: {manager2.list_users()}")

    print("\n✅ 个性化基线模块验证通过")
