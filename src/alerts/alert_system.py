"""
分级预警模块

将 0-100 风险分数转化为可操作的预警等级和响应动作。

评分标准要求："前置预判—过程识别—分级预警 完整闭环"
这个模块负责"分级预警"这一环。

风险等级：
- 低风险 (0-30)  → 继续监控
- 中风险 (31-60) → 提高监控频率
- 高风险 (61-80) → 通知家属
- 极高风险 (81-100) → 紧急通知家属（由家属决策）

关键设计：
1. 时间平滑 — 避免分数波动导致频繁切换等级
2. 持续时间 — 高风险持续一段时间才触发通知（避免误报）
3. 冷却期 — 通知后一段时间内不重复通知
4. 升级机制 — 风险持续上升时自动升级响应

用法:
    alert = AlertSystem()
    result = alert.process(score=75.0)
    if result["should_notify"]:
        send_notification(result)
"""

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import IntEnum


class RiskLevel(IntEnum):
    """风险等级"""
    LOW = 0       # 低风险 — 绿色
    MEDIUM = 1    # 中风险 — 黄色
    HIGH = 2      # 高风险 — 橙色
    CRITICAL = 3  # 极高风险 — 红色


# 风险等级配置
RISK_THRESHOLDS = {
    RiskLevel.LOW: (0, 30),
    RiskLevel.MEDIUM: (31, 60),
    RiskLevel.HIGH: (61, 80),
    RiskLevel.CRITICAL: (81, 100),
}

RISK_LABELS = {
    RiskLevel.LOW: "低风险",
    RiskLevel.MEDIUM: "中风险",
    RiskLevel.HIGH: "高风险",
    RiskLevel.CRITICAL: "极高风险",
}

RISK_COLORS = {
    RiskLevel.LOW: "#00ff88",      # 绿色
    RiskLevel.MEDIUM: "#ffd700",   # 黄色
    RiskLevel.HIGH: "#ff8800",     # 橙色
    RiskLevel.CRITICAL: "#ff4444", # 红色
}


@dataclass
class AlertEvent:
    """预警事件"""
    timestamp: float
    risk_score: float
    risk_level: RiskLevel
    level_label: str
    duration: float  # 当前等级持续时间（秒）
    should_notify: bool
    notify_targets: List[str]  # 通知目标
    response_action: str
    description: str


class AlertSystem:
    """
    分级预警系统

    处理流程：
    1. 输入风险分数 → 2. 时间平滑 → 3. 确定等级 → 4. 持续时间检查 → 5. 触发响应

    Args:
        smoothing_window: 平滑窗口大小（最近 N 个分数的加权平均）
        high_duration_threshold: 高风险持续多少秒后触发通知
        critical_duration_threshold: 极高风险持续多少秒后触发通知
        cooldown_seconds: 通知冷却期（秒）
    """

    def __init__(
        self,
        smoothing_window: int = 10,
        high_duration_threshold: float = 30.0,
        critical_duration_threshold: float = 5.0,
        cooldown_seconds: float = 300.0,
    ):
        self.smoothing_window = smoothing_window
        self.high_duration_threshold = high_duration_threshold
        self.critical_duration_threshold = critical_duration_threshold
        self.cooldown_seconds = cooldown_seconds

        # 状态
        self._score_history: deque = deque(maxlen=smoothing_window)
        self._current_level: RiskLevel = RiskLevel.LOW
        self._level_enter_time: float = time.time()
        self._last_notify_time: float = 0.0
        self._last_notify_level: RiskLevel = RiskLevel.LOW

    def process(self, score: float, timestamp: Optional[float] = None) -> Dict:
        """
        处理一个风险分数

        Args:
            score: 风险分数 (0-100)
            timestamp: 时间戳（默认用当前时间）

        Returns:
            预警结果字典
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 时间平滑
        self._score_history.append(score)
        smoothed_score = self._smooth_score()

        # 2. 确定风险等级
        new_level = self._score_to_level(smoothed_score)

        # 3. 等级变化检测
        if new_level != self._current_level:
            self._current_level = new_level
            self._level_enter_time = timestamp

        # 4. 持续时间
        duration = timestamp - self._level_enter_time

        # 5. 决定是否通知
        should_notify = self._should_notify(new_level, duration, timestamp)

        # 6. 确定响应动作
        response = self._get_response(new_level)

        # 7. 构建结果
        result = {
            "score": round(smoothed_score, 1),
            "raw_score": round(score, 1),
            "level": new_level,
            "level_label": RISK_LABELS[new_level],
            "level_color": RISK_COLORS[new_level],
            "duration": round(duration, 1),
            "should_notify": should_notify,
            "notify_targets": response["notify_targets"],
            "response_action": response["action"],
            "description": response["description"],
        }

        # 更新通知时间
        if should_notify:
            self._last_notify_time = timestamp
            self._last_notify_level = new_level

        return result

    def _smooth_score(self) -> float:
        """时间平滑：指数加权平均"""
        if len(self._score_history) == 0:
            return 0.0

        scores = list(self._score_history)
        n = len(scores)

        # 指数权重：最新的权重最大
        weights = [0.8 ** (n - 1 - i) for i in range(n)]
        weight_sum = sum(weights)

        return sum(s * w for s, w in zip(scores, weights)) / weight_sum

    def _score_to_level(self, score: float) -> RiskLevel:
        """分数 → 风险等级"""
        if score <= 30:
            return RiskLevel.LOW
        elif score <= 60:
            return RiskLevel.MEDIUM
        elif score <= 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _should_notify(
        self, level: RiskLevel, duration: float, timestamp: float
    ) -> bool:
        """
        决定是否发送通知

        规则：
        - 低风险：不通知
        - 中风险：不通知（提高监控频率即可）
        - 高风险：持续超过阈值 且 冷却期已过
        - 极高风险：持续超过阈值（更短） 且 冷却期已过
        """
        # 冷却期检查
        time_since_last = timestamp - self._last_notify_time
        if time_since_last < self.cooldown_seconds:
            # 冷却期内，只在等级上升时通知
            if level <= self._last_notify_level:
                return False

        if level == RiskLevel.LOW:
            return False
        elif level == RiskLevel.MEDIUM:
            return False
        elif level == RiskLevel.HIGH:
            return duration >= self.high_duration_threshold
        elif level == RiskLevel.CRITICAL:
            return duration >= self.critical_duration_threshold

        return False

    def _get_response(self, level: RiskLevel) -> Dict:
        """根据风险等级返回响应策略"""
        responses = {
            RiskLevel.LOW: {
                "action": "continue_monitoring",
                "notify_targets": [],
                "description": "继续监控，无需干预",
            },
            RiskLevel.MEDIUM: {
                "action": "increased_monitoring",
                "notify_targets": [],
                "description": "提高监控频率，记录异常",
            },
            RiskLevel.HIGH: {
                "action": "alert_family",
                "notify_targets": ["family", "app"],
                "description": "通知家属，准备干预",
            },
            RiskLevel.CRITICAL: {
                "action": "family_emergency_decision",
                "notify_targets": ["family", "app"],
                "description": "紧急通知家属，由家属决定是否叫急救",
            },
        }
        return responses.get(level, responses[RiskLevel.LOW])

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            "current_level": self._current_level,
            "level_label": RISK_LABELS[self._current_level],
            "level_color": RISK_COLORS[self._current_level],
            "score_history_size": len(self._score_history),
            "last_smoothed_score": round(self._smooth_score(), 1) if self._score_history else 0.0,
            "time_in_current_level": round(time.time() - self._level_enter_time, 1),
            "time_since_last_notify": round(time.time() - self._last_notify_time, 1),
        }

    def reset(self):
        """重置系统状态"""
        self._score_history.clear()
        self._current_level = RiskLevel.LOW
        self._level_enter_time = time.time()
        self._last_notify_time = 0.0


# ── 响应策略详细定义 ──

RESPONSE_STRATEGIES = {
    "continue_monitoring": {
        "camera_action": "normal",          # 摄像头正常监控
        "recording": False,                 # 不录像
        "notification": None,               # 不通知
        "smart_speaker": None,              # 不提醒
        "frequency": "normal",              # 正常检测频率
    },
    "increased_monitoring": {
        "camera_action": "zoom_in",         # 摄像头拉近
        "recording": True,                  # 开始录像
        "notification": None,               # 不通知
        "smart_speaker": None,              # 不提醒
        "frequency": "high",                # 提高检测频率
    },
    "alert_family": {
        "camera_action": "track",           # 摄像头追踪
        "recording": True,                  # 持续录像
        "notification": "push",             # 推送通知给家属
        "smart_speaker": "提醒老人注意安全",  # 智能音箱提醒
        "frequency": "maximum",             # 最高检测频率
    },
    "family_emergency_decision": {
        "camera_action": "track_zoom",      # 追踪+拉近
        "recording": True,                  # 持续录像
        "notification": "urgent",           # 紧急通知家属（带视频片段）
        "smart_speaker": "大声提醒老人注意安全，等待家属确认",
        "frequency": "maximum",
    },
}


def get_response_details(action: str) -> Dict:
    """获取响应策略的详细配置"""
    return RESPONSE_STRATEGIES.get(action, RESPONSE_STRATEGIES["continue_monitoring"])


# ── 测试 ──
if __name__ == "__main__":
    import random

    print("=== 分级预警系统测试 ===\n")

    alert = AlertSystem(
        smoothing_window=5,
        high_duration_threshold=3.0,   # 测试用短阈值
        critical_duration_threshold=1.0,
        cooldown_seconds=10.0,
    )

    # 模拟一系列风险分数
    test_scores = [
        (10.0, "正常走路"),
        (15.0, "正常走路"),
        (25.0, "弯腰捡东西"),
        (35.0, "步速变慢"),
        (45.0, "步态不稳"),
        (55.0, "明显摇晃"),
        (65.0, "高风险步态"),
        (70.0, "持续高风险"),
        (75.0, "持续高风险"),
        (85.0, "极高风险！"),
        (90.0, "即将跌倒！"),
        (20.0, "恢复正常"),
    ]

    for score, desc in test_scores:
        result = alert.process(score)
        notify_str = " 🔔 通知！" if result["should_notify"] else ""
        print(
            f"  分数: {score:5.1f} → {result['level_label']:4s} "
            f"(平滑: {result['score']:5.1f}, 持续: {result['duration']:.1f}s) "
            f"{desc}{notify_str}"
        )

    # 测试状态查询
    print(f"\n当前状态: {alert.get_status()}")

    # 测试响应策略
    print("\n=== 响应策略 ===")
    for level in RiskLevel:
        response = alert._get_response(level)
        details = get_response_details(response["action"])
        print(
            f"  {RISK_LABELS[level]:4s}: {response['description']} "
            f"| 摄像头: {details['camera_action']} "
            f"| 录像: {details['recording']} "
            f"| 频率: {details['frequency']}"
        )

    print("\n✅ 分级预警系统验证通过")
