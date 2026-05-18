"""
事件处理模块

处理风险预警事件，根据风险等级触发不同响应:
- 低风险: 日志记录
- 中风险: 日志 + 记录异常
- 高风险: 通知家属
- 极高风险: 紧急响应
"""
import json
import time
import logging
from typing import Dict, Optional, Callable, List
from pathlib import Path
from datetime import datetime

from config.settings import CFG_PATHS


class EventHandler:
    """
    预警事件处理器

    Args:
        log_dir: 事件日志目录
        notification_callback: 通知回调函数
    """

    def __init__(
        self,
        log_dir: str = str(CFG_PATHS.LOGS_DIR),
        notification_callback: Optional[Callable] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.notification_callback = notification_callback

        # 事件历史
        self.events: List[Dict] = []

        # 冷却（避免重复报警）
        self._last_alert_time = 0
        self._alert_cooldown = 30  # 秒

    def handle(self, risk_result: Dict):
        """
        处理风险评估结果

        Args:
            risk_result: 包含 score, risk_level, bbox 等
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "score": risk_result.get("score", 0),
            "risk_level": risk_result.get("risk_level", "low"),
            "bbox": risk_result.get("bbox"),
        }

        self.events.append(event)
        self._log_event(event)

        risk_level = event["risk_level"]

        # 高风险以上触发通知（带冷却）
        if risk_level in ("high", "critical"):
            now = time.time()
            if now - self._last_alert_time > self._alert_cooldown:
                self._send_alert(event)
                self._last_alert_time = now

    def _log_event(self, event: Dict):
        """记录事件到日志"""
        log_file = self.log_dir / f"events_{datetime.now():%Y%m%d}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _send_alert(self, event: Dict):
        """发送警报通知"""
        risk_level = event["risk_level"]
        score = event["score"]

        alert_msg = (
            f"⚠️ 跌倒风险警报\n"
            f"风险等级: {risk_level}\n"
            f"风险评分: {score}/100\n"
            f"时间: {event['timestamp']}"
        )

        print(f"\n{'!' * 40}")
        print(alert_msg)
        print(f"{'!' * 40}\n")

        if self.notification_callback:
            self.notification_callback(alert_msg, event)

    def get_recent_events(self, n: int = 10) -> List[Dict]:
        """获取最近 n 条事件"""
        return self.events[-n:]

    def get_statistics(self) -> Dict:
        """获取事件统计"""
        if not self.events:
            return {"total": 0}

        scores = [e["score"] for e in self.events]
        levels = [e["risk_level"] for e in self.events]

        return {
            "total": len(self.events),
            "mean_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "level_counts": {
                level: levels.count(level)
                for level in set(levels)
            },
        }
