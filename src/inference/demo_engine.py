"""Competition demo scenario engine.

This module powers the local dashboard used for XH-202617 judging demos.  It is
intentionally dependency-light so the web demo can run even before a trained
checkpoint or real EZVIZ device is configured.

The goal is not to replace the model pipeline.  It provides a deterministic
scenario layer for explaining the competition system: pre-fall prediction,
continuous recognition, graded warning, and response-loop evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
from urllib.parse import quote


LEVEL_LABELS = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
    "critical": "极高风险",
}

LEVEL_COLORS = {
    "low": "#16a34a",
    "medium": "#ca8a04",
    "high": "#ea580c",
    "critical": "#dc2626",
}


SCENARIOS: Dict[str, Dict] = {
    "normal": {
        "name": "稳定日常行走",
        "description": "步态稳定，环境风险低，仅持续记录基线。",
        "score": 18,
        "level": "low",
        "modalities": {"behavior": 16, "environment": 12, "baseline": 21},
        "features": {
            "步速波动": "低",
            "躯干摆动": "稳定",
            "重心偏移": "正常",
            "环境遮挡": "无",
        },
    },
    "night_bathroom": {
        "name": "夜间卫生间起身",
        "description": "夜间低照度、起身缓慢，系统提高监测频率。",
        "score": 49,
        "level": "medium",
        "modalities": {"behavior": 44, "environment": 58, "baseline": 45},
        "features": {
            "步速波动": "中等",
            "躯干摆动": "轻微增大",
            "重心偏移": "轻度偏移",
            "环境遮挡": "低照度",
        },
    },
    "stagger": {
        "name": "连续踉跄与扶墙",
        "description": "检测到步幅异常、横向摆动增大，触发家属预警。",
        "score": 73,
        "level": "high",
        "modalities": {"behavior": 78, "environment": 63, "baseline": 71},
        "features": {
            "步速波动": "显著",
            "躯干摆动": "明显增大",
            "重心偏移": "持续偏移",
            "环境遮挡": "局部遮挡",
        },
    },
    "pre_fall": {
        "name": "疑似跌倒前失衡",
        "description": "重心急剧下移且姿态失衡，进入紧急响应闭环。",
        "score": 88,
        "level": "critical",
        "modalities": {"behavior": 92, "environment": 70, "baseline": 89},
        "features": {
            "步速波动": "剧烈",
            "躯干摆动": "异常",
            "重心偏移": "急剧下移",
            "环境遮挡": "风险叠加",
        },
    },
}


@dataclass
class DemoResult:
    """Serializable payload returned by /api/demo/evaluate."""

    scenario: str
    scenario_name: str
    score: int
    level: str
    level_label: str
    should_notify: bool
    response_action: str
    notify_targets: List[str]
    modalities: Dict[str, int]
    features: Dict[str, str]
    visual: Dict[str, str]
    explanation: str


class CompetitionDemoEngine:
    """Deterministic judging-demo engine for risk escalation scenarios."""

    def evaluate(self, scenario: str = "normal") -> DemoResult:
        if scenario not in SCENARIOS:
            scenario = "normal"
        data = SCENARIOS[scenario]
        score = int(data["score"])
        level = data["level"]
        return DemoResult(
            scenario=scenario,
            scenario_name=data["name"],
            score=score,
            level=level,
            level_label=LEVEL_LABELS[level],
            should_notify=level in {"high", "critical"},
            response_action=self._response_action(level),
            notify_targets=self._notify_targets(level),
            modalities=dict(data["modalities"]),
            features=dict(data["features"]),
            visual={"frame_image": self._frame_image(data["name"], score, level)},
            explanation=data["description"],
        )

    def run_sequence(self) -> List[Dict]:
        """Return the full low-to-critical scenario sequence."""
        return [self.evaluate(key).__dict__ for key in SCENARIOS]

    @staticmethod
    def _response_action(level: str) -> str:
        actions = {
            "low": "继续无感监测",
            "medium": "提高采样频率并记录异常",
            "high": "推送家属预警并建议干预",
            "critical": "紧急通知家属并进入响应闭环",
        }
        return actions.get(level, actions["low"])

    @staticmethod
    def _notify_targets(level: str) -> List[str]:
        if level == "critical":
            return ["家属", "养老服务人员", "App 端"]
        if level == "high":
            return ["家属", "App 端"]
        return []

    @staticmethod
    def _frame_image(title: str, score: int, level: str) -> str:
        color = LEVEL_COLORS[level]
        # Simple SVG scene: room outline + skeleton pose + risk badge.
        lean = min(max((score - 30) / 70, 0), 1)
        head_x = 255 + int(lean * 55)
        hip_x = 245 + int(lean * 35)
        foot_x = 235 + int(lean * 20)
        svg = f"""
        <svg xmlns='http://www.w3.org/2000/svg' width='800' height='450' viewBox='0 0 800 450'>
          <defs>
            <linearGradient id='bg' x1='0' x2='1' y1='0' y2='1'>
              <stop offset='0%' stop-color='#f8fafc'/>
              <stop offset='100%' stop-color='#e2e8f0'/>
            </linearGradient>
          </defs>
          <rect width='800' height='450' fill='url(#bg)'/>
          <rect x='70' y='80' width='520' height='300' rx='24' fill='#ffffff' stroke='#cbd5e1' stroke-width='4'/>
          <rect x='100' y='275' width='130' height='72' rx='12' fill='#e2e8f0'/>
          <rect x='455' y='145' width='92' height='185' rx='14' fill='#dbeafe'/>
          <line x1='80' y1='350' x2='580' y2='350' stroke='#94a3b8' stroke-width='4'/>
          <circle cx='{head_x}' cy='160' r='28' fill='none' stroke='#0f172a' stroke-width='8'/>
          <line x1='{head_x}' y1='188' x2='{hip_x}' y2='255' stroke='#0f172a' stroke-width='9' stroke-linecap='round'/>
          <line x1='{hip_x}' y1='220' x2='{hip_x - 55}' y2='245' stroke='#0f172a' stroke-width='8' stroke-linecap='round'/>
          <line x1='{hip_x}' y1='220' x2='{hip_x + 62}' y2='250' stroke='#0f172a' stroke-width='8' stroke-linecap='round'/>
          <line x1='{hip_x}' y1='255' x2='{foot_x - 38}' y2='335' stroke='#0f172a' stroke-width='8' stroke-linecap='round'/>
          <line x1='{hip_x}' y1='255' x2='{foot_x + 60}' y2='338' stroke='#0f172a' stroke-width='8' stroke-linecap='round'/>
          <rect x='610' y='95' width='140' height='116' rx='18' fill='{color}' opacity='0.95'/>
          <text x='680' y='138' text-anchor='middle' font-size='18' font-family='Arial, sans-serif' fill='white'>综合风险</text>
          <text x='680' y='182' text-anchor='middle' font-size='42' font-weight='700' font-family='Arial, sans-serif' fill='white'>{score}</text>
          <text x='80' y='50' font-size='28' font-weight='700' font-family='Arial, sans-serif' fill='#0f172a'>{title}</text>
          <text x='80' y='410' font-size='20' font-family='Arial, sans-serif' fill='#334155'>多模态前置预判 · 行为 + 环境 + 个体基线</text>
        </svg>
        """.strip()
        return "data:image/svg+xml;charset=utf-8," + quote(svg)


demo_engine = CompetitionDemoEngine()
