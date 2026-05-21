#!/usr/bin/env python3
"""Aggregate available evaluation artifacts into a submission-facing readiness brief."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_artifacts() -> Dict[str, Dict[str, object]]:
    artifact_specs = {
        "demo_evaluation": RESULTS_DIR / "demo_evaluation.json",
        "public_ntu_evaluation": RESULTS_DIR / "public_ntu_evaluation.json",
        "classification_test": RESULTS_DIR / "test_results.json",
        "feature_metrics": ROOT / "checkpoints" / "feature_model_metrics.json",
    }

    artifacts = {}
    for name, path in artifact_specs.items():
        payload = load_json(path)
        artifacts[name] = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "payload": payload,
        }
    return artifacts


def summarize(artifacts: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    completed = []
    missing = []

    for name, info in artifacts.items():
        if info["exists"]:
            completed.append(name)
        else:
            missing.append(name)

    public_eval = artifacts["public_ntu_evaluation"]["payload"] or {}
    demo_eval = artifacts["demo_evaluation"]["payload"] or {}
    feature_metrics = artifacts["feature_metrics"]["payload"] or {}

    overview = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_focus": "fall_risk",
        "submission_goal": "high_preliminary_score",
        "completed_artifacts": completed,
        "missing_artifacts": missing,
        "readiness": {
            "code_and_demo": True,
            "offline_public_validation": bool(public_eval),
            "system_flow_validation": bool(demo_eval),
            "ezviz_platform_evidence": False,
            "real_home_scene_measurement": False,
        },
        "headline_metrics": {
            "feature_best_model": feature_metrics.get("best_model"),
            "feature_results": feature_metrics.get("results", {}),
            "public_level_accuracy": (public_eval.get("metrics") or {}).get("risk_level_accuracy"),
            "public_binary_recall": ((public_eval.get("metrics") or {}).get("binary_intervention") or {}).get("recall"),
            "demo_level_accuracy": (demo_eval.get("metrics") or {}).get("level_accuracy"),
        },
    }
    return overview


def to_markdown(summary: Dict[str, object], artifacts: Dict[str, Dict[str, object]]) -> str:
    readiness = summary["readiness"]
    metrics = summary["headline_metrics"]

    lines = [
        "# 参赛就绪概览",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 当前结论",
        "",
        "项目已具备代码、演示、离线验证和系统闭环的基础骨架，可以作为初审冲分版本继续完善。",
        "当前仍缺少萤石平台实证和真实居家场景实测，尚不能把演示结果当成最终落地效果。",
        "",
        "## 关键状态",
        "",
        f"- 代码与演示骨架：{'已完成' if readiness['code_and_demo'] else '未完成'}",
        f"- 公开数据集离线验证：{'已完成' if readiness['offline_public_validation'] else '未完成'}",
        f"- 系统流程验证：{'已完成' if readiness['system_flow_validation'] else '未完成'}",
        f"- 萤石平台证据：{'已完成' if readiness['ezviz_platform_evidence'] else '待补充'}",
        f"- 真实居家实测：{'已完成' if readiness['real_home_scene_measurement'] else '待补充'}",
        "",
        "## 已汇总指标",
        "",
        f"- 特征主线最佳模型：{metrics.get('feature_best_model') or '暂无'}",
        f"- 公开数据集风险等级准确率：{metrics.get('public_level_accuracy') or '暂无'}",
        f"- 公开数据集高风险召回率：{metrics.get('public_binary_recall') or '暂无'}",
        f"- 内置演示等级准确率：{metrics.get('demo_level_accuracy') or '暂无'}",
        "",
        "## 结果文件清单",
        "",
    ]

    for name, info in artifacts.items():
        status = "存在" if info["exists"] else "缺失"
        lines.append(f"- {name}: {status} · {info['path']}")

    lines.extend([
        "",
        "## 下一步最小补强",
        "",
        "1. 补一组真实居家场景视频或图片证据。",
        "2. 补一份萤石开放平台调用截图或日志。",
        "3. 用最终版本重新导出测试报告、专项研究报告和部署说明。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = collect_artifacts()
    summary = summarize(artifacts)

    json_path = RESULTS_DIR / "submission_readiness.json"
    md_path = RESULTS_DIR / "参赛就绪概览.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(summary, artifacts), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
