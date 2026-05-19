"""
评估指标模块

跌倒检测任务的关键指标:
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR（更适合不平衡数据）
- 特异度（Specificity）— 降低误报率
- 风险评分的 MAE, RMSE
"""
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix,
)
from scipy.stats import spearmanr


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    risk_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp + 1e-8)
        metrics["sensitivity"] = tp / (tp + fn + 1e-8)
        metrics["false_positive_rate"] = fp / (fp + tn + 1e-8)
        metrics["false_negative_rate"] = fn / (fn + tp + 1e-8)

    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            metrics["auc_pr"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0
            metrics["auc_pr"] = 0.0

    if risk_scores is not None:
        metrics["risk_score_mean_fall"] = risk_scores[y_true == 1].mean() if (y_true == 1).any() else 0
        metrics["risk_score_mean_adl"] = risk_scores[y_true == 0].mean() if (y_true == 0).any() else 0
        metrics["risk_score_separation"] = metrics["risk_score_mean_fall"] - metrics["risk_score_mean_adl"]

    return metrics


def compute_risk_metrics(
    pred_scores: np.ndarray,
    true_scores: np.ndarray,
) -> Dict[str, float]:
    error = pred_scores - true_scores
    abs_error = np.abs(error)

    try:
        spearman_corr, spearman_p = spearmanr(pred_scores, true_scores)
        if np.isnan(spearman_corr):
            spearman_corr, spearman_p = 0.0, 1.0
    except Exception:
        spearman_corr, spearman_p = 0.0, 1.0

    try:
        corr = np.corrcoef(pred_scores, true_scores)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    except Exception:
        corr = 0.0

    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mse": float(np.mean(error ** 2)),
        "max_error": float(np.max(abs_error)),
        "correlation": float(corr),
        "spearman": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "within_5": float(np.mean(abs_error <= 5.0)),
        "within_10": float(np.mean(abs_error <= 10.0)),
        "within_20": float(np.mean(abs_error <= 20.0)),
    }


def print_evaluation_report(
    metrics: Dict[str, float],
    title: str = "评估报告",
):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")

    sections = {
        "分类指标": ["accuracy", "precision", "recall", "f1", "specificity"],
        "AUC 指标": ["auc_roc", "auc_pr"],
        "误报分析": ["false_positive_rate", "false_negative_rate"],
        "风险评分": ["mae", "rmse", "spearman", "correlation", "within_10", "within_20"],
    }

    for section_name, keys in sections.items():
        values = {k: metrics.get(k) for k in keys if k in metrics}
        if values:
            print(f"\n  {section_name}:")
            for k, v in values.items():
                print(f"    {k:30s}: {v:.4f}")

    print(f"\n{'=' * 50}\n")
