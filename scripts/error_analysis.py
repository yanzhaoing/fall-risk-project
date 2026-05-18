#!/usr/bin/env python3
"""
错误分析脚本

找出模型错在哪些样本、什么模式，指导后续优化方向。

用法:
    python scripts/error_analysis.py --model gait_lstm --device cuda:0
"""
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.settings import CFG_TRAIN, CFG_MODEL, CFG_DATA, CFG_PATHS
from src.data.dataset import FallDetectionDataset
from src.models.gait_analysis import GaitLSTM, GaitTransformer
from src.models.stgcn import STGCN, STGCNClassifier
from src.utils.helpers import get_device, set_seed


def load_ntu_dataset(data_dir: str, split: str = "test") -> FallDetectionDataset:
    """加载 NTU 测试集"""
    data_path = Path(data_dir)
    split_ranges = {"train": (1, 16), "val": (17, 20), "test": (21, 24)}
    lo, hi = split_ranges.get(split, (21, 24))

    samples = []
    for subj_dir in sorted(data_path.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("Subject_"):
            continue
        subj_id = int(subj_dir.name.split("_")[-1])
        if not (lo <= subj_id <= hi):
            continue
        for act_dir in sorted(subj_dir.iterdir()):
            if not act_dir.is_dir():
                continue
            kpt_file = act_dir / "keypoints.json"
            if not kpt_file.exists():
                continue
            with open(kpt_file) as f:
                d = json.load(f)
            samples.append({
                "keypoints": np.array(d["keypoints"]),
                "label": d["label"],
                "metadata": {"subject": subj_dir.name, "action": act_dir.name},
            })
    return FallDetectionDataset(samples)


def build_model(model_name, input_dim, num_classes=2):
    if model_name == "gait_lstm":
        backbone = GaitLSTM(
            input_dim=input_dim,
            hidden_dim=CFG_MODEL.GAIT_HIDDEN_DIM,
            num_layers=CFG_MODEL.GAIT_NUM_LAYERS,
            dropout=CFG_MODEL.GAIT_DROPOUT,
        )
        class Cls(torch.nn.Module):
            def __init__(s, bb, fd, nc):
                super().__init__()
                s.backbone = bb
                s.classifier = torch.nn.Linear(fd, nc)
            def forward(s, x):
                return s.classifier(s.backbone(x))
        return Cls(backbone, backbone.output_dim, num_classes)
    elif model_name == "stgcn":
        backbone = STGCN(
            input_dim=3,
            num_nodes=CFG_DATA.NUM_KEYPOINTS,
            base_channels=CFG_MODEL.STGCN_BASE_CHANNELS,
            num_stages=CFG_MODEL.STGCN_NUM_STAGES,
            temporal_kernel=CFG_MODEL.STGCN_TEMPORAL_KERNEL,
            dropout=CFG_MODEL.STGCN_DROPOUT,
        )
        return STGCNClassifier(backbone, num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="错误分析")
    parser.add_argument("--model", default="gait_lstm", choices=["gait_lstm", "gait_transformer", "stgcn"])
    parser.add_argument("--data-dir", default=str(CFG_PATHS.PROCESSED_DIR / "ntu_coco"))
    parser.add_argument("--checkpoint", default=str(CFG_PATHS.CHECKPOINTS_DIR / "best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=CFG_TRAIN.DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    print(f"\n{'='*60}")
    print(f"  错误分析")
    print(f"{'='*60}\n")

    # 加载测试集
    print("[数据] 加载测试集...")
    test_dataset = load_ntu_dataset(args.data_dir, split="test")

    # 收集元信息（DataLoader 会打乱顺序，所以先存下来）
    all_metadata = [s["metadata"] for s in test_dataset.samples]
    all_labels = [s["label"] for s in test_dataset.samples]

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 加载模型
    input_dim = CFG_DATA.NUM_KEYPOINTS * 3
    model = build_model(args.model, input_dim, num_classes=2)
    model = model.to(device)

    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[模型] 已加载 | 最佳 Val F1: {checkpoint.get('best_val_metric', 0):.4f}")

    # 推理
    model.eval()
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs[:, 1].cpu().numpy())
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.array(all_labels)

    # ─── 1. 总体错误统计 ─────────────────────────────────────
    errors = y_pred != y_true
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    tp_mask = (y_pred == 1) & (y_true == 1)
    tn_mask = (y_pred == 0) & (y_true == 0)

    print(f"\n{'='*60}")
    print(f"  总体统计")
    print(f"{'='*60}")
    print(f"  总样本: {len(y_true)}")
    print(f"  正确: {(~errors).sum()} ({(~errors).mean()*100:.1f}%)")
    print(f"  错误: {errors.sum()} ({errors.mean()*100:.1f}%)")
    print(f"  TP: {tp_mask.sum()} | TN: {tn_mask.sum()} | FP: {fp_mask.sum()} | FN: {fn_mask.sum()}")

    # ─── 2. 按被试分析 ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  按被试分析")
    print(f"{'='*60}")
    print(f"  {'被试':>12s}  {'总数':>5s}  {'Fall':>5s}  {'错误':>5s}  {'FP':>5s}  {'FN':>5s}  {'错误率':>7s}")
    print(f"  {'-'*55}")

    subj_stats = defaultdict(lambda: {"total": 0, "fall": 0, "errors": 0, "fp": 0, "fn": 0})
    for i, meta in enumerate(all_metadata):
        subj = meta["subject"]
        subj_stats[subj]["total"] += 1
        if y_true[i] == 1:
            subj_stats[subj]["fall"] += 1
        if errors[i]:
            subj_stats[subj]["errors"] += 1
        if fp_mask[i]:
            subj_stats[subj]["fp"] += 1
        if fn_mask[i]:
            subj_stats[subj]["fn"] += 1

    for subj in sorted(subj_stats.keys()):
        s = subj_stats[subj]
        rate = s["errors"] / max(s["total"], 1) * 100
        print(f"  {subj:>12s}  {s['total']:>5d}  {s['fall']:>5d}  {s['errors']:>5d}  {s['fp']:>5d}  {s['fn']:>5d}  {rate:>6.1f}%")

    # ─── 3. 按动作类型分析 ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  按动作类型分析（错误最多的 Top 20）")
    print(f"{'='*60}")
    print(f"  {'动作':>30s}  {'标签':>5s}  {'总数':>5s}  {'错误':>5s}  {'错误率':>7s}")
    print(f"  {'-'*60}")

    act_stats = defaultdict(lambda: {"total": 0, "errors": 0, "label": 0})
    for i, meta in enumerate(all_metadata):
        act = meta["action"]
        act_stats[act]["total"] += 1
        act_stats[act]["label"] = y_true[i]
        if errors[i]:
            act_stats[act]["errors"] += 1

    # 按错误数排序
    sorted_acts = sorted(act_stats.items(), key=lambda x: x[1]["errors"], reverse=True)
    for act, s in sorted_acts[:20]:
        rate = s["errors"] / max(s["total"], 1) * 100
        label_str = "Fall" if s["label"] == 1 else "ADL"
        print(f"  {act:>30s}  {label_str:>5s}  {s['total']:>5d}  {s['errors']:>5d}  {rate:>6.1f}%")

    # ─── 4. 置信度分析 ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  置信度分析（预测概率分布）")
    print(f"{'='*60}")

    # 正确样本的置信度
    correct_probs = y_prob[~errors]
    wrong_probs = y_prob[errors]

    # FP 的置信度（预测为 Fall 但实际是 ADL）
    fp_probs = y_prob[fp_mask]
    # FN 的置信度（预测为 ADL 但实际是 Fall）
    fn_probs = y_prob[fn_mask]

    print(f"  正确样本置信度: mean={correct_probs.mean():.3f}, std={correct_probs.std():.3f}")
    if len(wrong_probs) > 0:
        print(f"  错误样本置信度: mean={wrong_probs.mean():.3f}, std={wrong_probs.std():.3f}")
    if len(fp_probs) > 0:
        print(f"  FP 置信度:      mean={fp_probs.mean():.3f}, std={fp_probs.std():.3f}")
    if len(fn_probs) > 0:
        print(f"  FN 置信度:      mean={fn_probs.mean():.3f}, std={fn_probs.std():.3f}")

    # 置信度分布直方图（文本版）
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"\n  预测概率分布:")
    print(f"  {'区间':>10s}  {'正确':>6s}  {'FP':>6s}  {'FN':>6s}")
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j+1]
        c = ((correct_probs >= lo) & (correct_probs < hi)).sum()
        fp = ((fp_probs >= lo) & (fp_probs < hi)).sum() if len(fp_probs) > 0 else 0
        fn = ((fn_probs >= lo) & (fn_probs < hi)).sum() if len(fn_probs) > 0 else 0
        print(f"  [{lo:.1f}-{hi:.1f})  {c:>6d}  {fp:>6d}  {fn:>6d}")

    # ─── 5. 高置信度错误 ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  高置信度错误（模型很确定但判错了）")
    print(f"{'='*60}")

    # FP 中概率 > 0.8 的
    high_conf_fp = np.where(fp_mask & (y_prob > 0.8))[0]
    print(f"\n  FP 且概率 > 0.8: {len(high_conf_fp)} 个")
    for idx in high_conf_fp[:10]:
        meta = all_metadata[idx]
        print(f"    [{idx}] {meta['subject']} / {meta['action']} | prob={y_prob[idx]:.3f}")

    # FN 中概率 < 0.2 的
    high_conf_fn = np.where(fn_mask & (y_prob < 0.2))[0]
    print(f"\n  FN 且概率 < 0.2: {len(high_conf_fn)} 个")
    for idx in high_conf_fn[:10]:
        meta = all_metadata[idx]
        print(f"    [{idx}] {meta['subject']} / {meta['action']} | prob={y_prob[idx]:.3f}")

    # ─── 6. 边界样本分析 ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  边界样本（概率在 0.4~0.6 之间，模型不确定）")
    print(f"{'='*60}")

    boundary_mask = (y_prob >= 0.4) & (y_prob <= 0.6)
    boundary_errors = boundary_mask & errors
    print(f"  边界样本总数: {boundary_mask.sum()}")
    print(f"  其中判错: {boundary_errors.sum()} ({boundary_errors.sum()/max(boundary_mask.sum(),1)*100:.1f}%)")

    # ─── 7. 结论 ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  分析结论")
    print(f"{'='*60}")

    # 被试差异
    subj_errors = [(s, d["errors"]/max(d["total"],1)) for s, d in subj_stats.items()]
    subj_errors.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  1. 被试差异:")
    print(f"     错误率最高: {subj_errors[0][0]} ({subj_errors[0][1]*100:.1f}%)")
    print(f"     错误率最低: {subj_errors[-1][0]} ({subj_errors[-1][1]*100:.1f}%)")

    # FP vs FN 比例
    print(f"\n  2. 错误类型:")
    print(f"     FP (误报): {fp_mask.sum()} 个 | FN (漏报): {fn_mask.sum()} 个")
    if fp_mask.sum() > fn_mask.sum():
        print(f"     → 误报为主，说明阈值偏低或正常样本特征空间重叠")
    else:
        print(f"     → 漏报为主，说明跌倒特征学习不充分")

    # 置信度
    if len(fp_probs) > 0 and fp_probs.mean() > 0.7:
        print(f"\n  3. ⚠️ FP 置信度偏高 ({fp_probs.mean():.3f})，说明模型对某些正常动作有系统性误判")
    if len(fn_probs) > 0 and fn_probs.mean() < 0.3:
        print(f"\n  3. ⚠️ FN 置信度偏低 ({fn_probs.mean():.3f})，说明某些跌倒模型完全没学到")

    print(f"\n{'='*60}\n")

    # 保存详细结果
    results = {
        "total": len(y_true),
        "errors": int(errors.sum()),
        "tp": int(tp_mask.sum()),
        "tn": int(tn_mask.sum()),
        "fp": int(fp_mask.sum()),
        "fn": int(fn_mask.sum()),
        "subject_stats": {k: v for k, v in subj_stats.items()},
        "high_conf_fp_count": len(high_conf_fp),
        "high_conf_fn_count": len(high_conf_fn),
        "boundary_count": int(boundary_mask.sum()),
        "boundary_errors": int(boundary_errors.sum()),
    }
    output_path = Path(__file__).parent.parent / "results" / "error_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[结果] 已保存到: {output_path}")


if __name__ == "__main__":
    main()
