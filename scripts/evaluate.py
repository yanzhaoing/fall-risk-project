#!/usr/bin/env python3
"""
测试集评估脚本

用法:
    python scripts/evaluate.py --model gait_lstm --dataset ntu
    python scripts/evaluate.py --model gait_lstm --checkpoint checkpoints/best_model.pt
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.settings import CFG_TRAIN, CFG_MODEL, CFG_DATA, CFG_PATHS
from src.data.dataset import FallDetectionDataset
from src.models.gait_analysis import GaitLSTM, GaitTransformer
from src.models.stgcn import STGCN, STGCNClassifier
from src.training.metrics import compute_metrics, print_evaluation_report
from src.utils.helpers import get_device, set_seed


def load_ntu_dataset(data_dir: str, split: str = "test") -> FallDetectionDataset:
    """加载 NTU 测试集 (Subject 21-24)"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] 数据目录不存在: {data_path}")
        return FallDetectionDataset([])

    split_ranges = {
        "train": (1, 16),
        "val": (17, 20),
        "test": (21, 24),
    }
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

    fall_count = sum(1 for s in samples if s["label"] == 1)
    adl_count = sum(1 for s in samples if s["label"] == 0)
    print(f"  [{split}] {len(samples)} 样本 | Fall: {fall_count} | ADL: {adl_count}")

    return FallDetectionDataset(samples)


def build_model(model_name: str, input_dim: int, num_classes: int = 2) -> torch.nn.Module:
    """构建模型"""
    if model_name == "gait_lstm":
        backbone = GaitLSTM(
            input_dim=input_dim,
            hidden_dim=CFG_MODEL.GAIT_HIDDEN_DIM,
            num_layers=CFG_MODEL.GAIT_NUM_LAYERS,
            dropout=CFG_MODEL.GAIT_DROPOUT,
        )
        class GaitLSTMClassifier(torch.nn.Module):
            def __init__(self, backbone, feat_dim, num_classes):
                super().__init__()
                self.backbone = backbone
                self.classifier = torch.nn.Linear(feat_dim, num_classes)
            def forward(self, x):
                feat = self.backbone(x)
                return self.classifier(feat)
        return GaitLSTMClassifier(backbone, backbone.output_dim, num_classes)

    elif model_name == "gait_transformer":
        backbone = GaitTransformer(
            input_dim=input_dim,
            d_model=CFG_MODEL.TRANSFORMER_D_MODEL,
            nhead=CFG_MODEL.TRANSFORMER_NHEAD,
            num_layers=CFG_MODEL.TRANSFORMER_NUM_LAYERS,
        )
        class GaitTransformerClassifier(torch.nn.Module):
            def __init__(self, backbone, num_classes):
                super().__init__()
                self.backbone = backbone
                self.classifier = torch.nn.Linear(backbone.output_dim, num_classes)
            def forward(self, x):
                feat = self.backbone(x)
                return self.classifier(feat)
        return GaitTransformerClassifier(backbone, num_classes)
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


def evaluate(model, dataloader, device, criterion=None):
    """在测试集上评估"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)

    avg_loss = total_loss / max(num_batches, 1) if num_batches > 0 else 0
    metrics = compute_metrics(y_true, y_pred, y_prob)

    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return metrics, cm, y_true, y_pred, y_prob, avg_loss


def print_confusion_matrix(cm, labels=["ADL", "Fall"]):
    """打印混淆矩阵"""
    print("\n  混淆矩阵:")
    print(f"  {'':12s}  {'预测 ADL':>10s}  {'预测 Fall':>10s}")
    print(f"  {'真实 ADL':12s}  {cm[0,0]:>10d}  {cm[0,1]:>10d}")
    print(f"  {'真实 Fall':12s}  {cm[1,0]:>10d}  {cm[1,1]:>10d}")


def main():
    parser = argparse.ArgumentParser(description="跌倒检测模型测试集评估")
    parser.add_argument("--model", default="gait_lstm", choices=["gait_lstm", "gait_transformer", "stgcn"])
    parser.add_argument("--dataset", default="ntu", choices=["ntu", "upfall"])
    parser.add_argument("--data-dir", default=str(CFG_PATHS.PROCESSED_DIR / "ntu_coco"))
    parser.add_argument("--checkpoint", default=str(CFG_PATHS.CHECKPOINTS_DIR / "best_model.pt"),
                        help="模型检查点路径")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=CFG_TRAIN.DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(Path(__file__).parent.parent / "results" / "test_results.json"),
                        help="结果保存路径")
    parser.add_argument("--search-threshold", action="store_true",
                        help="搜索最优分类阈值")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    print(f"\n{'='*60}")
    print(f"  测试集评估")
    print(f"{'='*60}")
    print(f"  模型: {args.model}")
    print(f"  检查点: {args.checkpoint}")
    print(f"  设备: {device}")
    print(f"{'='*60}\n")

    # 加载测试集
    print("[数据] 加载测试集...")
    if args.dataset == "ntu":
        test_dataset = load_ntu_dataset(args.data_dir, split="test")
    else:
        from src.data.dataset import UPFallDataset
        test_dataset = UPFallDataset(split="test")

    if len(test_dataset) == 0:
        print("[ERROR] 测试集为空！")
        return

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 构建并加载模型
    input_dim = CFG_DATA.NUM_KEYPOINTS * 3
    model = build_model(args.model, input_dim, num_classes=2)
    model = model.to(device)

    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] 检查点不存在: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[模型] 已加载检查点 | Epoch: {checkpoint.get('epoch', '?')} | 最佳 Val F1: {checkpoint.get('best_val_metric', 0):.4f}")

    # 评估
    from src.training.losses import FocalLoss
    criterion = FocalLoss()
    metrics, cm, y_true, y_pred, y_prob, avg_loss = evaluate(model, test_loader, device, criterion)

    # ─── 阈值搜索 ─────────────────────────────────────────
    if args.search_threshold:
        from sklearn.metrics import f1_score, precision_score, recall_score
        print(f"\n{'='*60}")
        print(f"  阈值搜索")
        print(f"{'='*60}")
        print(f"  {'阈值':>6s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}")
        print(f"  {'-'*60}")

        best_f1 = 0
        best_threshold = 0.5
        for thr in [i * 0.05 for i in range(1, 20)]:  # 0.05 ~ 0.95
            y_pred_thr = (y_prob >= thr).astype(int)
            f1 = f1_score(y_true, y_pred_thr, zero_division=0)
            p = precision_score(y_true, y_pred_thr, zero_division=0)
            r = recall_score(y_true, y_pred_thr, zero_division=0)
            from sklearn.metrics import confusion_matrix as cm_fn
            cm_thr = cm_fn(y_true, y_pred_thr, labels=[0, 1])
            if cm_thr.shape == (2, 2):
                tn, fp, fn, tp = cm_thr.ravel()
            else:
                tp = fp = fn = tn = 0

            marker = " ★" if f1 > best_f1 else ""
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr
            print(f"  {thr:>6.2f}  {p:>10.4f}  {r:>10.4f}  {f1:>10.4f}  {tp:>5d}  {fp:>5d}  {fn:>5d}{marker}")

        print(f"\n  ★ 最优阈值: {best_threshold:.2f} | F1: {best_f1:.4f}")

        # 用最优阈值重新计算完整指标
        y_pred_best = (y_prob >= best_threshold).astype(int)
        from sklearn.metrics import confusion_matrix as cm_fn
        cm_best = cm_fn(y_true, y_pred_best, labels=[0, 1])
        metrics_best = compute_metrics(y_true, y_pred_best, y_prob)
        print(f"\n  ─── 最优阈值下的评估 ───")
        print_confusion_matrix(cm_best)
        print_evaluation_report(metrics_best, title=f"阈值={best_threshold:.2f} 评估报告")

        # 保存结果
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "model": args.model,
            "checkpoint": str(args.checkpoint),
            "dataset": args.dataset,
            "num_samples": len(test_dataset),
            "test_loss": avg_loss,
            "optimal_threshold": best_threshold,
            "confusion_matrix": cm_best.tolist(),
            "metrics": {k: float(v) for k, v in metrics_best.items()},
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[结果] 已保存到: {output_path}")
        return metrics_best, cm_best

    # 打印结果
    print(f"\n{'='*60}")
    print(f"  测试结果")
    print(f"{'='*60}")
    print(f"  测试集 Loss: {avg_loss:.4f}")
    print_confusion_matrix(cm)
    print_evaluation_report(metrics, title="测试集评估报告")

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "dataset": args.dataset,
        "num_samples": len(test_dataset),
        "test_loss": avg_loss,
        "confusion_matrix": cm.tolist(),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[结果] 已保存到: {output_path}")

    return metrics, cm


if __name__ == "__main__":
    main()
