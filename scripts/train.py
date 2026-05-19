#!/usr/bin/env python3
"""
训练入口脚本

用法:
    # 分类模式（二分类，基准线）
    python scripts/train.py --model gait_lstm --dataset ntu --epochs 50

    # 回归模式（0-100 连续风险评分）
    python scripts/train.py --model gait_lstm --task regression --dataset ntu --epochs 100
"""
import sys
import argparse
import json
import platform
import shutil
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 自动清理 __pycache__（防止主机/VM 代码不同步导致崩溃）
if platform.system() == "Windows":
    for pycache in Path(__file__).parent.parent.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)

import numpy as np
import torch
from config.settings import CFG_TRAIN, CFG_MODEL, CFG_DATA, CFG_PATHS
from src.data.dataset import FallDetectionDataset, UPFallDataset
from src.data.dataloader import create_dataloaders
from src.data.augmentation import get_train_augmentor
from src.data.risk_dataset import RiskAwareFallDataset
from src.features.context_features import SCENE_FEATURE_DIM
from src.models.gait_analysis import GaitLSTM, GaitTransformer, GaitRiskScorer
from src.models.stgcn import STGCN, STGCNClassifier
from src.models.multimodal_risk import MultiModalRiskScorer
from src.training.losses import FocalLoss, RiskScoreLoss
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, get_device, count_parameters


def load_ntu_dataset(data_dir: str, split: str = "train", max_samples: int = 0,
                     risk_mode: bool = False, multimodal: bool = False):
    """加载 NTU COCO 格式数据集"""
    data_path = Path(data_dir)
    dataset_cls = RiskAwareFallDataset if (risk_mode or multimodal) else FallDetectionDataset

    if not data_path.exists():
        print(f"[ERROR] NTU 数据目录不存在: {data_path}")
        return dataset_cls([], risk_mode=risk_mode, multimodal=multimodal)

    split_ranges = {
        "train": (1, 16),
        "val": (17, 20),
        "test": (21, 24),
    }
    lo, hi = split_ranges.get(split, (1, 16))

    samples = []
    loaded = 0
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
            loaded += 1

            if max_samples > 0 and loaded >= max_samples:
                break
        if max_samples > 0 and loaded >= max_samples:
            break

    fall_count = sum(1 for s in samples if s["label"] == 1)
    adl_count = sum(1 for s in samples if s["label"] == 0)
    if multimodal:
        mode_str = "多模态（风险评分+真实上下文特征）"
    elif risk_mode:
        mode_str = "风险评分（前置风险伪标签）"
    else:
        mode_str = "二分类"
    print(f"  [{split}] {len(samples)} 样本 | Fall: {fall_count} | ADL: {adl_count} | 模式: {mode_str}")

    return dataset_cls(samples, risk_mode=risk_mode, multimodal=multimodal)


def build_model(model_name: str, input_dim: int, num_classes: int = 2,
                task: str = "classification") -> torch.nn.Module:
    """构建模型"""
    if model_name == "gait_lstm":
        backbone = GaitLSTM(
            input_dim=input_dim,
            hidden_dim=CFG_MODEL.GAIT_HIDDEN_DIM,
            num_layers=CFG_MODEL.GAIT_NUM_LAYERS,
            dropout=CFG_MODEL.GAIT_DROPOUT,
        )

        if task == "multimodal":
            return MultiModalRiskScorer(
                gait_backbone=backbone,
                gait_dim=backbone.output_dim,
                scene_dim=SCENE_FEATURE_DIM,
                fusion_dim=128,
                fusion_strategy="gated",
            )

        if task == "regression":
            return GaitRiskScorer(backbone, backbone.output_dim)

        class GaitLSTMClassifier(torch.nn.Module):
            def __init__(self, backbone, feat_dim, num_classes):
                super().__init__()
                self.backbone = backbone
                self.classifier = torch.nn.Linear(feat_dim, num_classes)

            def forward(self, x):
                feat = self.backbone(x)
                return self.classifier(feat)

        return GaitLSTMClassifier(backbone, backbone.output_dim, num_classes)

    if model_name == "gait_transformer":
        backbone = GaitTransformer(
            input_dim=input_dim,
            d_model=CFG_MODEL.TRANSFORMER_D_MODEL,
            nhead=CFG_MODEL.TRANSFORMER_NHEAD,
            num_layers=CFG_MODEL.TRANSFORMER_NUM_LAYERS,
        )

        if task == "multimodal":
            return MultiModalRiskScorer(
                gait_backbone=backbone,
                gait_dim=backbone.output_dim,
                scene_dim=SCENE_FEATURE_DIM,
                fusion_dim=128,
                fusion_strategy="gated",
            )

        if task == "regression":
            return GaitRiskScorer(backbone, backbone.output_dim)

        class GaitTransformerClassifier(torch.nn.Module):
            def __init__(self, backbone, num_classes):
                super().__init__()
                self.backbone = backbone
                self.classifier = torch.nn.Linear(backbone.output_dim, num_classes)

            def forward(self, x):
                feat = self.backbone(x)
                return self.classifier(feat)

        return GaitTransformerClassifier(backbone, num_classes)

    if model_name == "stgcn":
        backbone = STGCN(
            input_dim=3,
            num_nodes=CFG_DATA.NUM_KEYPOINTS,
            base_channels=CFG_MODEL.STGCN_BASE_CHANNELS,
            num_stages=CFG_MODEL.STGCN_NUM_STAGES,
            temporal_kernel=CFG_MODEL.STGCN_TEMPORAL_KERNEL,
            dropout=CFG_MODEL.STGCN_DROPOUT,
        )

        if task == "multimodal":
            return MultiModalRiskScorer(
                gait_backbone=backbone,
                gait_dim=backbone.output_dim,
                scene_dim=SCENE_FEATURE_DIM,
                fusion_dim=128,
                fusion_strategy="gated",
            )

        if task == "regression":
            return GaitRiskScorer(backbone, backbone.output_dim)

        return STGCNClassifier(backbone, num_classes)

    raise ValueError(f"未知模型: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="跌倒风险模型训练")
    parser.add_argument("--model", default="gait_lstm", choices=["gait_lstm", "gait_transformer", "stgcn"])
    parser.add_argument("--task", default="classification", choices=["classification", "regression", "multimodal"],
                        help="classification=二分类 | regression=0-100风险评分 | multimodal=步态+上下文融合")
    parser.add_argument("--dataset", default="ntu", choices=["ntu", "upfall"])
    parser.add_argument("--data-dir", default=str(CFG_PATHS.PROCESSED_DIR / "ntu_coco"), help="NTU 数据目录")
    parser.add_argument("--epochs", type=int, default=CFG_TRAIN.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CFG_TRAIN.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=CFG_TRAIN.LEARNING_RATE)
    parser.add_argument("--device", default=CFG_TRAIN.DEVICE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="最多加载多少样本（0=全部，调试用）")
    parser.add_argument("--pretrained-backbone", default="", help="预训练骨干权重路径（多模态微调用）")
    parser.add_argument("--freeze-gait", action="store_true", help="冻结步态骨干（多模态微调时用）")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    is_regression = (args.task in ["regression", "multimodal"])
    is_multimodal = (args.task == "multimodal")
    print(f"[训练] 模型: {args.model} | 任务: {args.task} | 数据集: {args.dataset} | 设备: {device} | 轮数: {args.epochs}")

    if args.dataset == "ntu":
        print(f"[数据] 加载 NTU 数据: {args.data_dir}")
        train_dataset = load_ntu_dataset(args.data_dir, split="train", max_samples=args.max_samples,
                                         risk_mode=is_regression, multimodal=is_multimodal)
        val_dataset = load_ntu_dataset(args.data_dir, split="val", max_samples=args.max_samples,
                                       risk_mode=is_regression, multimodal=is_multimodal)
        test_dataset = load_ntu_dataset(args.data_dir, split="test", max_samples=args.max_samples,
                                        risk_mode=is_regression, multimodal=is_multimodal)
        train_dataset.transform = get_train_augmentor()
    else:
        train_dataset = UPFallDataset(split="train", transform=get_train_augmentor())
        val_dataset = UPFallDataset(split="val")
        test_dataset = None
        if is_regression:
            print("⚠️ [WARNING] 当前更完整的前置风险标签与 scene/context 特征仅接入 NTU 主线。")

    if len(train_dataset) == 0:
        print("[ERROR] 训练集为空！")
        return

    loaders = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        use_weighted_sampling=not is_regression,
    )

    input_dim = CFG_DATA.NUM_KEYPOINTS * 3
    model = build_model(args.model, input_dim, num_classes=2, task=args.task)

    if args.pretrained_backbone and Path(args.pretrained_backbone).exists():
        ckpt = torch.load(args.pretrained_backbone, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if hasattr(model, "gait_backbone"):
            backbone_state = {k.replace("backbone.", "gait_backbone."): v for k, v in state.items()
                              if k.startswith("backbone.")}
            missing, unexpected = model.load_state_dict(backbone_state, strict=False)
            print(f"[训练] 加载预训练骨干: {args.pretrained_backbone}")
            print(f"  加载: {len(backbone_state)} 层 | 缺失: {len(missing)} | 多余: {len(unexpected)}")

    if args.freeze_gait and hasattr(model, "gait_backbone"):
        for p in model.gait_backbone.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[训练] 步态骨干已冻结 | 可训练: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")

    if is_multimodal and args.dataset == "ntu":
        print("[训练] 多模态上下文特征已启用：不再使用全零 scene_feat。")
        print("  当前特征来自骨架轨迹、姿态稳定性和可推断的房间先验。")

    model = model.to(device)
    print(f"[训练] 参数量: {count_parameters(model):,}")

    criterion = RiskScoreLoss(mse_weight=1.0, ranking_weight=0.5, margin=10.0) if is_regression else FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=CFG_TRAIN.WEIGHT_DECAY)

    warmup_epochs = min(5, args.epochs // 10)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=str(device), task=args.task)
    trainer.train(loaders["train"], loaders["val"], epochs=args.epochs)

    if "test" in loaders:
        print("\n" + "=" * 50)
        print("  测试集评估")
        print("=" * 50)

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loaders["test"]:
                if len(batch) == 4:
                    inputs, scene, targets, _ = batch
                    inputs = inputs.to(device)
                    scene = scene.to(device)
                    outputs = model(inputs, scene)
                    preds = outputs
                elif len(batch) == 3:
                    inputs, targets, _ = batch
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds = outputs
                else:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)

        if is_regression:
            from src.training.metrics import compute_risk_metrics
            metrics = compute_risk_metrics(y_pred, y_true)
            print(f"  MAE:        {metrics['mae']:.2f}")
            print(f"  RMSE:       {metrics['rmse']:.2f}")
            print(f"  Spearman:   {metrics['spearman']:.4f}")
            print(f"  Correlation:{metrics['correlation']:.4f}")
            print(f"  ±10 分命中率: {metrics['within_10']:.2%}")
            print(f"  ±20 分命中率: {metrics['within_20']:.2%}")

            from src.models.risk_scoring import get_risk_level
            level_correct = 0
            for pred, true in zip(y_pred, y_true):
                if get_risk_level(float(pred)) == get_risk_level(float(true)):
                    level_correct += 1
            level_acc = level_correct / len(y_pred)
            print(f"  风险等级准确率: {level_acc:.2%}")
        else:
            from sklearn.metrics import classification_report
            print(classification_report(y_true, y_pred, target_names=["ADL", "Fall"], zero_division=0))


if __name__ == "__main__":
    main()
