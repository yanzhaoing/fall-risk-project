#!/usr/bin/env python3
"""
方案A：特征工程 — 步态特征 + 简单模型

替代 LSTM，用 16 维生物力学特征 + MLP/GBDT → 风险分数 (0-100)

预期结果：
  MAE: 8-10 (LSTM=12.55)
  Spearman: 0.4+ (LSTM=0.13)
  Risk Level Acc: 92%+ (LSTM=88.89%)
  训练时间: 秒级 (LSTM=40 epochs)

用法:
    python scripts/train_feature_model.py
    python scripts/train_feature_model.py --skip-mlp
"""
import sys
import json
import pickle
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.gait_features import GaitFeatureExtractor, FEATURE_NAMES
from src.models.risk_scoring import get_risk_level
from src.models.feature_mlp import FeatureMLP

# ── NTU 动作风险标签映射（跟 dataset.py 保持一致）──
ACTION_RISK_SCORES = {
    "A1": 10, "A2": 8, "A3": 5, "A4": 12, "A5": 15,
    "A6": 18, "A7": 8, "A8": 10, "A9": 10, "A10": 5,
    "A11": 12, "A12": 8, "ADL": 15,
    "A43": 85, "A44": 85, "A45": 85, "A46": 85,
    "A47": 90, "A48": 85, "A49": 80, "A50": 85,
    "Fall": 85,
}


# ══════════════════════════════════════════════════════
#  特征提取
# ══════════════════════════════════════════════════════

def assign_risk_score(action_name, label):
    """根据动作名称和标签确定风险分数"""
    if label == 1 or "Fall" in action_name or "fall" in action_name:
        return float(ACTION_RISK_SCORES.get("Fall", 85))
    elif "ADL" in action_name:
        try:
            act_num = int(action_name.split("_")[-1])
            return float(ACTION_RISK_SCORES.get(f"A{act_num}", 15))
        except (ValueError, IndexError):
            return float(ACTION_RISK_SCORES.get("ADL", 15))
    return float(ACTION_RISK_SCORES.get("ADL", 15))


def load_ntu_samples(data_dir, split, max_samples=0):
    """加载 NTU COCO 数据集样本"""
    data_path = Path(data_dir)
    split_ranges = {"train": (1, 16), "val": (17, 20), "test": (21, 24)}
    lo, hi = split_ranges[split]

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
                "action": act_dir.name,
            })
            loaded += 1
            if max_samples > 0 and loaded >= max_samples:
                break
        if max_samples > 0 and loaded >= max_samples:
            break
    return samples


def extract_features(samples, extractor, cache_path=None):
    """
    从样本中提取步态特征和风险标签

    Returns:
        features: (N, 16) numpy array
        risk_scores: (N,) numpy array
        labels: (N,) numpy array
    """
    # 检查缓存（验证样本数一致）
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path)
        cached_n = len(data["features"])
        if cached_n == len(samples):
            print(f"  加载缓存: {cache_path} ({cached_n} 样本)")
            return data["features"], data["risk_scores"], data["labels"]
        else:
            print(f"  ⚠️ 缓存样本数不匹配: 缓存={cached_n}, 当前={len(samples)}，重新提取")

    features_list = []
    risk_scores = []
    labels = []

    for i, sample in enumerate(samples):
        skeleton = sample["keypoints"]  # (T, 17, 3)

        # 确保序列长度
        T = skeleton.shape[0]
        if T > 30:
            skeleton = skeleton[:30]
        elif T < 30:
            pad = np.zeros((30 - T, 17, 3))
            skeleton = np.concatenate([skeleton, pad], axis=0)

        # 提取特征
        feat = extractor.extract_vector(skeleton)  # (16,)

        # 替换 NaN/Inf
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        features_list.append(feat)
        risk_scores.append(assign_risk_score(sample["action"], sample["label"]))
        labels.append(sample["label"])

        if (i + 1) % 2000 == 0:
            print(f"    已提取 {i+1}/{len(samples)} 样本...")

    features = np.array(features_list, dtype=np.float32)
    risk_scores = np.array(risk_scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # 保存缓存
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features,
                            risk_scores=risk_scores, labels=labels)
        print(f"  缓存已保存: {cache_path}")

    return features, risk_scores, labels


# ══════════════════════════════════════════════════════
#  GBDT 模型
# ══════════════════════════════════════════════════════

def train_gbr(X_train, y_train, X_val, y_val, n_estimators=300):
    """训练 Gradient Boosting Regressor"""
    print("\n  训练 GBDT...")
    t0 = time.time()

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # 评估
    val_pred = model.predict(X_val)
    val_pred = np.clip(val_pred, 0, 100)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_spearman, val_sp_p = spearmanr(y_val, val_pred)
    val_level_acc = np.mean([
        get_risk_level(float(t)) == get_risk_level(float(p))
        for t, p in zip(y_val, val_pred)
    ])

    print(f"  GBDT 训练时间: {train_time:.1f}s")
    print(f"  Val MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | "
          f"Spearman: {val_spearman:.4f} | Level Acc: {val_level_acc:.2%}")

    return model, {
        "mae": val_mae, "rmse": val_rmse,
        "spearman": val_spearman, "level_acc": val_level_acc,
        "train_time": train_time,
    }






def train_mlp(X_train, y_train, X_val, y_val, epochs=200, lr=0.001,
              batch_size=256, patience=20, device="cpu"):
    """训练 MLP"""
    print("\n  训练 MLP...")
    t0 = time.time()

    model = FeatureMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 转换为 tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    best_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy()
        val_pred = np.clip(val_pred, 0, 100)
        val_mae = mean_absolute_error(y_val, val_pred)

        if val_mae < best_mae - 0.01:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            val_spearman, _ = spearmanr(y_val, val_pred)
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss:.2f} | "
                  f"Val MAE: {val_mae:.2f} | Spearman: {val_spearman:.4f}")

        if patience_counter >= patience:
            print(f"    早停 (epoch {epoch+1})")
            break

    # 恢复最佳模型
    if best_state:
        model.load_state_dict(best_state)

    train_time = time.time() - t0

    # 最终评估
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()
    val_pred = np.clip(val_pred, 0, 100)
    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_spearman, _ = spearmanr(y_val, val_pred)
    val_level_acc = np.mean([
        get_risk_level(float(t)) == get_risk_level(float(p))
        for t, p in zip(y_val, val_pred)
    ])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MLP 训练时间: {train_time:.1f}s | 参数量: {n_params:,}")
    print(f"  Val MAE: {val_mae:.2f} | RMSE: {val_rmse:.2f} | "
          f"Spearman: {val_spearman:.4f} | Level Acc: {val_level_acc:.2%}")

    return model, {
        "mae": val_mae, "rmse": val_rmse,
        "spearman": val_spearman, "level_acc": val_level_acc,
        "train_time": train_time, "n_params": n_params,
    }




# ══════════════════════════════════════════════════════
#  评估与分析
# ══════════════════════════════════════════════════════

def evaluate_on_test(model, X_test, y_test, labels_test, model_name="Model"):
    """在测试集上全面评估"""
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            pred = model(X_t).cpu().numpy()
    else:
        pred = model.predict(X_test)

    pred = np.clip(pred, 0, 100)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    spearman, sp_p = spearmanr(y_test, pred)

    # 风险等级准确率
    level_correct = sum(
        1 for t, p in zip(y_test, pred)
        if get_risk_level(float(t)) == get_risk_level(float(p))
    )
    level_acc = level_correct / len(y_test)

    # Fall vs ADL 分离度
    fall_mask = labels_test == 1
    adl_mask = labels_test == 0
    fall_mean = pred[fall_mask].mean() if fall_mask.any() else 0
    adl_mean = pred[adl_mask].mean() if adl_mask.any() else 0

    print(f"\n  === {model_name} 测试集结果 ===")
    print(f"  MAE:              {mae:.2f}")
    print(f"  RMSE:             {rmse:.2f}")
    print(f"  Spearman:         {spearman:.4f} (p={sp_p:.2e})")
    print(f"  Risk Level Acc:   {level_acc:.2%}")
    print(f"  Fall 均分:         {fall_mean:.1f}")
    print(f"  ADL 均分:         {adl_mean:.1f}")
    print(f"  分离度:           {fall_mean - adl_mean:.1f}")

    return {
        "mae": mae, "rmse": rmse,
        "spearman": spearman, "level_acc": level_acc,
        "fall_mean": fall_mean, "adl_mean": adl_mean,
    }


def show_feature_importance(model, feature_names):
    """展示特征重要性（仅 GBDT）"""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n  === 特征重要性 (Top 10) ===")
    for rank, idx in enumerate(indices[:10]):
        bar = "█" * int(importances[idx] * 50)
        print(f"  {rank+1:2d}. {feature_names[idx]:20s} {importances[idx]:.3f} {bar}")

    return {feature_names[i]: float(importances[i]) for i in indices}


def save_model_artifacts(model, scaler, feature_names, metrics, save_dir):
    """保存模型和相关文件"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    if isinstance(model, nn.Module):
        model_path = save_dir / "feature_mlp.pt"
        torch.save(model.state_dict(), model_path)
    else:
        model_path = save_dir / "feature_gbr.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # 保存 scaler
    scaler_path = save_dir / "feature_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # 保存指标
    metrics_path = save_dir / "feature_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  模型已保存: {model_path}")
    print(f"  Scaler 已保存: {scaler_path}")
    print(f"  指标已保存: {metrics_path}")


# ══════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="特征工程方案 — 步态特征 + MLP/GBDT")
    parser.add_argument("--data-dir", default=str(
        Path(__file__).parent.parent / "data" / "processed" / "ntu_coco"))
    parser.add_argument("--model", default="both", choices=["gbr", "mlp", "both"])
    parser.add_argument("--epochs", type=int, default=200, help="MLP 训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="MLP 学习率")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=0, help="最大样本数（调试用）")
    parser.add_argument("--skip-mlp", action="store_true", help="跳过 MLP 训练")
    parser.add_argument("--cache-dir", default=str(
        Path(__file__).parent.parent / "data" / "processed" / "feature_cache"))
    args = parser.parse_args()

    print("=" * 60)
    print("  方案A：特征工程 — 步态特征 + 简单模型")
    print("=" * 60)

    # ── 1. 加载数据 ──
    print("\n[1/4] 加载 NTU 数据...")
    cache_dir = Path(args.cache_dir)

    extractor = GaitFeatureExtractor(fps=30.0)

    splits = {}
    for split in ["train", "val", "test"]:
        print(f"\n  === {split} ===")
        samples = load_ntu_samples(args.data_dir, split, args.max_samples)
        print(f"  样本数: {len(samples)}")

        fall_count = sum(1 for s in samples if s["label"] == 1)
        print(f"  Fall: {fall_count} | ADL: {len(samples) - fall_count}")

        cache_path = cache_dir / f"ntu_{split}_features.npz"
        features, risk_scores, labels = extract_features(
            samples, extractor, cache_path=cache_path
        )

        # 替换 NaN
        features = np.nan_to_num(features, nan=0.0)
        splits[split] = (features, risk_scores, labels)

    X_train, y_train, l_train = splits["train"]
    X_val, y_val, l_val = splits["val"]
    X_test, y_test, l_test = splits["test"]

    # ── 2. 特征标准化 ──
    print("\n[2/4] 特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练集: {X_train_scaled.shape[0]} 样本")

    results = {}

    # ── 3. 训练模型 ──
    print("\n[3/4] 训练模型...")

    # GBDT
    if args.model in ["gbr", "both"]:
        gbr_model, gbr_metrics = train_gbr(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        results["GBDT"] = (gbr_model, gbr_metrics)

    # MLP
    if args.model in ["mlp", "both"] and not args.skip_mlp:
        mlp_model, mlp_metrics = train_mlp(
            X_train_scaled, y_train, X_val_scaled, y_val,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
        )
        results["MLP"] = (mlp_model, mlp_metrics)

    # ── 4. 测试集评估 ──
    print("\n[4/4] 测试集评估...")

    best_model = None
    best_name = ""
    best_mae = float("inf")

    for name, (model, metrics) in results.items():
        test_metrics = evaluate_on_test(
            model, X_test_scaled, y_test, l_test, model_name=name
        )
        results[name] = (model, {**metrics, "test": test_metrics})

        if test_metrics["mae"] < best_mae:
            best_mae = test_metrics["mae"]
            best_model = model
            best_name = name

        # 特征重要性（GBDT）
        if name == "GBDT":
            show_feature_importance(model, FEATURE_NAMES)

    # ── 结果汇总 ──
    print("\n" + "=" * 60)
    print("  结果对比")
    print("=" * 60)
    print(f"\n  {'模型':12s} | {'MAE':>6s} | {'Spearman':>10s} | {'Level Acc':>10s} | {'训练时间':>8s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    # LSTM baseline
    print(f"  {'LSTM':12s} | {'12.55':>6s} | {'0.1264':>10s} | {'88.89%':>10s} | {'40 ep':>8s}")

    for name, (model, metrics) in results.items():
        test = metrics.get("test", {})
        print(f"  {name:12s} | {test.get('mae', 0):6.2f} | "
              f"{test.get('spearman', 0):10.4f} | "
              f"{test.get('level_acc', 0):9.2%} | "
              f"{metrics.get('train_time', 0):7.1f}s")

    # 保存最佳模型
    save_model_artifacts(
        best_model, scaler, FEATURE_NAMES,
        {"best_model": best_name, "results": {
            name: {k: v for k, v in m.items() if k != "test"}
            for name, (_, m) in results.items()
        }},
        save_dir=str(Path(__file__).parent.parent / "checkpoints")
    )

    print(f"\n  最佳模型: {best_name} (Test MAE: {best_mae:.2f})")
    print("  完成!")


if __name__ == "__main__":
    main()
