"""
训练器模块

封装完整的训练循环:
- 训练/验证循环
- 学习率调度
- 早停
- 模型保存
- TensorBoard/WandB 日志
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Callable
import time
import json
import numpy as np

from config.settings import CFG_TRAIN, CFG_PATHS
from .metrics import compute_metrics, compute_risk_metrics, print_evaluation_report


class Trainer:
    """
    通用训练器

    Args:
        model: 待训练模型
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        save_dir: 模型保存目录
        task: "classification" | "regression"
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = CFG_TRAIN.DEVICE,
        save_dir: str = str(CFG_PATHS.CHECKPOINTS_DIR),
        task: str = "classification",
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.is_regression = task in ["regression", "multimodal"]

        # 训练状态
        self.current_epoch = 0
        # 回归模式用 -inf（MAE 越低越好，-MAE 总是负数）
        self.best_val_metric = float("-inf") if self.is_regression else 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = CFG_TRAIN.EPOCHS,
        callback: Optional[Callable] = None,
    ) -> Dict:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 总轮数
            callback: 每轮结束后的回调函数

        Returns:
            训练历史字典
        """
        print(f"[Trainer] 开始训练 | 设备: {self.device} | 轮数: {epochs}")
        print(f"[Trainer] 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # ─── 训练阶段 ──────────────────────────────────────
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # ─── 验证阶段 ──────────────────────────────────────
            val_loss, val_metrics = self._validate(val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)

            # ─── 学习率调度 ────────────────────────────────────
            if self.scheduler:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # ─── 日志 ──────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.is_regression:
                val_mae = val_metrics.get("mae", 0)
                val_spearman = val_metrics.get("spearman", 0)
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val MAE: {val_mae:.2f} | "
                    f"Val Spearman: {val_spearman:.4f} | "
                    f"LR: {current_lr:.6f}"
                )
                # 早停：MAE 越低越好（用负值比较）
                val_f1 = -val_mae
            else:
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val F1: {val_metrics.get('f1', 0):.4f} | "
                    f"LR: {current_lr:.6f}"
                )
                val_f1 = val_metrics.get("f1", 0)
            if val_f1 > self.best_val_metric + CFG_TRAIN.MIN_DELTA:
                self.best_val_metric = val_f1
                self.patience_counter = 0
                self._save_checkpoint("best_model.pt")
                if self.is_regression:
                    print(f"    ★ 新最佳模型! MAE: {val_metrics.get('mae', 0):.2f}")
                else:
                    print(f"    ★ 新最佳模型! F1: {val_f1:.4f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= CFG_TRAIN.EARLY_STOP_PATIENCE:
                print(f"\n[Trainer] 早停触发（patience={CFG_TRAIN.EARLY_STOP_PATIENCE}）")
                break

            # ─── 定期保存 ─────────────────────────────────────
            if (epoch + 1) % CFG_TRAIN.SAVE_INTERVAL == 0:
                self._save_checkpoint(f"checkpoint_epoch{epoch+1}.pt")

            # ─── 回调 ────────────────────────────────────────
            if callback:
                callback(epoch, train_loss, val_loss, val_metrics)

        # 保存训练历史
        self._save_history()
        if self.is_regression:
            print(f"\n[Trainer] 训练完成 | 最佳 MAE: {-self.best_val_metric:.2f}")
        else:
            print(f"\n[Trainer] 训练完成 | 最佳 F1: {self.best_val_metric:.4f}")

        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            # 根据 batch 长度判断模式:
            # len=4: multimodal (skeleton, scene, targets, labels)
            # len=3: regression (skeleton, targets, labels) 或 old multimodal
            # len=2: classification (inputs, targets)
            if len(batch) == 4:
                # 多模态模式：(skeleton, scene, targets, labels)
                inputs, scene, targets, labels = batch
                inputs = inputs.to(self.device)
                scene = scene.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs, scene)
            elif len(batch) == 3:
                # 回归模式：(skeleton, targets, labels)
                inputs, targets, labels = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
            else:
                # 分类模式：(inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                labels = None
                outputs = self.model(inputs)

            # 前向传播（回归模式传 labels 用于排序损失）
            if self.is_regression and labels is not None:
                loss = self.criterion(outputs, targets, labels=labels)
            else:
                loss = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        if self.is_regression:
            # ─── 回归模式 ───────────────────────────────────
            all_preds = []
            all_targets = []

            for batch in loader:
                if len(batch) == 4:
                    inputs, scene, targets, labels = batch
                    inputs = inputs.to(self.device)
                    scene = scene.to(self.device)
                    targets = targets.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs, scene)
                    loss = self.criterion(outputs, targets, labels=labels)
                elif len(batch) == 3:
                    inputs, targets, labels = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets, labels=labels)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)

            avg_loss = total_loss / max(num_batches, 1)
            metrics = compute_risk_metrics(y_pred, y_true)

        else:
            # ─── 分类模式 ───────────────────────────────────
            all_preds = []
            all_targets = []
            all_probs = []

            for batch in loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                # 收集预测结果
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs[:, 1].cpu().numpy())  # 正类概率

            y_pred = np.concatenate(all_preds)
            y_true = np.concatenate(all_targets)
            y_prob = np.concatenate(all_probs)

            avg_loss = total_loss / max(num_batches, 1)
            metrics = compute_metrics(y_true, y_pred, y_prob)

        return avg_loss, metrics

    def _save_checkpoint(self, filename: str):
        """保存模型检查点"""
        path = self.save_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
            "history": self.history,
        }, path)

    def _save_history(self):
        """保存训练历史"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
