"""Offline Multiple-Instance Learning (MIL) training routine for Sentinel-VL (Milestone M3).

Trains an MIL anomaly scoring head on video bags using weakly supervised video labels.
Computes exact analytical gradients for BCE, temporal smoothness, and sparsity objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from sentinel_vl.models.anomaly import MILAnomalyHead, compute_roc_and_pr_auc


@dataclass
class TrainingHistory:
    """Records losses across epochs."""
    epochs: int
    train_losses: List[float]
    val_roc_aucs: List[float]
    best_roc_auc: float


class MILTrainer:
    """Trainer for Multiple-Instance Learning anomaly heads."""

    def __init__(
        self,
        model: MILAnomalyHead,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        lambda_smooth: float = 8e-4,
        lambda_sparse: float = 8e-4,
    ) -> None:
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse

        # Velocity for momentum optimizer
        self.v_w = np.zeros_like(self.model.weights)
        self.v_b = 0.0

    def train_epoch(
        self,
        train_bags: List[Tuple[np.ndarray, int]],  # List of (temporal_features, label)
    ) -> float:
        """Runs one epoch over training video bags with exact multi-objective gradients."""
        epoch_losses: List[float] = []

        # Shuffle bags
        indices = np.random.permutation(len(train_bags))

        for idx in indices:
            features, label = train_bags[idx]
            if len(features) == 0:
                continue

            loss, _, _ = self.model.compute_loss(
                features,
                label,
                lambda_smooth=self.lambda_smooth,
                lambda_sparse=self.lambda_sparse,
            )
            epoch_losses.append(loss)

            logits = np.dot(features, self.model.weights) + self.model.bias
            scores = self.model._sigmoid(logits)

            n = len(scores)
            k = max(self.model.min_k, max(1, int(np.ceil(n * self.model.top_k_ratio))))
            top_indices = np.argsort(-scores)[:k]
            bag_score = np.mean(scores[top_indices])

            # 1. BCE gradient term
            p = float(np.clip(bag_score, 1e-6, 1.0 - 1e-6))
            denom = max(p * (1.0 - p), 1e-3)
            dl_dp = (p - float(label)) / denom

            dl_ds = np.zeros(n, dtype=np.float32)
            for top_i in top_indices:
                dl_ds[top_i] += (1.0 / float(k)) * dl_dp

            # 2. Smoothness gradient term: d/ds_i of sum (s_{t+1} - s_t)^2
            if n > 1 and self.lambda_smooth > 0:
                grad_smooth_s = np.zeros(n, dtype=np.float32)
                grad_smooth_s[0] += 2.0 * (scores[0] - scores[1])
                grad_smooth_s[-1] += 2.0 * (scores[-1] - scores[-2])
                if n > 2:
                    grad_smooth_s[1:-1] += 2.0 * (2.0 * scores[1:-1] - scores[:-2] - scores[2:])
                dl_ds += self.lambda_smooth * grad_smooth_s

            # 3. Sparsity gradient term: d/ds_i of sum s_i
            if self.lambda_sparse > 0:
                dl_ds += self.lambda_sparse * 1.0

            # Chain rule to weights and bias: ds_i / dz_i = s_i * (1 - s_i)
            ds_dz = scores * (1.0 - scores)
            dl_dz = dl_ds * ds_dz

            grad_w = np.dot(features.T, dl_dz)
            grad_b = float(np.sum(dl_dz))

            # Gradient clipping for numerical stability
            grad_w = np.clip(grad_w, -5.0, 5.0)
            grad_b = float(np.clip(grad_b, -5.0, 5.0))

            # Weight decay
            grad_w += self.weight_decay * self.model.weights

            # Momentum step
            self.v_w = self.momentum * self.v_w + self.lr * grad_w
            self.v_b = self.momentum * self.v_b + self.lr * grad_b

            self.model.weights -= self.v_w
            self.model.bias -= self.v_b

        return float(np.mean(epoch_losses)) if epoch_losses else 0.0

    def fit(
        self,
        train_bags: List[Tuple[np.ndarray, int]],
        val_bags: Optional[List[Tuple[np.ndarray, int]]] = None,
        epochs: int = 15,
    ) -> TrainingHistory:
        """Trains model across specified epochs with validation evaluation."""
        train_losses: List[float] = []
        val_aucs: List[float] = []
        best_auc = 0.0

        for ep in range(epochs):
            avg_loss = self.train_epoch(train_bags)
            train_losses.append(round(avg_loss, 4))

            if val_bags:
                y_true = [b[1] for b in val_bags]
                y_preds = [
                    self.model.aggregate_video_score(self.model.predict_window_scores(b[0]))
                    for b in val_bags
                ]
                auc_res = compute_roc_and_pr_auc(y_true, y_preds)
                val_auc = auc_res["roc_auc"]
                val_aucs.append(val_auc)
                if val_auc > best_auc:
                    best_auc = val_auc
            else:
                val_aucs.append(0.0)

        return TrainingHistory(
            epochs=epochs,
            train_losses=train_losses,
            val_roc_aucs=val_aucs,
            best_roc_auc=best_auc,
        )


def train_mil_checkpoint(
    output_checkpoint: str | Path = "checkpoints/mil_head_baseline.json",
    feature_dim: int = 512,
    epochs: int = 20,
    seed: int = 42,
) -> Tuple[str, TrainingHistory]:
    """Callable runner to train and save an MIL anomaly head checkpoint."""
    np.random.seed(seed)
    head = MILAnomalyHead(feature_dim=feature_dim, top_k_ratio=0.2, seed=seed)
    trainer = MILTrainer(head, learning_rate=0.02, weight_decay=1e-4)

    bags = []
    for i in range(15):
        n_windows = np.random.randint(6, 12)
        norm_feat = np.random.randn(n_windows, feature_dim).astype(np.float32) - 0.4
        bags.append((norm_feat, 0))

        anom_feat = np.random.randn(n_windows, feature_dim).astype(np.float32) - 0.4
        peak_idx = np.random.randint(0, n_windows - 2)
        anom_feat[peak_idx : peak_idx + 2] += 1.8
        bags.append((anom_feat, 1))

    train_bags = bags[:20]
    val_bags = bags[20:]

    history = trainer.fit(train_bags, val_bags=val_bags, epochs=epochs)
    head.save_checkpoint(output_checkpoint)
    return str(output_checkpoint), history
