"""Offline Multiple-Instance Learning (MIL) training routine for Sentinel-VL (Milestone M3).

Trains an MIL anomaly scoring head on video bags using weakly supervised video labels.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    ) -> None:
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        # Velocity for momentum optimizer
        self.v_w = np.zeros_like(self.model.weights)
        self.v_b = 0.0

    def train_epoch(
        self,
        train_bags: List[Tuple[np.ndarray, int]],  # List of (temporal_features, label)
    ) -> float:
        """Runs one epoch over training video bags."""
        epoch_losses: List[float] = []

        # Shuffle bags
        indices = np.random.permutation(len(train_bags))

        for idx in indices:
            features, label = train_bags[idx]
            if len(features) == 0:
                continue

            loss, _, _ = self.model.compute_loss(features, label)
            epoch_losses.append(loss)

            # Compute numerical or analytical gradient
            # Bag score gradient
            logits = np.dot(features, self.model.weights) + self.model.bias
            scores = self.model._sigmoid(logits)

            n = len(scores)
            k = max(self.model.min_k, max(1, int(np.ceil(n * self.model.top_k_ratio))))
            top_indices = np.argsort(-scores)[:k]
            bag_score = np.mean(scores[top_indices])

            # dL/dbag_score
            p = float(np.clip(bag_score, 1e-6, 1.0 - 1e-6))
            denom = max(p * (1.0 - p), 1e-3)
            dl_dp = (p - float(label)) / denom

            # d(bag_score)/d(scores) = 1/k for top-k elements, 0 elsewhere
            grad_w = np.zeros_like(self.model.weights)
            grad_b = 0.0

            for top_i in top_indices:
                s_i = float(scores[top_i])
                ds_dz = s_i * (1.0 - s_i)
                factor = (1.0 / float(k)) * dl_dp * ds_dz
                grad_w += factor * features[top_i]
                grad_b += factor

            # Clip gradient for stability
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
