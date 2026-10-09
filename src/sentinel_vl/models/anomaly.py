"""Weakly Supervised Multiple-Instance Learning (MIL) Anomaly Model (Milestone M3).

Formulation:
- Each video is a bag of temporal window representations h_{ij}.
- Window anomaly score: a_{ij} = sigmoid(W^T h_{ij} + b).
- Video bag score: q_i = TopKMean_j(a_{ij}).
- Loss: Binary Cross-Entropy on q_i + Temporal Smoothness Regularization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sentinel_vl.models.base import BaseAnomalyHead


class MILAnomalyHead(BaseAnomalyHead):
    """Multiple-Instance Learning (MIL) anomaly scoring head."""

    def __init__(
        self,
        feature_dim: int = 512,
        top_k_ratio: float = 0.2,
        min_k: int = 1,
        seed: Optional[int] = 42,
    ) -> None:
        self.feature_dim = feature_dim
        self.top_k_ratio = top_k_ratio
        self.min_k = min_k

        rng = np.random.RandomState(seed)
        # Xavier/Glorot uniform initialization
        scale = np.sqrt(6.0 / (feature_dim + 1))
        self.weights = rng.uniform(-scale, scale, size=(feature_dim,)).astype(np.float32)
        self.bias = float(rng.uniform(-0.1, 0.1))

    @staticmethod
    def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    def predict_window_scores(self, temporal_features: np.ndarray) -> List[float]:
        """Computes anomaly score a_{ij} for each window in a video bag.

        Args:
            temporal_features: Shape (T, D)

        Returns:
            List of T anomaly scores in [0.0, 1.0].
        """
        if temporal_features.ndim != 2:
            raise ValueError(f"Expected 2D array (T, D), got {temporal_features.shape}")
        if temporal_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, got {temporal_features.shape[1]}"
            )

        logits = np.dot(temporal_features, self.weights) + self.bias
        scores = self._sigmoid(logits)
        return [round(float(s), 4) for s in scores]

    def aggregate_video_score(
        self,
        window_scores: List[float],
        top_k: Optional[int] = None,
    ) -> float:
        """Aggregates window-level anomaly scores into video-level prediction q_i."""
        if not window_scores:
            return 0.0

        n = len(window_scores)
        if top_k is None:
            k = max(self.min_k, int(math_ceil(n * self.top_k_ratio)))
        else:
            k = max(1, min(top_k, n))

        sorted_scores = sorted(window_scores, reverse=True)
        top_k_scores = sorted_scores[:k]
        return round(float(np.mean(top_k_scores)), 4)

    def compute_loss(
        self,
        window_features: np.ndarray,
        video_label: int,
        lambda_smooth: float = 8e-4,
        lambda_sparse: float = 8e-4,
    ) -> Tuple[float, float, float]:
        """Computes MIL loss, smoothness regularization, and total objective."""
        logits = np.dot(window_features, self.weights) + self.bias
        window_scores = self._sigmoid(logits)

        # Bag score Top-k
        n = len(window_scores)
        k = max(self.min_k, max(1, int(np.ceil(n * self.top_k_ratio))))
        top_indices = np.argsort(-window_scores)[:k]
        bag_score = np.mean(window_scores[top_indices])

        # Binary Cross-Entropy
        p = float(np.clip(bag_score, 1e-7, 1.0 - 1e-7))
        bce_loss = - (float(video_label) * np.log(p) + float(1 - video_label) * np.log(1.0 - p))

        # Temporal smoothness loss: (a_{t+1} - a_t)^2
        if n > 1:
            diffs = window_scores[1:] - window_scores[:-1]
            smooth_loss = float(np.sum(diffs ** 2))
        else:
            smooth_loss = 0.0

        # Sparsity penalty (normal videos should have few peaks)
        sparse_loss = float(np.sum(window_scores))

        total_loss = float(bce_loss + lambda_smooth * smooth_loss + lambda_sparse * sparse_loss)
        return total_loss, float(bce_loss), smooth_loss

    def save_checkpoint(self, checkpoint_path: str | Path) -> str:
        """Saves model weights and metadata to a structured JSON checkpoint."""
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": "MILAnomalyHead",
            "version": "1.0.0",
            "feature_dim": self.feature_dim,
            "top_k_ratio": self.top_k_ratio,
            "min_k": self.min_k,
            "bias": self.bias,
            "weights": self.weights.tolist(),
        }

        content_bytes = json.dumps(payload, indent=2).encode("utf-8")
        checksum = hashlib.sha256(content_bytes).hexdigest()
        payload["sha256"] = checksum

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return checksum

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str | Path) -> MILAnomalyHead:
        """Loads model weights and metadata from a JSON checkpoint."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("model_type") != "MILAnomalyHead":
            raise ValueError(f"Invalid model_type in checkpoint: {data.get('model_type')}")

        head = cls(
            feature_dim=int(data["feature_dim"]),
            top_k_ratio=float(data.get("top_k_ratio", 0.2)),
            min_k=int(data.get("min_k", 1)),
        )
        head.bias = float(data["bias"])
        head.weights = np.array(data["weights"], dtype=np.float32)
        return head


def math_ceil(x: float) -> int:
    import math
    return math.ceil(x)


def compute_roc_and_pr_auc(
    y_true: List[int],
    y_scores: List[float],
) -> Dict[str, float]:
    """Computes ROC-AUC and PR-AUC using trapezoidal integration."""
    if len(y_true) != len(y_scores) or len(y_true) == 0:
        return {"roc_auc": 0.5, "pr_auc": 0.0}

    y_t = np.array(y_true)
    y_s = np.array(y_scores)

    pos_count = np.sum(y_t == 1)
    neg_count = np.sum(y_t == 0)

    if pos_count == 0 or neg_count == 0:
        return {"roc_auc": 0.5, "pr_auc": 0.0}

    # Sort by scores descending
    desc_order = np.argsort(-y_s)
    y_t_sorted = y_t[desc_order]

    # ROC calculation
    tps = np.cumsum(y_t_sorted == 1)
    fps = np.cumsum(y_t_sorted == 0)

    tpr = tps / pos_count
    fpr = fps / neg_count

    # Add origin
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]

    # Trapezoidal rule for ROC AUC
    roc_auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))

    # Precision-Recall
    precisions = tps / (tps + fps)
    recalls = tpr[1:]
    precisions = np.r_[1.0, precisions]
    recalls = np.r_[0.0, recalls]

    pr_auc = float(np.trapezoid(precisions, recalls)) if hasattr(np, "trapezoid") else float(np.trapz(precisions, recalls))

    return {
        "roc_auc": round(abs(roc_auc), 4),
        "pr_auc": round(abs(pr_auc), 4),
    }
