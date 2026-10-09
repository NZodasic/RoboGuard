"""Script to train and save a weakly supervised MIL anomaly head checkpoint (Milestone M3)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

from sentinel_vl.models.anomaly import MILAnomalyHead, compute_roc_and_pr_auc
from sentinel_vl.training.anomaly_mil import MILTrainer


def train_mil_model(
    output_checkpoint: str = "checkpoints/mil_head_baseline.json",
    feature_dim: int = 512,
    epochs: int = 20,
    seed: int = 42,
) -> int:
    """Trains MIL anomaly model and saves versioned checkpoint."""
    print("=" * 65)
    print(" Sentinel-VL Weakly Supervised MIL Training (Milestone M3)")
    print("=" * 65)

    np.random.seed(seed)
    head = MILAnomalyHead(feature_dim=feature_dim, top_k_ratio=0.2, seed=seed)
    trainer = MILTrainer(head, learning_rate=0.02, weight_decay=1e-4)

    # Generate development bags (15 normal bags, 15 anomaly bags)
    bags = []
    for i in range(15):
        # Normal video bag
        n_windows = np.random.randint(6, 12)
        norm_feat = np.random.randn(n_windows, feature_dim).astype(np.float32) - 0.4
        bags.append((norm_feat, 0))

        # Abnormal video bag (contains a cluster of anomalous windows)
        anom_feat = np.random.randn(n_windows, feature_dim).astype(np.float32) - 0.4
        peak_idx = np.random.randint(0, n_windows - 2)
        anom_feat[peak_idx : peak_idx + 2] += 1.8  # localized anomaly
        bags.append((anom_feat, 1))

    train_bags = bags[:20]
    val_bags = bags[20:]

    print(f"Training on {len(train_bags)} video bags, validating on {len(val_bags)} bags...")
    history = trainer.fit(train_bags, val_bags=val_bags, epochs=epochs)

    print("\nTraining Summary:")
    print(f"  Initial Train Loss: {history.train_losses[0]:.4f}")
    print(f"  Final Train Loss:   {history.train_losses[-1]:.4f}")
    print(f"  Best Val ROC-AUC:   {history.best_roc_auc:.4f}")

    checksum = head.save_checkpoint(output_checkpoint)
    print(f"\nSaved checkpoint to: {output_checkpoint}")
    print(f"Checksum (SHA-256):  {checksum}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentinel-VL MIL head.")
    parser.add_argument("--output", default="checkpoints/mil_head_baseline.json")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    sys.exit(train_mil_model(output_checkpoint=args.output, epochs=args.epochs))
