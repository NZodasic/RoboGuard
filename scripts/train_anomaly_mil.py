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
    from sentinel_vl.training.anomaly_mil import train_mil_checkpoint

    print("=" * 65)
    print(" Sentinel-VL Weakly Supervised MIL Training (Milestone M3)")
    print(" Notice: Training on synthetic bags for optimization check.")
    print("=" * 65)

    saved_path, history = train_mil_checkpoint(
        output_checkpoint=output_checkpoint,
        feature_dim=feature_dim,
        epochs=epochs,
        seed=seed,
    )

    print("\nTraining Summary:")
    print(f"  Initial Train Loss: {history.train_losses[0]:.4f}")
    print(f"  Final Train Loss:   {history.train_losses[-1]:.4f}")
    print(f"  Best Val ROC-AUC:   {history.best_roc_auc:.4f}")
    print(f"\nSaved checkpoint to: {saved_path}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentinel-VL MIL head.")
    parser.add_argument("--output", default="checkpoints/mil_head_baseline.json")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    sys.exit(train_mil_model(output_checkpoint=args.output, epochs=args.epochs))
