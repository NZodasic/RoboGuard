"""Unit tests for Milestone M3 Multiple-Instance Learning (MIL) anomaly model."""

from pathlib import Path
import numpy as np
import pytest

from sentinel_vl.models.anomaly import MILAnomalyHead, compute_roc_and_pr_auc
from sentinel_vl.training.anomaly_mil import MILTrainer


def test_mil_anomaly_head_predict_and_aggregate() -> None:
    head = MILAnomalyHead(feature_dim=16, top_k_ratio=0.25, seed=123)
    # 8 windows, 16 features
    windows = np.random.randn(8, 16).astype(np.float32)

    scores = head.predict_window_scores(windows)
    assert len(scores) == 8
    for s in scores:
        assert 0.0 <= s <= 1.0

    bag_score = head.aggregate_video_score(scores)
    assert 0.0 <= bag_score <= 1.0
    # Bag score should be equal to mean of top-k (8 * 0.25 = top 2 scores)
    top2 = sorted(scores, reverse=True)[:2]
    assert pytest.approx(bag_score, abs=1e-3) == float(np.mean(top2))


def test_mil_anomaly_head_feature_dim_mismatch() -> None:
    head = MILAnomalyHead(feature_dim=16)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        head.predict_window_scores(np.random.randn(4, 32))


def test_mil_anomaly_head_checkpoint_roundtrip(tmp_path: Path) -> None:
    head = MILAnomalyHead(feature_dim=8, seed=42)
    ckpt_path = tmp_path / "mil_checkpoint.json"

    checksum = head.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()
    assert len(checksum) == 64  # SHA-256

    loaded_head = MILAnomalyHead.load_checkpoint(ckpt_path)
    assert loaded_head.feature_dim == head.feature_dim
    assert pytest.approx(loaded_head.bias, rel=1e-5) == head.bias
    assert np.allclose(loaded_head.weights, head.weights)

    # Test scoring equality
    test_feat = np.ones((2, 8), dtype=np.float32)
    assert head.predict_window_scores(test_feat) == loaded_head.predict_window_scores(test_feat)


def test_compute_roc_and_pr_auc() -> None:
    # Perfect ranking: positive scores > negative scores
    y_true = [1, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.2, 0.1]
    res = compute_roc_and_pr_auc(y_true, y_scores)
    assert res["roc_auc"] == 1.0
    assert res["pr_auc"] == 1.0

    # Inverted ranking
    y_scores_bad = [0.1, 0.2, 0.8, 0.9]
    res_bad = compute_roc_and_pr_auc(y_true, y_scores_bad)
    assert res_bad["roc_auc"] == 0.0


def test_mil_trainer_convergence() -> None:
    np.random.seed(42)
    dim = 8
    head = MILAnomalyHead(feature_dim=dim, seed=42)
    trainer = MILTrainer(head, learning_rate=0.05)

    # Synthetic training bags:
    # Normal bags: features mean = -1.0
    # Abnormal bags: features mean = +1.0
    train_bags = []
    for _ in range(10):
        # Normal (label 0)
        feat_norm = np.random.randn(4, dim).astype(np.float32) - 1.0
        train_bags.append((feat_norm, 0))
        # Anomalous (label 1)
        feat_anom = np.random.randn(4, dim).astype(np.float32) + 1.0
        train_bags.append((feat_anom, 1))

    history = trainer.fit(train_bags, epochs=5)
    assert len(history.train_losses) == 5
    # Loss should decrease over 5 epochs
    assert history.train_losses[-1] < history.train_losses[0]
