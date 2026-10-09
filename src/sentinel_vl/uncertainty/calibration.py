"""Uncertainty Quantification, Calibration, and Selective Decision Service (Milestone M4).

Scientific requirements:
- Head ensemble over fixed features (not full encoder ensemble).
- Temperature scaling T > 0 fitted strictly on validation/calibration split.
- Predictive disagreement proxy via entropy mutual information.
- Selective prediction returning accepted decisions vs review-required.
- Zero coverage yields risk = None (undefined).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sentinel_vl.models.anomaly import MILAnomalyHead
from sentinel_vl.uncertainty.contracts import (
    UQPrediction,
    compute_binary_entropy,
    compute_risk_and_coverage,
)


class HeadEnsemble:
    """Ensemble of independently initialized anomaly prediction heads over fixed representations."""

    def __init__(
        self,
        heads: Optional[List[MILAnomalyHead]] = None,
        feature_dim: int = 512,
        num_heads: int = 3,
        seed: int = 42,
    ) -> None:
        if heads is not None:
            self.heads = heads
        else:
            self.heads = [
                MILAnomalyHead(feature_dim=feature_dim, seed=seed + i * 17)
                for i in range(num_heads)
            ]

    def predict_bag(self, temporal_features: np.ndarray) -> Tuple[float, float, List[float]]:
        """Predicts bag anomaly score, disagreement, and mean window scores across ensemble.

        Returns:
            (mean_bag_score, disagreement, window_mean_scores)
        """
        all_window_scores = []
        all_bag_scores = []
        head_entropies = []

        for head in self.heads:
            win_scores = head.predict_window_scores(temporal_features)
            bag_s = head.aggregate_video_score(win_scores)
            all_window_scores.append(win_scores)
            all_bag_scores.append(bag_s)
            head_entropies.append(compute_binary_entropy(bag_s))

        # Mean bag score
        mean_bag = float(np.mean(all_bag_scores))

        # Entropy of ensemble mean
        mean_entropy = compute_binary_entropy(mean_bag)

        # Disagreement proxy = H(p_bar) - mean(H(p_m))
        disagreement = max(0.0, float(mean_entropy - np.mean(head_entropies)))

        # Mean window scores
        mean_win_scores = np.mean(all_window_scores, axis=0).tolist()
        mean_win_scores = [round(float(s), 4) for s in mean_win_scores]

        return round(mean_bag, 4), round(disagreement, 4), mean_win_scores


class TemperatureCalibrator:
    """Post-hoc temperature scaling for video-level binary anomaly predictions."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature)

    @staticmethod
    def _logit(p: float) -> float:
        p_clipped = max(min(p, 1.0 - 1e-7), 1e-7)
        return float(np.log(p_clipped / (1.0 - p_clipped)))

    def calibrate(self, raw_score: float) -> float:
        """Applies temperature scaling: p_T = sigmoid(logit(p) / T)."""
        z = self._logit(raw_score)
        scaled_z = np.clip(z / self.temperature, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-scaled_z))
        return round(float(prob), 4)

    def fit(
        self,
        val_raw_scores: List[float],
        val_labels: List[int],
        lr: float = 0.05,
        epochs: int = 50,
    ) -> float:
        """Optimizes temperature T on validation calibration partition using NLL."""
        if len(val_raw_scores) != len(val_labels) or len(val_raw_scores) == 0:
            raise ValueError("Validation scores and labels must be non-empty and of equal length.")

        logits = np.array([self._logit(s) for s in val_raw_scores], dtype=np.float32)
        y = np.array(val_labels, dtype=np.float32)

        # Optimize log_T for unconstrained parameterization (T = exp(log_T) > 0)
        log_t = float(np.log(max(self.temperature, 0.1)))

        for _ in range(epochs):
            t = np.exp(log_t)
            scaled = np.clip(logits / t, -30.0, 30.0)
            sig = 1.0 / (1.0 + np.exp(-scaled))

            # dNLL/dT = sum((sig - y) * (-logits / T^2))
            # dNLL/d(log_T) = dNLL/dT * T = sum((sig - y) * (-logits / T))
            grad_log_t = np.sum((sig - y) * (-logits / t))

            log_t -= lr * float(np.clip(grad_log_t, -5.0, 5.0))

        self.temperature = round(float(np.exp(log_t)), 4)
        return self.temperature

    def compute_ece(
        self,
        calibrated_probs: List[float],
        labels: List[int],
        num_bins: int = 5,
    ) -> float:
        """Computes Expected Calibration Error (ECE)."""
        probs = np.array(calibrated_probs)
        y = np.array(labels)
        n = len(probs)
        if n == 0:
            return 0.0

        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0

        for b in range(num_bins):
            in_bin = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1] if b < num_bins - 1 else probs <= bin_edges[b + 1])
            count = np.sum(in_bin)
            if count > 0:
                bin_acc = float(np.mean(y[in_bin]))
                bin_conf = float(np.mean(probs[in_bin]))
                ece += (count / n) * abs(bin_acc - bin_conf)

        return round(float(ece), 4)


class SelectiveDecisionService:
    """Decides between automated positive/negative decisions and human review abstention."""

    def __init__(
        self,
        high_threshold: float = 0.70,
        low_threshold: float = 0.30,
        max_disagreement: float = 0.12,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_disagreement = max_disagreement

    def decide(
        self,
        calibrated_prob: float,
        disagreement: float,
    ) -> str:
        """Returns 'accepted_normal', 'accepted_anomalous', or 'review_required'."""
        if disagreement > self.max_disagreement:
            return "review_required"

        if calibrated_prob >= self.high_threshold:
            return "accepted_anomalous"
        elif calibrated_prob <= self.low_threshold:
            return "accepted_normal"
        else:
            return "review_required"

    def evaluate_risk_coverage(
        self,
        calibrated_probs: List[float],
        disagreements: List[float],
        labels: List[int],
    ) -> Dict[str, Any]:
        """Evaluates policy coverage, error risk, and review rate."""
        decisions = [
            self.decide(p, d) for p, d in zip(calibrated_probs, disagreements)
        ]
        accepted_mask = [d != "review_required" for d in decisions]
        predicted_binary = [
            1 if d == "accepted_anomalous" else 0 for d in decisions
        ]

        cov, risk, rev_rate = compute_risk_and_coverage(
            accepted_mask=accepted_mask,
            ground_truth=labels,
            predicted_labels=predicted_binary,
        )

        return {
            "coverage": cov,
            "risk": risk,
            "review_rate": rev_rate,
            "total_samples": len(labels),
            "accepted_count": sum(accepted_mask),
            "review_count": len(labels) - sum(accepted_mask),
        }
