"""Uncertainty quantification contracts and evaluation utilities.

Scientific and engineering rules:
- Classifier uncertainty does not quantify generated-language correctness.
- Segment-level scores are weakly supervised and cannot be labeled as calibrated probabilities
  without verified segment-level calibration ground truth.
- Risk is undefined if coverage is zero; do not substitute 0.0 or NaN.
- Review rate must always be reported alongside accepted-decision risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import math


@dataclass
class UQPrediction:
    """Prediction output with uncertainty and abstention metadata."""
    raw_score: float
    calibrated_probability: Optional[float]
    disagreement: Optional[float]
    decision: Literal["accepted_normal", "accepted_anomalous", "review_required"]
    granularity: Literal["video_level_calibrated", "segment_level_uncalibrated"]
    calibration_version: str
    provenance: Literal["real_model", "synthetic_demo"]

    def __post_init__(self) -> None:
        if not (0.0 <= self.raw_score <= 1.0):
            raise ValueError(f"raw_score must be in [0.0, 1.0], got {self.raw_score}")
        if self.calibrated_probability is not None and not (0.0 <= self.calibrated_probability <= 1.0):
            raise ValueError(f"calibrated_probability must be in [0.0, 1.0], got {self.calibrated_probability}")
        if self.disagreement is not None and self.disagreement < 0.0:
            raise ValueError(f"disagreement must be non-negative, got {self.disagreement}")


def compute_risk_and_coverage(
    accepted_mask: List[bool],
    ground_truth: List[int],
    predicted_labels: List[int],
) -> Tuple[float, Optional[float], float]:
    """Computes selective coverage, risk on accepted predictions, and review rate.

    Returns:
        (coverage, risk, review_rate)
        coverage: Fraction of predictions accepted for automated decision [0.0, 1.0].
        risk: Error rate strictly among accepted predictions. None if coverage is 0.0.
        review_rate: Fraction of predictions flagged for human review (1.0 - coverage).
    """
    n = len(accepted_mask)
    if n == 0:
        return 0.0, None, 1.0

    if len(ground_truth) != n or len(predicted_labels) != n:
        raise ValueError("Length mismatch between accepted_mask, ground_truth, and predicted_labels")

    accepted_count = sum(1 for a in accepted_mask if a)
    coverage = accepted_count / float(n)
    review_rate = 1.0 - coverage

    if accepted_count == 0:
        return coverage, None, review_rate

    errors = 0
    for is_acc, yt, yp in zip(accepted_mask, ground_truth, predicted_labels):
        if is_acc and yt != yp:
            errors += 1

    risk = errors / float(accepted_count)
    return coverage, risk, review_rate


def compute_binary_entropy(prob: float) -> float:
    """Computes Shannon entropy for binary probability p."""
    p = max(min(prob, 1.0 - 1e-12), 1e-12)
    return - (p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

