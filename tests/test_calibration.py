"""Unit tests for Milestone M4 Uncertainty Quantification and Calibration."""

import numpy as np
import pytest

from sentinel_vl.uncertainty.calibration import (
    HeadEnsemble,
    SelectiveDecisionService,
    TemperatureCalibrator,
)


def test_head_ensemble_predictions_and_disagreement() -> None:
    ens = HeadEnsemble(feature_dim=16, num_heads=3, seed=42)
    feat = np.random.randn(6, 16).astype(np.float32)

    mean_bag, disagreement, win_scores = ens.predict_bag(feat)
    assert 0.0 <= mean_bag <= 1.0
    assert disagreement >= 0.0
    assert len(win_scores) == 6


def test_temperature_calibrator_positive_constraint() -> None:
    with pytest.raises(ValueError, match="Temperature must be strictly positive"):
        TemperatureCalibrator(temperature=-0.5)

    with pytest.raises(ValueError, match="Temperature must be strictly positive"):
        TemperatureCalibrator(temperature=0.0)


def test_temperature_calibrator_fit_and_ece() -> None:
    calib = TemperatureCalibrator(temperature=1.0)
    # Synthetic validation set: overconfident predictions
    val_scores = [0.95, 0.90, 0.85, 0.80, 0.20, 0.15, 0.10, 0.05]
    val_labels = [1, 1, 0, 1, 0, 0, 1, 0]

    fitted_t = calib.fit(val_scores, val_labels, epochs=30)
    assert fitted_t > 0.0
    assert calib.temperature > 0.0

    calibrated_probs = [calib.calibrate(s) for s in val_scores]
    ece = calib.compute_ece(calibrated_probs, val_labels, num_bins=4)
    assert 0.0 <= ece <= 1.0


def test_selective_decision_service_thresholds() -> None:
    service = SelectiveDecisionService(high_threshold=0.75, low_threshold=0.25, max_disagreement=0.10)

    # Confident anomaly with low disagreement
    assert service.decide(0.85, 0.03) == "accepted_anomalous"

    # Confident normal with low disagreement
    assert service.decide(0.15, 0.04) == "accepted_normal"

    # Ambiguous probability
    assert service.decide(0.50, 0.02) == "review_required"

    # High probability but high disagreement -> must abstain!
    assert service.decide(0.88, 0.18) == "review_required"


def test_selective_decision_zero_coverage_undefined_risk() -> None:
    # All cases fall into review zone
    service = SelectiveDecisionService(high_threshold=0.99, low_threshold=0.01, max_disagreement=0.01)
    probs = [0.5, 0.6, 0.4]
    disagreements = [0.05, 0.05, 0.05]
    labels = [1, 0, 0]

    res = service.evaluate_risk_coverage(probs, disagreements, labels)
    assert res["coverage"] == 0.0
    assert res["review_rate"] == 1.0
    assert res["risk"] is None  # Undefined
    assert res["accepted_count"] == 0
    assert res["review_count"] == 3
