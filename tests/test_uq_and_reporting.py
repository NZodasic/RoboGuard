"""Unit tests for uncertainty quantification contracts, risk-coverage, and incident reporting."""

import pytest

from sentinel_vl.reporting.schemas import IncidentReport
from sentinel_vl.uncertainty.contracts import (
    UQPrediction,
    compute_binary_entropy,
    compute_risk_and_coverage,
)


def test_uq_prediction_invariants() -> None:
    uq = UQPrediction(
        raw_score=0.75,
        calibrated_probability=0.70,
        disagreement=0.05,
        decision="accepted_anomalous",
        granularity="video_level_calibrated",
        calibration_version="video_calib_v1",
        provenance="synthetic_demo",
    )
    assert uq.raw_score == 0.75
    assert uq.decision == "accepted_anomalous"

    # Out of range raw score
    with pytest.raises(ValueError, match="raw_score must be in"):
        UQPrediction(
            raw_score=1.5,
            calibrated_probability=None,
            disagreement=None,
            decision="review_required",
            granularity="segment_level_uncalibrated",
            calibration_version="none",
            provenance="synthetic_demo",
        )


def test_risk_and_coverage_standard() -> None:
    accepted_mask = [True, True, True, False, False]
    targets = [1, 0, 1, 1, 0]
    preds = [1, 0, 0, 1, 0]  # On accepted (first 3): targets [1, 0, 1], preds [1, 0, 0] -> 1 error out of 3

    coverage, risk, review_rate = compute_risk_and_coverage(accepted_mask, targets, preds)
    assert coverage == 0.6  # 3 / 5
    assert review_rate == 0.4  # 2 / 5
    assert risk is not None
    assert round(risk, 4) == round(1 / 3, 4)


def test_risk_and_coverage_zero_coverage_undefined_risk() -> None:
    # Scientific requirement: When coverage is zero, risk must be undefined (None), not 0.0 or NaN
    accepted_mask = [False, False, False]
    targets = [1, 0, 1]
    preds = [1, 0, 0]

    coverage, risk, review_rate = compute_risk_and_coverage(accepted_mask, targets, preds)
    assert coverage == 0.0
    assert review_rate == 1.0
    assert risk is None  # Undefined!


def test_incident_report_valid() -> None:
    report = IncidentReport(
        video_id="video_001",
        interval_seconds=(10.0, 15.0),
        temporal_anomaly_score=0.82,
        score_status="weakly_supervised_not_segment_calibrated",
        video_decision="review_required",
        calibration_version="video_calib_v1",
        description="A person moves rapidly across the parking area.",
        evidence_frame_ids=["frame_010", "frame_012"],
        description_status="generated_unreviewed",
        provenance="synthetic_demo",
    )
    assert report.provenance == "synthetic_demo"
    assert report.description_status == "generated_unreviewed"

    # Check evidence frame validation
    report.validate_evidence_frames(["frame_010", "frame_012", "frame_014"])

    # Fail on missing frame
    with pytest.raises(ValueError, match="Evidence frame IDs do not exist in candidate pool"):
        report.validate_evidence_frames(["frame_001", "frame_002"])


def test_incident_report_prohibited_language() -> None:
    # Scientific & ethical requirement: Prohibit inflammatory / legal conclusion claims
    with pytest.raises(ValueError, match="Prohibited label 'guilty' found"):
        IncidentReport(
            video_id="video_001",
            interval_seconds=(0.0, 5.0),
            temporal_anomaly_score=0.9,
            score_status="uncalibrated",
            video_decision="accepted_anomalous",
            calibration_version="none",
            description="Suspect is guilty of theft in the corridor.",
            evidence_frame_ids=[],
        )

