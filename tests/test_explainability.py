"""Unit tests for Milestone M5 Explainability (XAI) and Grounded Reporting."""

import numpy as np
import pytest

from sentinel_vl.explainability.attribution import FrameAttributionService
from sentinel_vl.models.anomaly import MILAnomalyHead
from sentinel_vl.reporting.grounding import GroundedReportGenerator


def test_leave_one_out_importance() -> None:
    head = MILAnomalyHead(feature_dim=8, top_k_ratio=0.25, seed=42)
    # 8 windows: align window index 3 with weight direction to produce peak anomaly
    feat = np.zeros((8, 8), dtype=np.float32)
    feat[3] = head.weights * 5.0  # Strongly aligned with weights

    importances = FrameAttributionService.compute_leave_one_out_importance(head, feat)
    assert len(importances) == 8
    # Window index 3 should have highest importance
    assert np.argmax(importances) == 3
    assert importances[3] > 0.0


def test_attribution_faithfulness_versus_random() -> None:
    head = MILAnomalyHead(feature_dim=8, top_k_ratio=0.25, seed=42)
    feat = np.random.randn(10, 8).astype(np.float32)
    # Inject anomalous burst at index 4 and 5
    feat[4:6] += 2.5

    result = FrameAttributionService.evaluate_removal_versus_random(head, feat, remove_k=2)
    assert "top_k_drop" in result
    assert "mean_random_drop" in result
    assert "faithfulness_ratio" in result
    assert result["faithful"] is True


def test_grounded_report_generator_with_valid_evidence() -> None:
    pool = ["frame_001", "frame_002", "frame_003", "frame_004"]
    report = GroundedReportGenerator.generate_report(
        video_id="test_video_10",
        peak_interval=(12.0, 16.0),
        temporal_anomaly_score=0.82,
        video_decision="accepted_anomalous",
        evidence_frame_ids=["frame_002", "frame_003"],
        candidate_frame_pool=pool,
        calibrated_probability=0.78,
        disagreement=0.04,
    )

    assert report.video_decision == "accepted_anomalous"
    assert report.evidence_frame_ids == ["frame_002", "frame_003"]
    assert report.description_status == "generated_unreviewed"
    assert "Observable Event" in report.description


def test_grounded_report_generator_fallback_on_invalid_evidence() -> None:
    pool = ["frame_001", "frame_002"]
    # Evidence frames are outside candidate pool!
    report = GroundedReportGenerator.generate_report(
        video_id="test_video_10",
        peak_interval=(10.0, 15.0),
        temporal_anomaly_score=0.85,
        video_decision="accepted_anomalous",
        evidence_frame_ids=["frame_missing_99"],
        candidate_frame_pool=pool,
    )

    # Must fall back to conservative review_required notice
    assert report.video_decision == "review_required"
    assert report.evidence_frame_ids == []
    assert "verification incomplete" in report.description.lower()
