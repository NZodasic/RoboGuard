"""Unit tests for integrated inference pipeline (Milestone M6)."""

from pathlib import Path
import numpy as np
import pytest

from sentinel_vl.inference.pipeline import (
    InferenceDisabledError,
    PipelineConfig,
    SentinelInferencePipeline,
)
from sentinel_vl.models.anomaly import MILAnomalyHead


def test_integrated_pipeline_execution(tmp_path: Path) -> None:
    # 1. Create and save a small checkpoint
    ckpt_path = tmp_path / "test_mil.json"
    head = MILAnomalyHead(feature_dim=16, top_k_ratio=0.25, seed=42)
    head.save_checkpoint(ckpt_path)

    # 2. Configure pipeline with real inference enabled
    config = PipelineConfig(
        real_inference_enabled=True,
        anomaly_head_checkpoint=str(ckpt_path),
        calibrator_temperature=0.8,
        selective_high_threshold=0.70,
        selective_low_threshold=0.30,
        selective_max_disagreement=0.15,
    )
    pipeline = SentinelInferencePipeline(config)

    # 3. Run integrated inference over 8 temporal windows
    features = np.random.randn(8, 16).astype(np.float32)
    output = pipeline.run_feature_inference(
        video_id="test_video_integrated",
        temporal_features=features,
        duration_seconds=32.0,
        query="person running across hallway",
    )

    assert output["provenance"] == "real_model"
    assert output["video_id"] == "test_video_integrated"
    assert len(output["windows"]) == 8
    assert "importance" in output["windows"][0]

    report = output["report"]
    assert report.provenance == "real_model"
    assert report.video_id == "test_video_integrated"
    assert report.video_decision in ("accepted_normal", "accepted_anomalous", "review_required")
    assert len(report.evidence_frame_ids) <= 2

    # UQ summary
    uq = output["uq_summary"]
    assert uq.provenance == "real_model"
    assert uq.calibrated_probability is not None

    # Retrieval
    retrieval = output["retrieval_results"]
    assert len(retrieval) > 0
    assert retrieval[0]["provenance"] == "real_model"


def test_integrated_pipeline_disabled_fails() -> None:
    config = PipelineConfig(real_inference_enabled=False)
    pipeline = SentinelInferencePipeline(config)

    with pytest.raises(InferenceDisabledError, match="Real model inference is disabled in pipeline config"):
        pipeline.run_feature_inference(
            video_id="test_fail",
            temporal_features=np.ones((4, 16)),
            duration_seconds=16.0,
        )
