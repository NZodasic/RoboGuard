"""Unit tests for inference pipeline contracts and synthetic demo safeguards."""

import pytest

from sentinel_vl.inference.pipeline import (
    InferenceDisabledError,
    PipelineConfig,
    SentinelInferencePipeline,
)


def test_real_inference_is_disabled_by_default() -> None:
    pipeline = SentinelInferencePipeline()
    assert pipeline.config.real_inference_enabled is False

    with pytest.raises(InferenceDisabledError, match="Real-video inference is disabled"):
        pipeline.run_video_inference("some_video.mp4")


def test_real_inference_fails_when_checkpoint_missing() -> None:
    cfg = PipelineConfig(
        real_inference_enabled=True,
        anomaly_head_checkpoint="non_existent_weights.pt",
    )
    with pytest.raises(InferenceDisabledError, match="does not exist"):
        SentinelInferencePipeline(config=cfg)


def test_run_synthetic_demo_provenance() -> None:
    demo_output = SentinelInferencePipeline.run_synthetic_demo(
        video_id="test_demo_01",
        duration_seconds=20.0,
        query="person running",
    )

    assert demo_output["provenance"] == "synthetic_demo"
    assert demo_output["video_id"] == "test_demo_01"
    assert len(demo_output["windows"]) > 0

    report = demo_output["report"]
    assert report.provenance == "synthetic_demo"
    assert "Synthetic Demo" in report.description
    assert report.description_status == "generated_unreviewed"

    uq = demo_output["uq_summary"]
    assert uq.provenance == "synthetic_demo"

    # Check retrieval results
    assert len(demo_output["retrieval_results"]) >= 1
    assert demo_output["retrieval_results"][0]["provenance"] == "synthetic_demo"

