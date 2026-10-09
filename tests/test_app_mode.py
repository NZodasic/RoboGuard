"""Tests for Streamlit application shell contracts and startup safety."""

import sys
from pathlib import Path

from sentinel_vl.inference.pipeline import SentinelInferencePipeline


def test_app_imports_cleanly_without_model_downloads() -> None:
    # Ensure app module can be imported without triggering downloads or network
    import app.streamlit_app as st_app
    assert hasattr(st_app, "main")
    assert hasattr(st_app, "render_sidebar")


def test_app_pipeline_provenance_is_always_synthetic_demo() -> None:
    output = SentinelInferencePipeline.run_synthetic_demo(
        video_id="demo_lobby_003",
        duration_seconds=40.0,
    )
    assert output["provenance"] == "synthetic_demo"
    assert output["report"].provenance == "synthetic_demo"
    assert "synthetic" in output["report"].description.lower() or "synthetic" in output["provenance"].lower()

