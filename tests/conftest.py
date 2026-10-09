"""Lightweight test fixtures for Sentinel-VL unit tests.

These fixtures provide isolated, deterministic in-memory and temporary-file test cases
without requiring real video downloads or network access.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Set
import pytest

# Ensure repository root is on sys.path for test imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sentinel_vl.data.schemas import (
    AnomalyWindow,
    CaptionSegment,
    DataManifest,
    SplitPartition,
    VideoRecord,
)


@pytest.fixture
def sample_video_record() -> VideoRecord:
    return VideoRecord(
        video_id="video_test_001",
        source_path="data/videos/video_test_001.mp4",
        duration_seconds=60.0,
        decoding_status="unverified",
        fps=30.0,
        resolution="1920x1080",
    )


@pytest.fixture
def sample_caption_segment() -> CaptionSegment:
    return CaptionSegment(
        segment_id="cap_001",
        video_id="video_test_001",
        start_seconds=10.0,
        end_seconds=20.0,
        caption="A person walks toward the front lobby door.",
        provenance="uca_human_reference",
    )


@pytest.fixture
def sample_anomaly_window() -> AnomalyWindow:
    return AnomalyWindow(
        window_id="win_001",
        video_id="video_test_001",
        start_seconds=12.0,
        end_seconds=16.0,
        label=1,
        label_provenance="temporal_gt",
    )


@pytest.fixture
def sample_manifest(
    sample_video_record: VideoRecord,
    sample_caption_segment: CaptionSegment,
    sample_anomaly_window: AnomalyWindow,
) -> DataManifest:
    return DataManifest(
        manifest_version="1.0.0",
        videos={sample_video_record.video_id: sample_video_record},
        captions=[sample_caption_segment],
        anomaly_windows=[sample_anomaly_window],
        excluded_videos={"video_corrupt_999": "File corrupted during decoding"},
    )


@pytest.fixture
def clean_partition() -> SplitPartition:
    return SplitPartition(
        name="test_protocol",
        train_video_ids={"vid_01", "vid_02", "vid_03"},
        val_video_ids={"vid_04", "vid_05"},
        test_video_ids={"vid_06", "vid_07"},
    )


@pytest.fixture
def leaking_partition() -> SplitPartition:
    return SplitPartition(
        name="leaking_protocol",
        train_video_ids={"vid_01", "vid_02", "vid_03"},
        val_video_ids={"vid_03", "vid_04"},  # vid_03 leaked!
        test_video_ids={"vid_05"},
    )


@pytest.fixture
def temp_manifest_file(sample_manifest: DataManifest, tmp_path: Path) -> Path:
    manifest_path = tmp_path / "test_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(sample_manifest.to_dict(), f, indent=2)
    return manifest_path
