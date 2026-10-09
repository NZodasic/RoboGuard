"""Unit tests for data schemas, interval bounds, and manifest management."""

from pathlib import Path
import pytest

from sentinel_vl.data.manifest import ManifestManager
from sentinel_vl.data.schemas import (
    AnomalyWindow,
    CaptionSegment,
    DataManifest,
    ValidationError,
    VideoRecord,
)


def test_video_record_valid(sample_video_record: VideoRecord) -> None:
    assert sample_video_record.video_id == "video_test_001"
    assert sample_video_record.duration_seconds == 60.0
    assert sample_video_record.decoding_status == "unverified"


def test_video_record_invalid_bounds() -> None:
    with pytest.raises(ValidationError, match="video_id must be a non-empty string"):
        VideoRecord(video_id="", source_path="path/to/vid.mp4")

    with pytest.raises(ValidationError, match="source_path must be a non-empty string"):
        VideoRecord(video_id="vid_1", source_path="   ")

    with pytest.raises(ValidationError, match="duration_seconds must be positive"):
        VideoRecord(video_id="vid_1", source_path="path/to/vid.mp4", duration_seconds=-5.0)

    with pytest.raises(ValidationError, match="decoding_status must be one of"):
        VideoRecord(video_id="vid_1", source_path="path/to/vid.mp4", decoding_status="invalid_status")


def test_caption_segment_invariants() -> None:
    # Valid
    cap = CaptionSegment("c1", "vid_1", 0.0, 5.0, "Valid caption")
    assert cap.start_seconds == 0.0
    assert cap.end_seconds == 5.0

    # Negative start time
    with pytest.raises(ValidationError, match="start_seconds must be non-negative"):
        CaptionSegment("c1", "vid_1", -1.0, 5.0, "Negative start")

    # Inverted interval (end <= start)
    with pytest.raises(ValidationError, match="end_seconds .* must be strictly greater than start_seconds"):
        CaptionSegment("c1", "vid_1", 5.0, 5.0, "Zero duration")

    with pytest.raises(ValidationError, match="end_seconds .* must be strictly greater than start_seconds"):
        CaptionSegment("c1", "vid_1", 10.0, 5.0, "Inverted interval")

    # Empty caption
    with pytest.raises(ValidationError, match="caption must be a non-empty string"):
        CaptionSegment("c1", "vid_1", 0.0, 5.0, "   ")


def test_anomaly_window_invariants() -> None:
    win = AnomalyWindow("w1", "vid_1", 2.0, 6.0, label=1)
    assert win.label == 1

    # Invalid label (must be 0, 1, or None)
    with pytest.raises(ValidationError, match="label must be 0, 1, or None"):
        AnomalyWindow("w1", "vid_1", 2.0, 6.0, label=2)

    with pytest.raises(ValidationError, match="end_seconds .* must be strictly greater than start_seconds"):
        AnomalyWindow("w1", "vid_1", 10.0, 2.0)


def test_manifest_cross_reference_integrity() -> None:
    video = VideoRecord(video_id="vid_100", source_path="test.mp4", duration_seconds=30.0, decoding_status="ok")
    caption_valid = CaptionSegment("c1", "vid_100", 0.0, 20.0, "A valid description")
    caption_unknown_vid = CaptionSegment("c2", "vid_missing", 0.0, 10.0, "Unknown video")
    caption_exceeds_duration = CaptionSegment("c3", "vid_100", 0.0, 35.0, "Exceeds duration")

    # Unknown video in caption segment
    m_bad_ref = DataManifest(videos={"vid_100": video}, captions=[caption_unknown_vid])
    with pytest.raises(ValidationError, match="references unknown video 'vid_missing'"):
        m_bad_ref.validate_integrity()

    # Caption interval exceeds video duration
    m_bad_time = DataManifest(videos={"vid_100": video}, captions=[caption_exceeds_duration])
    with pytest.raises(ValidationError, match="exceeds video duration"):
        m_bad_time.validate_integrity()

    # Valid manifest
    m_good = DataManifest(videos={"vid_100": video}, captions=[caption_valid])
    m_good.validate_integrity()  # Should not raise


def test_manifest_manager_save_and_load(sample_manifest: DataManifest, tmp_path: Path) -> None:
    out_file = tmp_path / "saved_manifest.json"
    ManifestManager.save_manifest(sample_manifest, out_file)
    assert out_file.exists()

    loaded = ManifestManager.load_manifest(out_file)
    assert loaded.manifest_version == sample_manifest.manifest_version
    assert len(loaded.videos) == len(sample_manifest.videos)
    assert len(loaded.captions) == len(sample_manifest.captions)
    assert len(loaded.excluded_videos) == len(sample_manifest.excluded_videos)


def test_manifest_disk_file_verification(sample_manifest: DataManifest, tmp_path: Path) -> None:
    # Point video path to a real temporary file
    real_video_file = tmp_path / "actual_video.mp4"
    real_video_file.touch()

    sample_manifest.videos["video_test_001"].source_path = str(real_video_file)
    sample_manifest.videos["video_missing_002"] = VideoRecord(
        video_id="video_missing_002",
        source_path=str(tmp_path / "does_not_exist.mp4"),
        decoding_status="unverified",
    )

    found, missing = ManifestManager.verify_files_on_disk(sample_manifest)
    assert found == 1
    assert missing == ["video_missing_002"]

