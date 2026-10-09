"""Unit and integration tests for native VideoDecoder."""

import subprocess
import tempfile
from pathlib import Path
import pytest

from sentinel_vl.data.video_decoder import (
    VideoDecoder,
    VideoDecoderError,
    VideoMetadata,
    TimestampedFrame,
)


@pytest.fixture(scope="session")
def sample_video_clip(tmp_path_factory) -> Path:
    """Generates a tiny 2-second 10fps MP4 video clip using ffmpeg for testing."""
    decoder = VideoDecoder()
    if not decoder.is_available:
        pytest.skip("ffmpeg or ffprobe not installed on PATH")

    tmp_dir = tmp_path_factory.mktemp("test_videos")
    clip_path = tmp_dir / "test_clip_10fps.mp4"

    cmd = [
        decoder.ffmpeg_path,
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=2:size=160x120:rate=10",
        "-pix_fmt", "yuv420p",
        str(clip_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        pytest.skip(f"Failed to create test video fixture: {res.stderr.decode()}")

    return clip_path


def test_parse_frame_rate():
    assert VideoDecoder._parse_frame_rate("30/1") == 30.0
    assert VideoDecoder._parse_frame_rate("30000/1001") == 29.97
    assert VideoDecoder._parse_frame_rate("25") == 25.0
    assert VideoDecoder._parse_frame_rate("invalid/str") == 0.0
    assert VideoDecoder._parse_frame_rate("invalid") == 0.0


def test_probe_nonexistent_file():
    decoder = VideoDecoder()
    with pytest.raises(FileNotFoundError):
        decoder.probe(Path("nonexistent_video_path.mp4"))


def test_extract_invalid_fps():
    decoder = VideoDecoder()
    with pytest.raises(ValueError, match="target_fps must be positive"):
        decoder.extract_regular_frames(Path("any.mp4"), target_fps=0.0)


def test_probe_real_video(sample_video_clip):
    decoder = VideoDecoder()
    meta = decoder.probe(sample_video_clip)

    assert isinstance(meta, VideoMetadata)
    assert meta.path == sample_video_clip.resolve()
    assert meta.width == 160
    assert meta.height == 120
    assert abs(meta.fps - 10.0) < 0.1
    assert abs(meta.duration_seconds - 2.0) < 0.2
    assert meta.total_frames in (19, 20, 21)


def test_extract_frames_regular_sampling(sample_video_clip, tmp_path):
    decoder = VideoDecoder()
    out_dir = tmp_path / "frames"

    # Sample at 2 FPS (expecting ~4 frames for a 2s video)
    frames = decoder.extract_regular_frames(
        sample_video_clip,
        target_fps=2.0,
        output_dir=out_dir,
    )

    assert len(frames) >= 3
    for idx, frame in enumerate(frames):
        assert isinstance(frame, TimestampedFrame)
        assert frame.frame_index == idx
        assert frame.image_path.is_file()
        assert frame.timestamp_seconds == pytest.approx(idx * 0.5, abs=0.05)
