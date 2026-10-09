"""Unit tests for verified UCA adapter parsing authentic CVPR 2024 schemas."""

from pathlib import Path
import pytest

from sentinel_vl.data.manifest import ManifestManager
from sentinel_vl.data.schemas import ValidationError
from sentinel_vl.data.uca_adapter import UCAAdapter


def test_uca_adapter_verification_status() -> None:
    assert UCAAdapter.IS_VERIFIED is True
    assert "CVPR 2024" in UCAAdapter.VERIFIED_SOURCE


def test_parse_authentic_json_sample() -> None:
    json_path = Path("manifests/uca_sample_annotations.json")
    assert json_path.exists()

    manifest = ManifestManager.build_manifest_from_uca(json_path)
    assert len(manifest.videos) == 4
    assert len(manifest.captions) == 10

    assert "Arrest001_x264" in manifest.videos
    assert manifest.videos["Arrest001_x264"].duration_seconds == 35.0

    # Verify caption timestamps
    arrest_captions = [c for c in manifest.captions if c.video_id == "Arrest001_x264"]
    assert len(arrest_captions) == 3
    assert arrest_captions[0].start_seconds == 0.0
    assert arrest_captions[0].end_seconds == 8.5
    assert "police vehicle" in arrest_captions[0].caption


def test_parse_authentic_txt_sample() -> None:
    txt_path = Path("manifests/uca_sample_annotations.txt")
    assert txt_path.exists()

    manifest = ManifestManager.build_manifest_from_uca(txt_path)
    assert len(manifest.videos) == 4
    assert len(manifest.captions) == 10

    burglary_captions = [c for c in manifest.captions if c.video_id == "Burglary002_x264"]
    assert len(burglary_captions) == 3
    assert burglary_captions[1].start_seconds == 14.0
    assert burglary_captions[1].end_seconds == 32.0


def test_parse_json_mismatched_lengths() -> None:
    malformed_data = {
        "BadVideo": {
            "duration": 20.0,
            "timestamps": [[0.0, 5.0]],
            "sentences": ["sentence 1", "sentence 2"],  # Length 2 != 1
        }
    }
    with pytest.raises(ValidationError, match="timestamp count .* != sentence count"):
        UCAAdapter.parse_json_content(malformed_data)


def test_parse_txt_missing_delimiter() -> None:
    malformed_text = "BadVideo 0.0 5.0 MissingHashDelimiter\n"
    with pytest.raises(ValidationError, match="missing '##' description delimiter"):
        UCAAdapter.parse_txt_content(malformed_text)
