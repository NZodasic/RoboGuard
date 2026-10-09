"""Unit tests for the Sentinel-VL CLI commands."""

import json
from pathlib import Path
import pytest
from sentinel_vl.cli import main


def test_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Sentinel-VL: Uncertainty-Aware and Explainable" in captured.out


def test_cli_version(capsys) -> None:
    # --version calls parser.exit(), but in standard argparse it exits with 0
    import pytest
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_cli_status(capsys) -> None:
    code = main(["status"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Sentinel-VL Environment & Compute Status" in captured.out
    assert "Python:" in captured.out


def test_cli_demo_summary(capsys) -> None:
    code = main(["demo-summary", "--video-id", "test_cli_clip", "--duration", "24.0"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Sentinel-VL Synthetic Demo Output [PROVENANCE: SYNTHETIC_DEMO]" in captured.out
    assert "test_cli_clip" in captured.out


def test_cli_demo_summary_json(capsys) -> None:
    code = main(["demo-summary", "--video-id", "test_cli_clip", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["provenance"] == "synthetic_demo"
    assert payload["video_id"] == "test_cli_clip"


def test_cli_validate_manifest(tmp_path: Path, capsys) -> None:
    # Create valid manifest json
    manifest_data = {
        "manifest_version": "1.0.0",
        "created_at": "2026-10-09T00:00:00Z",
        "videos": {
            "v1": {
                "video_id": "v1",
                "source_path": "path.mp4",
                "duration_seconds": 10.0,
                "decoding_status": "ok",
            }
        },
        "captions": [],
        "anomaly_windows": [],
        "excluded_videos": {},
    }
    m_file = tmp_path / "valid_manifest.json"
    with open(m_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    code = main(["validate-manifest", "--manifest-path", str(m_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert "VALID" in captured.out


def test_cli_audit_splits_clean(tmp_path: Path, capsys) -> None:
    splits_data = {
        "name": "custom_eval",
        "train": ["v1", "v2"],
        "val": ["v3"],
        "test": ["v4"],
    }
    s_file = tmp_path / "clean_splits.json"
    with open(s_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f)

    code = main(["audit-splits", "--splits-file", str(s_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert "PASSED" in captured.out


def test_cli_audit_splits_leak(tmp_path: Path, capsys) -> None:
    splits_data = {
        "name": "leaking_eval",
        "train": ["v1", "v2"],
        "val": ["v2"],  # Leak!
        "test": ["v4"],
    }
    s_file = tmp_path / "leaking_splits.json"
    with open(s_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f)

    code = main(["audit-splits", "--splits-file", str(s_file)])
    assert code == 1
    captured = capsys.readouterr()
    assert "FAILED (LEAKAGE DETECTED)" in captured.out
