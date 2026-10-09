"""Incident reporting and structured explanation schemas for Sentinel-VL.

Strict integrity rules:
- Generated text is labeled "generated_unreviewed" by default; never presented as verified finding.
- Temporal anomaly score is explicitly labeled as uncalibrated or weakly supervised.
- No legal or definitive conclusions ("guilty", "crime confirmed", "safe") permitted.
- Evidence frame IDs must be validated against available frame keys.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import html
import json
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class IncidentReport:
    """Structured report for an analyzed video segment."""
    video_id: str
    interval_seconds: Tuple[float, float]
    temporal_anomaly_score: float
    score_status: str  # e.g., "weakly_supervised_not_segment_calibrated"
    video_decision: Literal["accepted_normal", "accepted_anomalous", "review_required"]
    calibration_version: str
    description: str
    evidence_frame_ids: List[str]
    description_status: Literal["generated_unreviewed", "human_reviewed", "rejected"] = "generated_unreviewed"
    model_version: str = "sentinel-vl-m0"
    provenance: Literal["real_model", "synthetic_demo"] = "synthetic_demo"
    calibrated_video_probability: Optional[float] = None
    ensemble_disagreement: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.video_id or not self.video_id.strip():
            raise ValueError("video_id must not be empty.")
        if len(self.interval_seconds) != 2:
            raise ValueError("interval_seconds must contain exactly [start, end].")
        start, end = self.interval_seconds
        if start < 0 or end <= start:
            raise ValueError(f"Invalid interval: start={start}, end={end}")
        if not (0.0 <= self.temporal_anomaly_score <= 1.0):
            raise ValueError(f"temporal_anomaly_score must be in [0, 1], got {self.temporal_anomaly_score}")

        # Prohibit inflammatory or legally conclusive labels
        prohibited_phrases = ["crime confirmed", "criminal detected", "guilty", "perpetrator identified", "safe guaranteed"]
        desc_lower = self.description.lower()
        for phrase in prohibited_phrases:
            if phrase in desc_lower:
                raise ValueError(f"Prohibited label '{phrase}' found in description. Sentinel-VL generates objective visual event descriptions only.")

    def validate_evidence_frames(self, available_frame_ids: List[str]) -> None:
        """Validates that all evidence frame IDs exist in the provided candidate pool."""
        available_set = set(available_frame_ids)
        missing = [fid for fid in self.evidence_frame_ids if fid not in available_set]
        if missing:
            raise ValueError(f"Evidence frame IDs do not exist in candidate pool: {missing}")

    def to_dict(self, escape_html: bool = False) -> Dict[str, Any]:
        """Converts report to dictionary with optional HTML escaping for dashboard security."""
        data = asdict(self)
        if escape_html:
            data["description"] = html.escape(self.description)
            data["video_id"] = html.escape(self.video_id)
        return data

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

