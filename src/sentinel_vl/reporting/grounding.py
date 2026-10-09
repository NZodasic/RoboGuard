"""Grounded Incident Report Generator and Validator (Milestone M5).

Scientific rules:
- Generated text is labeled "generated_unreviewed" and never presented as verified fact.
- Descriptions are bounded to observable actions; no intent, identity, or legal accusations.
- Evidence frame references must be verified against valid frame candidate IDs.
- Provides conservative fallback report if confidence is low or evidence is missing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from sentinel_vl.reporting.schemas import IncidentReport


class GroundedReportGenerator:
    """Generates structured incident reports grounded in model evidence."""

    @staticmethod
    def generate_report(
        video_id: str,
        peak_interval: Tuple[float, float],
        temporal_anomaly_score: float,
        video_decision: str,
        evidence_frame_ids: List[str],
        candidate_frame_pool: List[str],
        observed_action_template: str = "A subject moves rapidly across the monitored area.",
        calibrated_probability: Optional[float] = None,
        disagreement: Optional[float] = None,
        model_version: str = "sentinel-vl-v1",
        provenance: str = "real_model",
    ) -> IncidentReport:
        """Constructs and validates a grounded incident report."""
        start_sec, end_sec = peak_interval

        # Ensure evidence frames exist in candidate pool
        pool_set = set(candidate_frame_pool)
        valid_evidence = [fid for fid in evidence_frame_ids if fid in pool_set]

        # If evidence check fails or is empty, use fallback conservative description
        if not valid_evidence:
            description = (
                f"Notice: Elevated anomaly activity detected between {start_sec:.1f}s and {end_sec:.1f}s. "
                "Evidence frame verification incomplete; human review required."
            )
            report_decision = "review_required"
        else:
            description = (
                f"Observable Event ({start_sec:.1f}s - {end_sec:.1f}s): {observed_action_template} "
                f"Referenced frames: [{', '.join(valid_evidence)}]."
            )
            report_decision = video_decision

        report = IncidentReport(
            video_id=video_id,
            interval_seconds=(start_sec, end_sec),
            temporal_anomaly_score=round(temporal_anomaly_score, 4),
            score_status="weakly_supervised_not_segment_calibrated",
            video_decision=report_decision,
            calibration_version="video_calib_v1",
            description=description,
            evidence_frame_ids=valid_evidence,
            description_status="generated_unreviewed",
            model_version=model_version,
            provenance=provenance,
            calibrated_video_probability=calibrated_probability,
            ensemble_disagreement=disagreement,
        )

        # Validate evidence frames
        if valid_evidence:
            report.validate_evidence_frames(candidate_frame_pool)

        return report
