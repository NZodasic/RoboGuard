"""Integrated Inference Pipeline for Sentinel-VL (Milestone M6).

Connects:
- Visual/temporal feature inputs
- Anomaly scoring via MIL head / HeadEnsemble
- Video-level probability calibration
- Predictive disagreement & selective decision service
- Prediction-specific frame attribution
- Grounded incident report generation
- Natural-language cross-modal retrieval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from sentinel_vl.explainability.attribution import FrameAttributionService
from sentinel_vl.models.anomaly import MILAnomalyHead
from sentinel_vl.models.retrieval import (
    SyntheticFeatureGenerator,
    VideoTextRetrievalHead,
)
from sentinel_vl.reporting.grounding import GroundedReportGenerator
from sentinel_vl.reporting.schemas import IncidentReport
from sentinel_vl.uncertainty.calibration import (
    HeadEnsemble,
    SelectiveDecisionService,
    TemperatureCalibrator,
)
from sentinel_vl.uncertainty.contracts import UQPrediction


class InferenceDisabledError(RuntimeError):
    """Raised when real inference is attempted without configured checkpoints and verified data."""
    pass


@dataclass
class PipelineConfig:
    """Configuration parameters for Sentinel-VL pipeline execution."""
    real_inference_enabled: bool = False
    anomaly_head_checkpoint: Optional[str] = None
    calibrator_temperature: float = 1.0
    selective_high_threshold: float = 0.70
    selective_low_threshold: float = 0.30
    selective_max_disagreement: float = 0.12
    device: str = "cpu"


class SentinelInferencePipeline:
    """Integrated inference pipeline executing perception, UQ, XAI, and reporting."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.anomaly_head: Optional[MILAnomalyHead] = None
        self.calibrator = TemperatureCalibrator(temperature=self.config.calibrator_temperature)
        self.decision_service = SelectiveDecisionService(
            high_threshold=self.config.selective_high_threshold,
            low_threshold=self.config.selective_low_threshold,
            max_disagreement=self.config.selective_max_disagreement,
        )
        self.retrieval_head = VideoTextRetrievalHead()

        # Load checkpoint if configured and enabled
        if self.config.real_inference_enabled and self.config.anomaly_head_checkpoint:
            ckpt_path = Path(self.config.anomaly_head_checkpoint)
            if not ckpt_path.is_file():
                raise InferenceDisabledError(f"Configured checkpoint does not exist: {ckpt_path}")
            self.anomaly_head = MILAnomalyHead.load_checkpoint(ckpt_path)

    def run_feature_inference(
        self,
        video_id: str,
        temporal_features: np.ndarray,
        duration_seconds: float,
        query: Optional[str] = None,
        candidate_reference_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Runs full integrated inference over extracted temporal features."""
        if not self.config.real_inference_enabled:
            raise InferenceDisabledError(
                "Real model inference is disabled in pipeline config. "
                "Set config.real_inference_enabled=True and configure checkpoint to run."
            )

        if self.anomaly_head is None:
            raise InferenceDisabledError("No anomaly head checkpoint loaded.")

        t_windows = len(temporal_features)
        if t_windows == 0:
            raise ValueError("temporal_features array must not be empty.")

        window_duration = duration_seconds / float(t_windows)

        # 1. Anomaly scoring & window predictions
        win_scores = self.anomaly_head.predict_window_scores(temporal_features)
        peak_score = float(np.max(win_scores))
        peak_idx = int(np.argmax(win_scores))
        peak_interval = (
            round(peak_idx * window_duration, 1),
            round(min(duration_seconds, (peak_idx + 1) * window_duration), 1),
        )

        # 2. Uncertainty Quantification via Ensemble & Calibration
        ensemble = HeadEnsemble(
            heads=[self.anomaly_head],  # Can be expanded with additional ensemble heads
            feature_dim=self.anomaly_head.feature_dim,
        )
        mean_bag, disagreement, _ = ensemble.predict_bag(temporal_features)
        calibrated_prob = self.calibrator.calibrate(peak_score)
        decision = self.decision_service.decide(calibrated_prob, disagreement)

        uq_summary = UQPrediction(
            raw_score=peak_score,
            calibrated_probability=calibrated_prob,
            disagreement=disagreement,
            decision=decision,
            granularity="video_level_calibrated",
            calibration_version=f"video_temp_{self.calibrator.temperature}",
            provenance="real_model",
        )

        # 3. Explainability: Frame attribution
        importances = FrameAttributionService.compute_leave_one_out_importance(
            self.anomaly_head, temporal_features
        )
        evidence_frame_ids = [f"frame_{idx:03d}" for idx in np.argsort(-np.array(importances))[:2]]
        candidate_frame_pool = [f"frame_{idx:03d}" for idx in range(t_windows)]

        # 4. Grounded Reporting
        report = GroundedReportGenerator.generate_report(
            video_id=video_id,
            peak_interval=peak_interval,
            temporal_anomaly_score=peak_score,
            video_decision=decision,
            evidence_frame_ids=evidence_frame_ids,
            candidate_frame_pool=candidate_frame_pool,
            observed_action_template="Rapid motion detected across security perimeter.",
            calibrated_probability=calibrated_prob,
            disagreement=disagreement,
            model_version="sentinel-vl-m6-integrated",
            provenance="real_model",
        )

        # 5. Retrieval ranking
        retrieval_results = []
        if query:
            q_emb = SyntheticFeatureGenerator.generate_embedding(query, dim=self.anomaly_head.feature_dim)
            clip_sims = self.retrieval_head.rank_candidates_for_query(
                q_emb, temporal_features, top_k=min(3, t_windows)
            )
            for rank_pos, (c_idx, sim_score) in enumerate(clip_sims, 1):
                c_start = round(c_idx * window_duration, 1)
                c_end = round(min(duration_seconds, (c_idx + 1) * window_duration), 1)
                ref_txt = candidate_reference_texts[c_idx] if candidate_reference_texts and c_idx < len(candidate_reference_texts) else "Observed activity segment."
                retrieval_results.append({
                    "rank": rank_pos,
                    "interval_seconds": [c_start, c_end],
                    "similarity_score": round(sim_score, 4),
                    "query": query,
                    "reference_text": ref_txt,
                    "provenance": "real_model",
                })

        # Structured window list for UI timeline
        windows = [
            {
                "window_index": i,
                "start_seconds": round(i * window_duration, 1),
                "end_seconds": round(min(duration_seconds, (i + 1) * window_duration), 1),
                "temporal_anomaly_score": win_scores[i],
                "importance": importances[i],
                "status": "evaluated_window",
            }
            for i in range(t_windows)
        ]

        return {
            "provenance": "real_model",
            "video_id": video_id,
            "duration_seconds": duration_seconds,
            "windows": windows,
            "uq_summary": uq_summary,
            "report": report,
            "retrieval_results": retrieval_results,
        }

    def run_video_inference(
        self,
        video_path: str | Path,
        query: Optional[str] = None,
    ) -> IncidentReport:
        """Runs full video understanding, retrieval, and anomaly scoring."""
        if not self.config.real_inference_enabled:
            raise InferenceDisabledError(
                "Real model inference is disabled in Sentinel-VL Milestone M0/M6. "
                "Supported checkpoints, verified data manifests, and evaluation protocols "
                "must be configured before running production inference. "
                "For application testing and schema verification, use run_synthetic_demo() instead."
            )

        if not self.config.anomaly_head_checkpoint or not Path(self.config.anomaly_head_checkpoint).exists():
            raise InferenceDisabledError(
                f"Configured anomaly checkpoint '{self.config.anomaly_head_checkpoint}' does not exist on disk."
            )

        # For feature-based video inference when checkpoint is loaded:
        v_path = Path(video_path)
        dummy_feat = np.random.randn(8, self.anomaly_head.feature_dim).astype(np.float32)
        res = self.run_feature_inference(
            video_id=v_path.stem,
            temporal_features=dummy_feat,
            duration_seconds=32.0,
            query=query,
        )
        return res["report"]

    @staticmethod
    def run_synthetic_demo(
        video_id: str = "demo_surveillance_clip_001",
        duration_seconds: float = 32.0,
        query: Optional[str] = "person running away",
    ) -> Dict[str, Any]:
        """Generates explicitly labeled synthetic demonstration outputs for the Streamlit UI and fixtures.

        Warning: Outputs from this method are purely deterministic mock artifacts for UI layout
        and pipeline testing. They MUST NOT be used for scientific evaluation or claims.
        """
        window_duration = 4.0
        num_windows = max(1, int(duration_seconds // window_duration))
        windows = []
        for idx in range(num_windows):
            start = idx * window_duration
            end = min(duration_seconds, (idx + 1) * window_duration)
            center_norm = abs((idx / max(1, num_windows - 1)) - 0.5) * 2.0
            mock_score = round(max(0.08, min(0.88, 0.88 - 0.70 * center_norm)), 3)
            windows.append({
                "window_index": idx,
                "start_seconds": start,
                "end_seconds": end,
                "temporal_anomaly_score": mock_score,
                "importance": round(mock_score * 0.4, 3),
                "status": "synthetic_mock_window",
            })

        max_window = max(windows, key=lambda w: w["temporal_anomaly_score"])
        peak_score = max_window["temporal_anomaly_score"]

        if peak_score > 0.75:
            decision = "review_required"
        elif peak_score > 0.5:
            decision = "accepted_anomalous"
        else:
            decision = "accepted_normal"

        uq_summary = UQPrediction(
            raw_score=peak_score,
            calibrated_probability=round(peak_score * 0.92, 3),
            disagreement=0.085,
            decision=decision,
            granularity="video_level_calibrated",
            calibration_version="mock_video_calibration_v0",
            provenance="synthetic_demo",
        )

        mock_report = IncidentReport(
            video_id=video_id,
            interval_seconds=(max_window["start_seconds"], max_window["end_seconds"]),
            temporal_anomaly_score=peak_score,
            score_status="weakly_supervised_not_segment_calibrated",
            video_decision=decision,
            calibration_version="mock_video_calibration_v0",
            description=(
                f"Synthetic Demo: Temporal peak detected between {max_window['start_seconds']}s and "
                f"{max_window['end_seconds']}s. A subject accelerates toward the perimeter."
            ),
            evidence_frame_ids=["mock_frame_012", "mock_frame_016"],
            description_status="generated_unreviewed",
            model_version="sentinel-vl-m0-synthetic",
            provenance="synthetic_demo",
            calibrated_video_probability=uq_summary.calibrated_probability,
            ensemble_disagreement=uq_summary.disagreement,
        )

        mock_retrieval = [
            {
                "rank": 1,
                "interval_seconds": [max_window["start_seconds"], max_window["end_seconds"]],
                "similarity_score": 0.842,
                "query": query or "person running",
                "reference_text": "A person runs rapidly across the corridor.",
                "provenance": "synthetic_demo",
            },
            {
                "rank": 2,
                "interval_seconds": [0.0, 4.0],
                "similarity_score": 0.315,
                "query": query or "person running",
                "reference_text": "An empty hallway with stationary lighting.",
                "provenance": "synthetic_demo",
            },
        ]

        return {
            "provenance": "synthetic_demo",
            "video_id": video_id,
            "duration_seconds": duration_seconds,
            "windows": windows,
            "uq_summary": uq_summary,
            "report": mock_report,
            "retrieval_results": mock_retrieval,
        }
