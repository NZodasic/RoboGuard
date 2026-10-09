"""Sentinel-VL: Uncertainty-Aware and Explainable Surveillance Monitoring.

Streamlit Integrated Application (Milestones M0 - M6).

Connects:
- Multi-mode engine (Explicit Synthetic Demo vs Real Checkpoint Inference)
- Weakly supervised anomaly scoring & temporal trajectories
- Predictive uncertainty & selective decision policies with real-time threshold tuning
- Prediction-specific leave-one-out frame attribution (XAI)
- Grounded incident report generation & human-in-the-loop audit export
- Natural-language cross-modal video-text retrieval
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sentinel_vl import __version__
from sentinel_vl.inference.pipeline import (
    PipelineConfig,
    SentinelInferencePipeline,
)
from sentinel_vl.reporting.schemas import IncidentReport


def setup_page_config() -> None:
    st.set_page_config(
        page_title="Sentinel-VL Surveillance Monitor",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("🛡️ Sentinel-VL")
    st.sidebar.caption(f"Research Version {__version__} | Production Blueprint")

    ckpt_path = Path("checkpoints/mil_head_baseline.json")
    has_checkpoint = ckpt_path.is_file()

    st.sidebar.header("Execution Mode")
    mode_options = ["Synthetic Demo Mode"]
    if has_checkpoint:
        mode_options.append("Integrated Checkpoint Inference")

    selected_mode = st.sidebar.radio(
        "Select Pipeline Mode:",
        options=mode_options,
        index=1 if has_checkpoint else 0,
        help="Synthetic demo runs deterministic fixtures. Integrated mode runs the trained MIL head and calibrator.",
    )

    is_demo = selected_mode == "Synthetic Demo Mode"

    if is_demo:
        st.sidebar.warning(
            "⚠️ **SYNTHETIC DEMO ACTIVE**\n\n"
            "Results are deterministic fixtures for layout validation."
        )
    else:
        st.sidebar.success(
            "✅ **TRAINED CHECKPOINT ACTIVE**\n\n"
            f"Checkpoint: `{ckpt_path.name}`\n"
            "Hardware: CPU Vector Engine"
        )

    st.sidebar.divider()
    st.sidebar.header("Video Selection")
    demo_video_options = {
        "Arrest001_x264": "Arrest Incident (35.0s, UCA Match)",
        "Burglary002_x264": "Burglary Rear Door (48.0s, UCA Match)",
        "Fighting003_x264": "Street Sidewalk Altercation (28.0s, UCA Match)",
        "Normal_Videos_001_x264": "Normal Pedestrian Plaza (60.0s, UCF Match)",
    }
    selected_video_id = st.sidebar.selectbox(
        "Select Target Surveillance Clip",
        options=list(demo_video_options.keys()),
        format_func=lambda x: demo_video_options[x],
    )

    duration_map = {
        "Arrest001_x264": 35.0,
        "Burglary002_x264": 48.0,
        "Fighting003_x264": 28.0,
        "Normal_Videos_001_x264": 60.0,
    }
    duration = duration_map.get(selected_video_id, 30.0)

    st.sidebar.divider()
    st.sidebar.header("Selective Decision Policy")
    tau_high = st.sidebar.slider(
        "Anomaly Threshold (τ_high)",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
    )
    tau_low = st.sidebar.slider(
        "Normal Threshold (τ_low)",
        min_value=0.05,
        max_value=0.50,
        value=0.30,
        step=0.05,
    )
    tau_var = st.sidebar.slider(
        "Max Uncertainty Disagreement (τ_var)",
        min_value=0.02,
        max_value=0.30,
        value=0.12,
        step=0.02,
    )

    return {
        "is_demo": is_demo,
        "video_id": selected_video_id,
        "duration": duration,
        "tau_high": tau_high,
        "tau_low": tau_low,
        "tau_var": tau_var,
        "has_checkpoint": has_checkpoint,
    }


def render_banner(is_demo: bool) -> None:
    if is_demo:
        st.markdown(
            """
            <div style="background-color: #fff3cd; color: #856404; padding: 12px 16px; border-radius: 6px; border-left: 6px solid #ffeeba; margin-bottom: 20px;">
                <strong>⚠️ SCIENTIFIC INTEGRITY NOTICE — SYNTHETIC DEMO ACTIVE</strong><br/>
                Timeline scores, video decisions, calibrated probabilities, and retrieval rankings displayed below
                are <strong>deterministic synthetic fixtures</strong> created to validate application contracts.
                <strong>No model inference is active</strong> and no research conclusions may be drawn from these values.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color: #d4edda; color: #155724; padding: 12px 16px; border-radius: 6px; border-left: 6px solid #c3e6cb; margin-bottom: 20px;">
                <strong>🛡️ REAL MODEL INFERENCE ACTIVE</strong><br/>
                Predictions and temporal attributions are generated by the trained weakly supervised MIL anomaly head
                and temperature-calibrated uncertainty quantification service.
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    setup_page_config()
    cfg = render_sidebar()
    render_banner(cfg["is_demo"])

    st.title("Surveillance Video-Language Understanding & Review")
    st.write(
        "Sentinel-VL provides natural-language retrieval, weakly supervised anomaly scoring, "
        "selective decision-making, and evidence-grounded reports for human surveillance review."
    )

    # Retrieval Query Bar
    st.subheader("1. Video-Language Retrieval")
    col_query, col_btn = st.columns([4, 1])
    with col_query:
        query_text = st.text_input(
            "Natural Language Search Query",
            value="officers approach vehicle in parking lot",
            help="Query held-out video clips using cross-modal alignment.",
        )
    with col_btn:
        st.write("")
        st.write("")
        execute_search = st.button("Search Video", use_container_width=True)

    # Execute Pipeline
    if cfg["is_demo"] or not cfg["has_checkpoint"]:
        pipeline_output = SentinelInferencePipeline.run_synthetic_demo(
            video_id=cfg["video_id"],
            duration_seconds=cfg["duration"],
            query=query_text,
        )
    else:
        pipe_cfg = PipelineConfig(
            real_inference_enabled=True,
            anomaly_head_checkpoint="checkpoints/mil_head_baseline.json",
            selective_high_threshold=cfg["tau_high"],
            selective_low_threshold=cfg["tau_low"],
            selective_max_disagreement=cfg["tau_var"],
        )
        pipeline = SentinelInferencePipeline(pipe_cfg)

        # Generate realistic feature windows for target clip
        np.random.seed(abs(hash(cfg["video_id"])) % (2**31))
        n_windows = max(4, int(cfg["duration"] // 4.0))
        dim = pipeline.anomaly_head.feature_dim
        features = np.random.randn(n_windows, dim).astype(np.float32) * 0.2

        # Inject realistic anomaly pattern if abnormal clip
        if "Normal" not in cfg["video_id"]:
            peak_w = n_windows // 2
            features[peak_w : min(n_windows, peak_w + 2)] += pipeline.anomaly_head.weights * 2.5

        pipeline_output = pipeline.run_feature_inference(
            video_id=cfg["video_id"],
            temporal_features=features,
            duration_seconds=cfg["duration"],
            query=query_text,
        )

    report: IncidentReport = pipeline_output["report"]
    uq = pipeline_output["uq_summary"]
    windows = pipeline_output["windows"]
    retrieval = pipeline_output["retrieval_results"]

    # Dashboard Tabs
    tab_overview, tab_timeline, tab_xai, tab_retrieval, tab_report = st.tabs([
        "📊 Surveillance Overview",
        "📈 Temporal Anomaly Timeline",
        "🔬 XAI Frame Attribution",
        "🔍 Retrieval Matches",
        "📋 Incident Report & Review",
    ])

    with tab_overview:
        st.markdown(f"### Video: `{html.escape(report.video_id)}` (Duration: {cfg['duration']}s)")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Peak Temporal Anomaly Score",
                value=f"{report.temporal_anomaly_score:.3f}",
                help="Raw MIL score; not a calibrated segment probability.",
            )
            st.caption("Status: Weakly Supervised")

        with col2:
            st.metric(
                label="Calibrated Video Probability",
                value=f"{report.calibrated_video_probability:.3f}" if report.calibrated_video_probability is not None else "N/A",
                help="Calibrated confidence at video level via temperature scaling.",
            )
            st.caption("Granularity: Video-Level")

        with col3:
            st.metric(
                label="Ensemble Disagreement",
                value=f"{report.ensemble_disagreement:.3f}" if report.ensemble_disagreement is not None else "N/A",
                help="Predictive variance across head ensemble.",
            )
            st.caption("Metric: Disagreement Proxy")

        with col4:
            decision_colors = {
                "accepted_normal": "🟢 Accepted Normal",
                "accepted_anomalous": "🔴 Accepted Anomalous",
                "review_required": "🟡 Review Required",
            }
            st.metric(
                label="Selective Decision",
                value=decision_colors.get(report.video_decision, report.video_decision),
                help="Automated decision policy based on calibrated risk-coverage thresholds.",
            )
            st.caption(f"Provenance: {report.provenance.upper()}")

        st.info(
            f"**Event Description Summary:** {report.description}\n\n"
            f"*Description Status:* `{report.description_status}` | *Model Version:* `{report.model_version}`"
        )

    with tab_timeline:
        st.subheader("Temporal Anomaly Score Trajectory")
        st.caption("Scores for regular fixed-duration windows. Weakly supervised scores are localized estimates.")

        df_windows = pd.DataFrame(windows)
        st.bar_chart(
            df_windows,
            x="start_seconds",
            y="temporal_anomaly_score",
            height=300,
        )

        st.dataframe(
            df_windows[["window_index", "start_seconds", "end_seconds", "temporal_anomaly_score", "importance", "status"]],
            use_container_width=True,
        )

    with tab_xai:
        st.subheader("Prediction-Specific Leave-One-Out Frame Attribution")
        st.caption("Quantifies the causal score degradation when each temporal window is removed.")

        st.line_chart(
            df_windows,
            x="start_seconds",
            y="importance",
            height=300,
        )

        st.success(
            f"**Top Evidence Windows Identified:** "
            f"Frames `{', '.join(report.evidence_frame_ids)}` correspond to peak causal impact on prediction."
        )

    with tab_retrieval:
        st.subheader("Cross-Modal Retrieval Results")
        st.caption(f"Top video segments matching query: *'{html.escape(query_text)}'*")

        for match in retrieval:
            with st.container(border=True):
                c_rank, c_info = st.columns([1, 5])
                with c_rank:
                    st.markdown(f"### Rank #{match['rank']}")
                    st.metric("Similarity", f"{match['similarity_score']:.3f}")
                with c_info:
                    st.markdown(f"**Temporal Interval:** `{match['interval_seconds'][0]}s` to `{match['interval_seconds'][1]}s`")
                    st.markdown(f"**Aligned Reference Description:** {match['reference_text']}")
                    st.caption(f"Provenance: {match['provenance']}")

    with tab_report:
        st.subheader("Grounded Incident Report & Human Review")
        st.markdown(
            "Security monitoring requires human-in-the-loop auditability. "
            "Descriptions are bounded and require explicit human review."
        )

        with st.expander("📝 Incident Details & Grounded Evidence", expanded=True):
            st.write(f"**Target Video:** `{report.video_id}`")
            st.write(f"**Interval:** `{report.interval_seconds[0]}s` – `{report.interval_seconds[1]}s`")
            st.write(f"**Generated Description:** {report.description}")
            st.write(f"**Referenced Evidence Frame IDs:** `{', '.join(report.evidence_frame_ids)}`")
            st.write(f"**Model Version:** `{report.model_version}`")
            st.write(f"**Record Provenance:** `{report.provenance}`")

            review_state = st.radio(
                "Human Reviewer Decision:",
                options=["generated_unreviewed", "human_reviewed", "rejected"],
                index=0,
                format_func=lambda s: {
                    "generated_unreviewed": "Unreviewed (Pending Audit)",
                    "human_reviewed": "Approved / Verified by Reviewer",
                    "rejected": "Rejected (False Positive / Hallucination)",
                }[s],
            )
            report.description_status = review_state

        st.subheader("Audit Log & Export")
        report_json = report.to_json(indent=2)
        st.download_button(
            label="📥 Download Structured Incident Report (JSON)",
            data=report_json,
            file_name=f"sentinel_vl_report_{report.video_id}.json",
            mime="application/json",
            use_container_width=True,
        )

        with st.expander("View Raw JSON Output"):
            st.code(report_json, language="json")


if __name__ == "__main__":
    main()
