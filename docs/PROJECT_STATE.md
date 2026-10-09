# Sentinel-VL Project State

**Current As Of:** Milestone M0/M1 Engineering Stabilization & Provenance Correction  
**Status:** 
- **Milestone M0 (Foundation & Contracts):** Complete & Verified.
- **Milestone M1 (Annotation Ingestion & Partition Audit):** In Progress (UCA adapter and split auditor verified on sample manifests; raw dataset download pending).
- **Milestones M2–M6 (Retrieval, MIL, Uncertainty, XAI, Reporting):** Packaged Infrastructure Implemented; Validated via Synthetic Smoke Tests; Real Surveillance Feature Benchmarks Pending Ingestion.

---

## 1. Milestone Status Breakdown

| Milestone | Scope | Implementation State | Verification Status | Real Data Status |
|---|---|---|---|---|
| **M0** | Package, Contracts, CLI, Demo Shell | Complete | Passing unit tests | Synthetic fixtures only |
| **M1** | Ingestion, Manifests, Split Audit | In Progress | Ingestion verified on sample UCA manifests | Full video downloads pending approval |
| **M2** | Video-Language Retrieval | Packaged Code | Verified with hash pseudo-embeddings | Frozen visual-text backbone pending |
| **M3** | Weakly Supervised Anomaly Head (MIL) | Packaged Code | Loss convergence verified on synthetic bags | UCF-Crime feature benchmark pending |
| **M4** | Calibration & Uncertainty (UQ) | Packaged Code | Temperature scaling verified on mock split | Real-score calibration pending |
| **M5** | Explainability (XAI) & Grounding | Packaged Code | Attribution logic verified | Honest score reporting; real VLM pending |
| **M6** | Inference Pipeline & Dashboard | Packaged Code | Streamlit app running; video inference gated | Real video decoding / encoding pending |

---

## 2. Implemented Code & Package Structure

- **Packaging & Core Dependencies:**
  - `pyproject.toml`: Modern Python packaging with `dependencies = ["numpy>=1.24.0"]` and optional extras (`app`, `dev`, `ml`).
  - `.gitignore`: Configured to exclude environments, caches, raw datasets, and model weight checkpoints.
- **Data Protocols & Adapters (`src/sentinel_vl/data/`):**
  - Typed contracts: `VideoRecord`, `CaptionSegment`, `AnomalyWindow`, `DataManifest`, `SplitPartition`.
  - `UCAAdapter`: Validated against CVPR 2024 UCA schema (`Xuange923/Surveillance-Video-Understanding`), supporting both JSON and TXT annotation formats.
  - `ManifestManager` & `SplitAuditor`: Partition validation ensuring strict video-level isolation and zero cross-protocol contamination.
- **Models & Algorithms (`src/sentinel_vl/models/`, `training/`):**
  - `MILAnomalyHead`: Top-$k$ mean bag aggregation with temporal smoothness and sparsity regularization. Checkpoint loading verifies SHA-256 integrity. Tie-order-invariant ROC-AUC and PR-AUC.
  - `MILTrainer`: Momentum SGD training loop with exact analytical gradients for BCE, smoothness, and sparsity.
  - `VideoTextRetrievalHead`: Cosine similarity ranking with multi-positive minimum rank handling.
- **Uncertainty & Calibration (`src/sentinel_vl/uncertainty/`):**
  - `CalibrationArtifact`: Serialized dataclass storing temperature, video aggregation definition, head checkpoint hashes, and granularity.
  - `TemperatureCalibrator`: Calibrates video bag scores consistently (not peak segment scores); returns `None` (uncalibrated) if no valid calibration artifact is loaded.
  - `HeadEnsemble`: Requires $\ge 2$ independently trained heads for predictive disagreement calculation.
  - `SelectiveDecisionService`: Abstention policy with risk-coverage analysis; reports risk as undefined (`None`) when coverage is zero.
- **Explainability & Grounded Reporting (`src/sentinel_vl/explainability/`, `reporting/`):**
  - `FrameAttributionService`: Leave-one-out causal importance ($I_j = q_{\text{full}} - q_{\setminus \{j\}}$). Consistent typed schemas for small inputs; bag-length reduction confound documented.
  - `GroundedReportGenerator`: Honest score-based incident reports (`"The model assigned an elevated anomaly score ... No visual event description is available"`). Prevents unsupported visual descriptions.
- **Inference Pipeline & UI (`src/sentinel_vl/inference/`, `app/`):**
  - `SentinelInferencePipeline`:
    - `run_video_inference()` explicitly raises `InferenceDisabledError` until timestamped video decoding and visual encoders are integrated.
    - `run_feature_inference()` evaluates pre-extracted features with explicit provenance tracking (`"feature_test"`, `"synthetic_feature_demo"`).
  - `app/streamlit_app.py`: Clearly labeled "Synthetic Demo Mode" and "Synthetic Feature Inference"; label-dependent feature injection removed; SHA-256 deterministic seeding.

---

## 3. Engineering Verifications vs. Real Benchmarks

To maintain strict scientific integrity, all test results are segregated into two distinct categories:

1. **Synthetic Pipeline Smoke Tests** ([docs/SYNTHETIC_SMOKE_TESTS.md](file:///home/raymond/Desktop/RoboGuard/docs/SYNTHETIC_SMOKE_TESTS.md)):
   - Verifies numerical stability of MIL optimization on separable synthetic bags.
   - Verifies temperature fitting routines and coverage calculations.
   - Verifies leave-one-out sensitivity logic on synthetic weights.
   - Verifies retrieval matrix metrics on hash pseudo-embeddings.
2. **Empirical Evaluation Protocols** ([docs/EXPERIMENTS.md](file:///home/raymond/Desktop/RoboGuard/docs/EXPERIMENTS.md)):
   - Defines target protocols for real UCA retrieval and UCF-Crime anomaly detection.
   - Real-data benchmarks are currently pending video downloading and feature extraction.

---

## 4. Current Limitations & Blockers

- **Real Video Inference:** Raw video decoding (`mp4`, `mkv`) and frame feature extraction are not yet wired into the inference entrypoint. `run_video_inference()` raises `InferenceDisabledError`.
- **Pretrained Encoders:** Pretrained visual and text foundation backbones (e.g., CLIP) are not packaged locally; retrieval currently relies on feature inputs or synthetic smoke tests.
- **VLM Generation:** Visual event descriptions are not generated by a multimodal language model; Sentinel-VL generates honest score-based telemetry notices.
- **Dataset Storage:** Full UCF-Crime (110 hours, ~30–50 GB) and full UCA surveillance videos have not been downloaded to respect compute constraints and user authorization.

---

## 5. Execution Commands

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run unit test suite
pytest -v

# 3. Run synthetic pipeline smoke tests
sentinel-vl run-experiments

# 4. Check CLI status
sentinel-vl status

# 5. Launch Streamlit UI
streamlit run app/streamlit_app.py
```
