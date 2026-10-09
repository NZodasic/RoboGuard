# Sentinel-VL

**Uncertainty-Aware and Explainable Video–Language Understanding for Surveillance Monitoring**

Sentinel-VL is a software-only research system and prototype application for surveillance video understanding. It integrates cross-modal video-language retrieval, weakly supervised multiple-instance learning (MIL) anomaly detection, uncertainty quantification (selective prediction and calibration), prediction-specific attribution (XAI), and grounded incident reporting for human review.

---

## ⚠️ Scientific Integrity & Scope Notice (Milestone M0)

- **Synthetic Demo Active:** All timeline scores, video decisions, calibrated probabilities, and retrieval rankings currently displayed in the Streamlit application shell are deterministic synthetic fixtures.
- **No Fabricated Inference:** Real model inference is disabled by design until verified pretrained checkpoints and validated dataset manifests are configured.
- **Surveillance Understanding Only:** Sentinel-VL evaluates observable events in recorded surveillance video. It does not perform facial recognition, identity matching, intent inference, or autonomous physical enforcement.

---

## Repository Layout

```text
sentinel-vl/
├── pyproject.toml              # Packaging, dependencies, and tool configs
├── README.md                   # Setup guide and usage documentation
├── .gitignore                  # Git exclusions for checkpoints, venv, and large data
├── src/
│   └── sentinel_vl/
│       ├── __init__.py         # Package entry and versioning (v0.1.0)
│       ├── cli.py              # Sentinel-VL command-line interface
│       ├── data/
│       │   ├── schemas.py      # Typed contracts (VideoRecord, CaptionSegment, AnomalyWindow)
│       │   ├── manifest.py     # Manifest serialization, integrity, and disk checks
│       │   ├── split_auditor.py# Train/val/test and cross-protocol leakage auditor
│       │   └── uca_adapter.py  # Provisional UCA adapter (unverified until M1 sample)
│       ├── models/
│       │   └── base.py         # Abstract interfaces for visual/temporal/alignment/MIL models
│       ├── training/           # Offline training routines (staged for M2-M3)
│       │   └── __init__.py
│       ├── inference/
│       │   └── pipeline.py     # Inference safeguards & deterministic synthetic demo engine
│       ├── uncertainty/
│       │   └── contracts.py    # UQ predictions, risk-coverage, and entropy calculations
│       ├── explainability/     # Attribution and XAI contracts (staged for M5)
│       │   └── __init__.py
│       └── reporting/
│           └── schemas.py      # IncidentReport schema, evidence validation, and JSON export
├── app/
│   └── streamlit_app.py        # Streamlit surveillance replay and audit shell
├── tests/
│   ├── conftest.py             # Deterministic test fixtures
│   ├── test_data_contracts.py  # Contract invariant and bounds testing
│   ├── test_split_auditor.py   # Partition disjointness and leak detection tests
│   ├── test_uca_adapter.py     # Adapter status and unverified safeguard tests
│   ├── test_uq_and_reporting.py# Selective prediction and incident report schema tests
│   ├── test_inference_pipeline.py # Safeguard and synthetic demo tests
│   ├── test_cli.py             # CLI command execution tests
│   └── test_app_mode.py        # Streamlit startup safety and provenance tests
└── docs/
    ├── ARCHITECTURE.md         # Layered design, contracts, and safety engineering
    ├── DATA_PROTOCOL.md        # Video/annotation schemas, split rules, and integrity
    ├── EXPERIMENTS.md          # Empirical baseline results and ablation studies
    ├── PROJECT_PROPOSAL.md     # Full research blueprint and scientific background
    └── PROJECT_STATE.md        # Living status, decisions, blockers, and next steps
```

---

## Environment Setup

### 1. Prerequisites
- Linux OS (tested on Ubuntu / x86_64)
- Python 3.10+ (tested on Python 3.14.4)
- Git

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install Sentinel-VL
Install core package with application dependencies in editable mode:
```bash
pip install -e ".[app,dev]"
```

---

## Verification & Testing

Run the full automated test suite (32 unit tests):
```bash
pytest -v
```

---

## CLI Usage

Sentinel-VL includes a command-line interface:

```bash
# Display help and available subcommands
sentinel-vl --help

# Inspect system environment, compute, and data readiness
sentinel-vl status

# Run built-in split auditor (verifies partition disjointness)
sentinel-vl audit-splits

# Inspect synthetic demo output payload
sentinel-vl demo-summary

# Output synthetic demo as formatted JSON
sentinel-vl demo-summary --json

# Validate a manifest file (schema and cross-references)
sentinel-vl validate-manifest --manifest-path manifests/sample_manifest.json
```

---

## Launching the Streamlit Application

Start the surveillance replay and audit interface in synthetic demo mode:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

The application includes:
- **Synthetic Demo Banner:** Clearly displays demo status and alerts the operator that real inference is disabled.
- **Surveillance Overview:** Displays peak temporal anomaly score, video-level calibrated probability, ensemble disagreement, and selective review decision.
- **Temporal Timeline:** Interactive bar chart visualizing scores across regular temporal windows (4-second blocks).
- **Retrieval Engine:** Query input field with ranked video clip matches and aligned reference descriptions.
- **Human-in-the-Loop Audit:** Human reviewer decision toggle (`Unreviewed`, `Approved`, `Rejected`) and structured JSON report download.

---

## Project Status & Roadmap

Sentinel-VL development follows a staged milestone plan:
- **M0: Repository Foundation & Honest Demo (Completed)**
- **M1: Real Annotation Audit & Ingestion (Next Milestone)**
- **M2: Video-Language Retrieval Baseline**
- **M3: Weakly Supervised Anomaly Baseline (MIL)**
- **M4: Calibration & Selective Prediction**
- **M5: Explainability (XAI) & Grounded Reporting**
- **M6: Full Application Integration & Final Experiments**

Refer to `docs/PROJECT_STATE.md` for current progress, architecture decisions, and active blockers.

