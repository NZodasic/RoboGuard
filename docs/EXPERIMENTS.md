# Sentinel-VL Empirical Evaluation Protocols & Benchmark Roadmap

> [!IMPORTANT]
> **Real Dataset Benchmark Status: Pending Feature Ingestion**
> Real video files and pretrained foundation model checkpoints (e.g., CLIP / Video-Language encoders) are not yet bundled or ingested into this workspace.
> Synthetic pipeline verifications have been separated and documented in [docs/SYNTHETIC_SMOKE_TESTS.md](file:///home/raymond/Desktop/RoboGuard/docs/SYNTHETIC_SMOKE_TESTS.md).
> This document specifies the rigorous evaluation protocols, data splits, and baseline targets for empirical evaluation on real surveillance benchmarks once features are extracted.

---

## 1. Benchmark Datasets & Splits

Sentinel-VL targets evaluation on two primary surveillance video understanding benchmarks:

### 1.1 UCA (Surveillance Video Understanding & Captioning)
- **Source**: CVPR 2024 (`Xuange923/Surveillance-Video-Understanding`).
- **Annotations**: Temporal intervals paired with natural-language surveillance activity descriptions.
- **Protocol**: Held-out test split adhering strictly to video-level isolation. No clip segments from the same source video may span across training and test partitions.
- **Tasks**:
  1. Natural-language cross-modal video retrieval (text-to-clip and clip-to-text).
  2. Grounded evidence retrieval for automated review.

### 1.2 UCF-Crime / TAD Benchmarks
- **Task**: Weakly supervised video anomaly detection (WS-VAD).
- **Labels**: Video-level normal vs. abnormal labels during training; temporal segment annotations for evaluation.
- **Evaluation Splits**: Standard UCF-Crime split protocol (810 normal videos, 800 anomaly videos in train; 150 normal, 140 anomaly in test).

---

## 2. Experimental Protocols

### 2.1 Video-Language Retrieval (Task 1)
- **Feature Extraction Protocol**:
  - Sample regular non-overlapping windows (e.g., 16 frames at 16 FPS / 1-second stride).
  - Extract visual representations using a validated, paired visual-language backbone (e.g., CLIP ViT-B/16 or ViT-L/14).
  - Extract text embeddings for user queries and ground-truth reference captions using the identical matched text encoder.
  - Apply temporal pooling (`MeanTemporalAggregator`) followed by L2 normalization.
- **Metrics**:
  - Recall@1, Recall@5, Recall@10.
  - Mean Rank (MR) and Median Rank (MedR).
  - Multi-positive handling: evaluate rank of the most relevant positive description per query.

### 2.2 Weakly Supervised Anomaly Detection (Task 2)
- **Model Architecture**:
  - `MILAnomalyHead` with top-$k$ ratio ($k = \max(1, \lceil \alpha T \rceil)$, $\alpha \in [0.1, 0.2]$).
  - Loss formulation:
    $$\mathcal{L} = \mathcal{L}_{\text{BCE}}(q_i, Y_i) + \lambda_{\text{smooth}} \sum_{t=1}^{T-1} (s_t - s_{t+1})^2 + \lambda_{\text{sparse}} \sum_{t=1}^T s_t$$
- **Metrics**:
  - Frame-level Area Under the ROC Curve (ROC-AUC).
  - Precision-Recall AUC (PR-AUC).
  - Tie handling: distinct score threshold grouping to ensure tie-order invariance.

### 2.3 Uncertainty Quantification & Selective Prediction (Task 3)
- **Calibration Protocol**:
  - Platt / Temperature scaling fitted strictly on a held-out calibration split (disjoint from train and test splits).
  - Consistent score aggregation: calibrate the exact video bag score ($q_i$) used for decision-making.
  - Metric: Expected Calibration Error (ECE) across 10 equal-frequency or equal-width bins.
- **Selective Decision Policy**:
  - Risk-coverage curve across variable operating thresholds ($\tau_{\text{high}}, \tau_{\text{low}}, \tau_{\text{var}}$).
  - Abstention tracking: when coverage is zero, risk is recorded as undefined (`None`).

### 2.4 Explainability & Attribution Faithfulness (Task 4)
- **Method**: Prediction-specific leave-one-out feature attribution ($I_j = q_{\text{full}} - q_{\setminus \{j\}}$).
- **Control Benchmark**:
  - Measure anomaly score drop when removing top-$k$ attributed temporal windows versus removing $k$ random windows across 50 Monte Carlo trials.
  - Report mean degradation and standard error.

---

## 3. Results Log

| Milestone | Task | Dataset | Encoder Backbone | Split / Partition | Status | Key Metrics |
|---|---|---|---|---|---|---|
| **M1** | Data Audit | UCA / UCF-Crime | N/A | `benchmark_splits.json` | **Verified** | 0 leaking video IDs |
| **M2** | Text-Video Retrieval | UCA | CLIP (pending weights) | Test Partition | **Pending Real Features** | Target R@1 > 0.05 |
| **M3** | Anomaly Detection | UCF-Crime | Pretrained Encoder | Standard Test | **Pending Real Features** | Target ROC-AUC > 0.80 |
| **M4** | Calibration & UQ | UCF-Crime / UCA | Multi-Head Ensemble | Val Calibration Split | **Pending Real Features** | Target ECE < 0.05 |
| **M5** | XAI & Reporting | UCA / UCF-Crime | Real Features | Held-Out Test | **Pending Real Features** | Faithfulness ratio > 2.0x |

---

## 4. Instructions for Running Real Experiments

Once video feature files are placed in `data/features/` and verified manifests are configured:
```bash
# 1. Audit partition integrity
sentinel-vl audit-splits --manifests-dir data/manifests

# 2. Train MIL anomaly head on real features
sentinel-vl train-mil --features-dir data/features --manifest data/manifests/train.json --epochs 50

# 3. Fit temperature calibration artifact on validation split
sentinel-vl calibrate --val-features data/features/val --checkpoint checkpoints/mil_head.json

# 4. Evaluate retrieval on test split
sentinel-vl evaluate-retrieval --features-dir data/features/test --test-manifest data/manifests/test_uca.json
```
