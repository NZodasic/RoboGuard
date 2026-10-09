# Sentinel-VL Synthetic Pipeline & Smoke Test Verifications

> [!NOTE]
> **Engineering Sanity Checks Only**: The metrics and tests documented here were executed on synthetic feature vectors, hash-derived pseudo-embeddings, and mock validation arrays. They serve exclusively to verify numerical stability, loss convergence, metric calculation pipelines, and software contracts. They **do not** represent empirical benchmarks or scientific claims on real surveillance datasets (e.g., UCA or UCF-Crime).

---

## 1. Overview of Synthetic Verification Suite

To verify the mathematical and architectural components of Sentinel-VL without requiring multi-gigabyte video downloads or external GPU servers, an automated suite of synthetic smoke tests is implemented in `sentinel-vl run-experiments`.

The verification suite exercises:
1. **MIL Anomaly Optimization**: Gradient-based loss convergence on linearly separable synthetic feature bags.
2. **Temperature Calibration & Selective Prediction**: Numerical fitting of Platt/temperature scaling on mock score arrays and threshold coverage calculations.
3. **Leave-One-Out Temporal Attribution**: Sanity check verifying that masking high-weight windows degrades the bag score more than masking random windows.
4. **Retrieval Metric Pipeline**: Recall@K and rank calculations on deterministic hash pseudo-embeddings.

---

## 2. Weakly Supervised Anomaly Head Smoke Test

- **Model**: `MILAnomalyHead` ($D = 512$, $\text{top\_k\_ratio} = 0.2$).
- **Objective**: Multiple Instance Learning (MIL) loss with temporal smoothness ($\lambda_{\text{smooth}} = 8 \times 10^{-5}$) and sparsity ($\lambda_{\text{sparse}} = 8 \times 10^{-5}$) penalties.
- **Input Data**: 20 normal bags and 20 abnormal bags with Gaussian noise ($D=512$). Abnormal bags have synthetic positive offsets injected in a subset of windows.
- **Purpose**: Verify that gradient computation, temporal difference gradients, L1 sparsity subgradients, and momentum updates execute without numerical instability (NaN/inf).
- **Observed Behavior**:
  - Initial Loss: `~0.14`
  - Converged Loss (50 epochs): `~0.02`
  - Optimization converges stably without numerical overflow.
- **Limitation**: Evaluates artificial feature separation on synthetic vectors; does not constitute surveillance anomaly detection accuracy on raw video.

---

## 3. Calibration & Selective Prediction Sanity Check

- **Calibrator**: `TemperatureCalibrator` optimizing negative log-likelihood (NLL) via bounded scalar search.
- **Input Scores**: Toy validation score array ($N=8$, balanced positive and negative scores).
- **Purpose**: Ensure temperature parameter $T > 0$ optimizes calibration loss on probabilities, outputs valid $[0, 1]$ calibrated values, and exports a serializable `CalibrationArtifact`.
- **Observed Behavior**:
  - Fitted temperature $T \approx 0.22$.
  - Expected Calibration Error (ECE) metric runs without division by zero.
  - Risk-coverage calculations correctly return `None` (undefined) when coverage is 0.0% instead of fabricating zero risk.
- **Limitation**: Evaluated on mock validation points. True operational calibration requires temperature fitting over held-out real-dataset bag predictions.

---

## 4. Temporal Attribution Sensitivity Sanity Check

- **Method**: Prediction-specific leave-one-out importance:
  $$I_j = q_{\text{full}} - q_{\setminus \{j\}}$$
- **Purpose**: Ensure that removing windows identified with high feature weights results in greater score decrease than removing arbitrary random windows.
- **Observed Behavior**:
  - Bag score degradation on removing top attributed windows: `~0.05`.
  - Bag score degradation on removing random windows: `~0.0001`.
- **Limitation**: Note the bag-length reduction confound documented in the module: removing a window shortens bag length $T \to T-1$, which can alter top-$k$ index selection.

---

## 5. Retrieval Rank Metric Pipeline Check

- **Components**: `VideoTextRetrievalHead`, `MeanTemporalAggregator`, `compute_retrieval_metrics`.
- **Input**: Small manifest (4 sample clips) with hash-derived pseudo-embeddings.
- **Purpose**: Verify ranking logic, cosine similarity matrix calculation, multi-positive minimum rank handling, and Recall@K indexing.
- **Limitation**: Because the candidate pool contains only 4 videos, Recall@5 and Recall@10 are trivially 1.0 (100%). Pseudo-embeddings have no semantic video-language alignment.

---

## 6. Execution Command

To re-run these synthetic engineering checks locally:
```bash
sentinel-vl run-experiments
```
