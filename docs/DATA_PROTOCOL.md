# Sentinel-VL Data Protocol

## 1. Primary Datasets

Sentinel-VL combines two complementary surveillance datasets:

1. **UCA (UCF-Crime Annotation):**
   - Introduced at CVPR 2024 ("Towards Surveillance Video-and-Language Understanding").
   - Provides temporally localized natural-language event descriptions (23,542 sentences) across 1,854 videos.
   - Used for video-text alignment, text-to-video retrieval, and grounded event description.
   - **Crucial Rule:** UCA sentences describe observable actions and context; they are **not** automatic binary anomaly labels.

2. **UCF-Crime:**
   - Provides the underlying surveillance video files across 13 anomaly categories and normal activities.
   - Used for weakly supervised video-level anomaly learning and temporal evaluation where ground truth is verified.
   - **Crucial Rule:** UCA includes a subset of UCF-Crime. Video IDs and counts must be audited and distinguished.

---

## 2. Ingestion & Schema Contracts

Raw upstream files must be parsed through versioned adapters into normalized internal dataclasses:

### 2.1 Video Record (`VideoRecord`)
- `video_id`: Unique identifier matching source video filename stem.
- `source_path`: Absolute or relative path to media file.
- `duration_seconds`: Total video duration in seconds (must be positive).
- `decoding_status`: `ok`, `corrupt`, `unverified`, or `missing`.
- `fps`: Verified frame rate (handling variable frame rates via timestamps).
- `checksum`: SHA-256 hash of media file for provenance tracking.

### 2.2 Caption Segment (`CaptionSegment`)
- `segment_id`: Unique identifier for the caption event.
- `video_id`: Foreign key referencing a declared `VideoRecord`.
- `start_seconds`: Event start timestamp (non-negative).
- `end_seconds`: Event end timestamp (strictly greater than start time).
- `caption`: Non-empty description string.
- `provenance`: `uca_human_reference` or `provisional_annotation`.

### 2.3 Anomaly Window (`AnomalyWindow`)
- `window_id`: Unique temporal window identifier.
- `video_id`: Foreign key referencing a declared `VideoRecord`.
- `start_seconds` / `end_seconds`: Window boundaries.
- `label`: `0` (normal), `1` (anomalous), or `None` (unlabeled).
- `label_provenance`: `ucf_crime_video_level`, `temporal_gt`, or `unlabeled`.

---

## 3. Segmentation Strategies

Sentinel-VL maintains two independent segmentation modes:

1. **Caption-Aligned Segmentation (Retrieval Task):**
   - Temporal intervals match UCA start and end timestamps.
   - Used for training and evaluating cross-modal retrieval.
2. **Fixed-Window Segmentation (Anomaly Detection Task):**
   - Uniform, regular temporal blocks (pilot default: 4.0 to 8.0 seconds).
   - Timestamp-aware frame sampling (e.g., 8–16 uniformly spaced frames per window).
   - Test-time anomaly detection must **never** be segmented around known ground-truth events.

---

## 4. Split Integrity & Leakage Prevention Rules

1. **Source-Video Isolation:**
   All temporal segments, frames, and captions originating from the same source video file **must** belong to the same partition (train, validation, or test). Partitioning by frame or clip is prohibited.
2. **Cross-Protocol Audit:**
   Before training a shared visual or temporal encoder across tasks, audit video membership between UCA retrieval splits and UCF-Crime anomaly splits. A video used for training in one task must not appear in the held-out evaluation set of another task.
3. **No Reference Caption Leakage:**
   Reference captions are strictly training targets or retrieval candidates. Test captions must never be supplied to video-only anomaly detection models.
4. **Calibration Partition Separation:**
   Parameters for temperature scaling and selective prediction thresholds must be fitted on a dedicated development calibration split, completely disjoint from the final test evaluation partition.

---

## 5. Current Implementation Status (Milestone M0)

- **Internal Schemas:** Fully defined and tested with interval and reference validations.
- **Manifest Manager:** Validates manifests, checks file existence on disk, and enforces bounds.
- **Split Auditor:** Implemented and verified for both internal partition disjointness and cross-protocol leakage detection.
- **UCA Adapter Status:** **PROVISIONAL & UNVERIFIED**. Ingestion of real annotation files is blocked until an actual upstream sample is inspected and verified in Milestone M1.

