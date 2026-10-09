"""Video-Language Retrieval Evaluation Runner (Packaged Implementation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from sentinel_vl.data.manifest import ManifestManager
from sentinel_vl.models.retrieval import (
    SyntheticFeatureGenerator,
    VideoTextRetrievalHead,
    evaluate_retrieval,
)


def run_retrieval_smoke_benchmark(
    manifest_path: str | Path = "manifests/uca_verified_manifest.json",
    embedding_dim: int = 512,
    temperature: float = 0.07,
) -> Dict[str, Any]:
    """Executes a retrieval evaluation over manifest using feature representations.

    Note: When run with SyntheticFeatureGenerator, this evaluates pipeline ranking logic
    and metrics computation as an engineering smoke test, not real visual semantic alignment.
    """
    manifest = ManifestManager.load_manifest(manifest_path)
    if not manifest.captions:
        raise ValueError(f"No caption segments found in manifest '{manifest_path}'.")

    video_list = list(manifest.videos.keys())
    video_to_idx = {vid: i for i, vid in enumerate(video_list)}

    video_embs: List[np.ndarray] = []
    for vid in video_list:
        emb = SyntheticFeatureGenerator.generate_embedding(f"video_feature_{vid}", dim=embedding_dim)
        video_embs.append(emb)

    text_embs: List[np.ndarray] = []
    gt_pairs: Set[Tuple[int, int]] = set()

    for qid, caption in enumerate(manifest.captions):
        t_emb = SyntheticFeatureGenerator.generate_embedding(
            f"caption_feature_{caption.caption}", dim=embedding_dim
        )
        text_embs.append(t_emb)

        vid_idx = video_to_idx.get(caption.video_id)
        if vid_idx is not None:
            gt_pairs.add((vid_idx, qid))

    V = np.array(video_embs, dtype=np.float32)
    T = np.array(text_embs, dtype=np.float32)

    head = VideoTextRetrievalHead(temperature=temperature)
    sim_matrix = head.compute_similarity_matrix(V, T)

    metrics = evaluate_retrieval(sim_matrix, gt_pairs, recall_ks=(1, 5, 10))

    return {
        "candidate_pool_size": len(video_list),
        "query_count": len(manifest.captions),
        "multi_positive_pairs": len(gt_pairs),
        "metrics": metrics,
        "recall_at_1": metrics.get("R@1", 0.0),
        "recall_at_5": metrics.get("R@5", 0.0),
        "recall_at_10": metrics.get("R@10", 0.0),
        "mean_rank": metrics.get("MeanRank", 0.0),
        "median_rank": metrics.get("MedianRank", 0.0),
        "is_synthetic_smoke_check": True,
    }
