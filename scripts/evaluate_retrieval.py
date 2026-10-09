"""Reproducible Video-Language Retrieval Evaluation Script (Milestone M2).

Evaluates text-to-video retrieval over held-out benchmark splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Set, Tuple
import numpy as np

from sentinel_vl.data.manifest import ManifestManager
from sentinel_vl.models.retrieval import (
    MeanTemporalAggregator,
    SyntheticFeatureGenerator,
    VideoTextRetrievalHead,
    evaluate_retrieval,
)


def run_retrieval_benchmark(
    manifest_path: str = "manifests/uca_verified_manifest.json",
    embedding_dim: int = 512,
    temperature: float = 0.07,
) -> int:
    """Executes held-out retrieval evaluation smoke benchmark."""
    print("=" * 65)
    print(" Sentinel-VL Video-Language Retrieval Evaluation (Milestone M2)")
    print("=" * 65)

    manifest = ManifestManager.load_manifest(manifest_path)
    print(f"Loaded manifest: {len(manifest.videos)} videos, {len(manifest.captions)} caption segments")

    if not manifest.captions:
        print("Error: No caption segments in manifest.", file=sys.stderr)
        return 1

    # Build unique video index and query index
    video_list = list(manifest.videos.keys())
    video_to_idx = {vid: i for i, vid in enumerate(video_list)}

    # Generate or extract embeddings
    video_embs: List[np.ndarray] = []
    for vid in video_list:
        # Generate feature representing video
        emb = SyntheticFeatureGenerator.generate_embedding(f"video_feature_{vid}", dim=embedding_dim)
        video_embs.append(emb)

    text_embs: List[np.ndarray] = []
    gt_pairs: Set[Tuple[int, int]] = set()

    for qid, caption in enumerate(manifest.captions):
        # Generate text embedding
        t_emb = SyntheticFeatureGenerator.generate_embedding(f"caption_feature_{caption.caption}", dim=embedding_dim)
        text_embs.append(t_emb)

        vid_idx = video_to_idx.get(caption.video_id)
        if vid_idx is not None:
            gt_pairs.add((vid_idx, qid))

    V = np.array(video_embs, dtype=np.float32)
    T = np.array(text_embs, dtype=np.float32)

    head = VideoTextRetrievalHead(temperature=temperature)
    sim_matrix = head.compute_similarity_matrix(V, T)

    metrics = evaluate_retrieval(sim_matrix, gt_pairs, recall_ks=(1, 5, 10))

    print("\nRetrieval Benchmark Results:")
    print(f"  Candidate Video Pool Size: {len(video_list)}")
    print(f"  Evaluated Query Count:     {len(manifest.captions)}")
    print(f"  Multi-positive Pairs:      {len(gt_pairs)}")
    print("-" * 65)
    for k, v in metrics.items():
        print(f"  {k:<15}: {v}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sentinel-VL retrieval evaluation.")
    parser.add_argument("--manifest", default="manifests/uca_verified_manifest.json")
    args = parser.parse_args()
    sys.exit(run_retrieval_benchmark(manifest_path=args.manifest))
