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
    from sentinel_vl.evaluation.retrieval_eval import run_retrieval_smoke_benchmark

    print("=" * 65)
    print(" Sentinel-VL Video-Language Retrieval Evaluation (Milestone M2)")
    print(" Notice: Evaluating on synthetic pseudo-embeddings as pipeline check.")
    print("=" * 65)

    res = run_retrieval_smoke_benchmark(
        manifest_path=manifest_path,
        embedding_dim=embedding_dim,
        temperature=temperature,
    )

    metrics = res["metrics"]
    print("\nRetrieval Benchmark Results:")
    print(f"  Candidate Video Pool Size: {res['candidate_pool_size']}")
    print(f"  Evaluated Query Count:     {res['query_count']}")
    print(f"  Multi-positive Pairs:      {res['multi_positive_pairs']}")
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
