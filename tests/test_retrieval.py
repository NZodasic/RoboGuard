"""Unit tests for Milestone M2 Video-Language Retrieval baseline and metrics."""

import numpy as np
import pytest

from sentinel_vl.models.retrieval import (
    MeanTemporalAggregator,
    SyntheticFeatureGenerator,
    VideoTextRetrievalHead,
    evaluate_retrieval,
)


def test_mean_temporal_aggregator() -> None:
    agg = MeanTemporalAggregator(normalize=True)
    # 4 frames, 8-dim features
    frames = np.ones((4, 8), dtype=np.float32)
    clip_emb = agg.forward(frames)

    assert clip_emb.shape == (8,)
    # Should be unit normalized
    assert pytest.approx(np.linalg.norm(clip_emb), rel=1e-5) == 1.0


def test_mean_temporal_aggregator_invalid_shape() -> None:
    agg = MeanTemporalAggregator()
    with pytest.raises(ValueError, match="Expected 2D array"):
        agg.forward(np.ones((4, 8, 2)))

    with pytest.raises(ValueError, match="Cannot aggregate empty"):
        agg.forward(np.empty((0, 8)))


def test_retrieval_head_cosine_similarity() -> None:
    head = VideoTextRetrievalHead(temperature=0.07)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    t_match = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    t_ortho = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    sim_match = head.score_similarity(v, t_match)
    sim_ortho = head.score_similarity(v, t_ortho)

    assert pytest.approx(sim_match, rel=1e-5) == 1.0
    assert pytest.approx(sim_ortho, abs=1e-5) == 0.0


def test_candidate_ranking_for_query() -> None:
    head = VideoTextRetrievalHead(temperature=0.07)
    q = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array([
        [0.0, 1.0],  # rank 2
        [1.0, 0.0],  # rank 1 (exact match)
        [0.707, 0.707],  # rank 2
    ], dtype=np.float32)

    ranked = head.rank_candidates_for_query(q, candidates, top_k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == 1  # Index 1 is exact match
    assert pytest.approx(ranked[0][1], rel=1e-5) == 1.0


def test_evaluate_retrieval_metrics() -> None:
    # 3 videos, 3 queries
    # Construct similarity matrix where query 0 matches video 0 (rank 1), query 1 matches video 2 (rank 2)
    sim_matrix = np.array([
        [0.9, 0.1, 0.2],
        [0.3, 0.4, 0.8],
        [0.1, 0.7, 0.1],
    ], dtype=np.float32)

    # Positives: (0, 0) -> rank 1, (2, 1) -> video 2 for query 1 has score 0.7, video 1 has 0.4, video 0 has 0.1 -> rank 1
    # query 2 has positive (1, 2) -> video 1 has score 0.8 -> rank 1
    gt_pairs = {(0, 0), (2, 1), (1, 2)}

    metrics = evaluate_retrieval(sim_matrix, gt_pairs, recall_ks=(1, 5))
    assert metrics["R@1"] == 1.0
    assert metrics["R@5"] == 1.0
    assert metrics["MedianRank"] == 1.0


def test_evaluate_retrieval_multi_positive() -> None:
    # Query 0 has two valid positives: video 0 and video 1
    # Ranked order: video 0 (0.9), video 2 (0.5), video 1 (0.4)
    sim_matrix = np.array([
        [0.9],
        [0.4],
        [0.5],
    ], dtype=np.float32)
    gt_pairs = {(0, 0), (1, 0)}

    # Multi-positive handling should count the best rank (rank 1 from video 0)
    metrics = evaluate_retrieval(sim_matrix, gt_pairs, recall_ks=(1,))
    assert metrics["R@1"] == 1.0
    assert metrics["MeanRank"] == 1.0
