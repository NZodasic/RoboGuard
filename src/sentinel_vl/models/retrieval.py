"""Video-Language Retrieval Baseline for Sentinel-VL (Milestone M2).

Implements:
1. Mean temporal aggregation over frame features (frozen baseline).
2. Cross-modal cosine similarity scoring.
3. Candidate pool ranking for text-to-video and video-to-text retrieval.
4. Comprehensive multi-positive evaluation metrics (Recall@1, 5, 10, Median Rank).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from sentinel_vl.models.base import BaseAlignmentHead, BaseTemporalEncoder


class MeanTemporalAggregator(BaseTemporalEncoder):
    """Aggregates frame spatial embeddings into a clip-level representation via mean pooling."""

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    def forward(self, frame_features: np.ndarray) -> np.ndarray:
        """Frame features shape: (num_frames, embedding_dim) -> (embedding_dim,)."""
        if frame_features.ndim != 2:
            raise ValueError(f"Expected 2D array (T, D), got shape {frame_features.shape}")
        if frame_features.shape[0] == 0:
            raise ValueError("Cannot aggregate empty frame features.")

        # Mean pooling across temporal dimension
        clip_emb = np.mean(frame_features, axis=0)

        if self.normalize:
            norm = np.linalg.norm(clip_emb)
            if norm > 1e-12:
                clip_emb = clip_emb / norm

        return clip_emb


class VideoTextRetrievalHead(BaseAlignmentHead):
    """Cosine similarity scoring and candidate ranking for video-language retrieval."""

    def __init__(self, temperature: float = 0.07) -> None:
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = temperature

    def score_similarity(self, video_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """Computes normalized cosine similarity between a video and text representation."""
        v_norm = np.linalg.norm(video_embedding)
        t_norm = np.linalg.norm(text_embedding)
        if v_norm < 1e-12 or t_norm < 1e-12:
            return 0.0
        return float(np.dot(video_embedding, text_embedding) / (v_norm * t_norm))

    def compute_similarity_matrix(
        self,
        video_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Computes NxM similarity matrix between N video clips and M text queries.

        Args:
            video_embeddings: Shape (N, D)
            text_embeddings: Shape (M, D)

        Returns:
            Similarity matrix: Shape (N, M) scaled by temperature
        """
        # Normalize rows
        v_norms = np.linalg.norm(video_embeddings, axis=1, keepdims=True)
        t_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        v_normed = np.divide(video_embeddings, np.maximum(v_norms, 1e-12))
        t_normed = np.divide(text_embeddings, np.maximum(t_norms, 1e-12))

        sim = np.matmul(v_normed, t_normed.T) / self.temperature
        return sim

    def rank_candidates_for_query(
        self,
        query_embedding: np.ndarray,
        candidate_video_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Ranks video candidates for a single text query."""
        sims = [
            (idx, self.score_similarity(v, query_embedding))
            for idx, v in enumerate(candidate_video_embeddings)
        ]
        sims.sort(key=lambda item: item[1], reverse=True)
        return sims[:top_k]


def evaluate_retrieval(
    similarity_matrix: np.ndarray,
    ground_truth_pairs: Set[Tuple[int, int]],
    recall_ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Evaluates text-to-video retrieval metrics with multi-positive handling.

    Args:
        similarity_matrix: Shape (N_videos, M_queries). Row i is video i, Col j is query j.
        ground_truth_pairs: Set of (video_idx, query_idx) positive associations.
        recall_ks: Tuple of K values for Recall@K.

    Returns:
        Dictionary with Recall@1, Recall@5, Recall@10, Mean Rank, and Median Rank.
    """
    n_videos, n_queries = similarity_matrix.shape
    if n_queries == 0 or len(ground_truth_pairs) == 0:
        return {f"R@{k}": 0.0 for k in recall_ks} | {"MeanRank": 0.0, "MedianRank": 0.0}

    # Group ground truth positives by query
    query_to_positive_videos: Dict[int, Set[int]] = {}
    for vid, qid in ground_truth_pairs:
        query_to_positive_videos.setdefault(qid, set()).add(vid)

    ranks: List[int] = []
    recalls: Dict[int, int] = {k: 0 for k in recall_ks}
    evaluated_queries = 0

    for qid in range(n_queries):
        pos_videos = query_to_positive_videos.get(qid, set())
        if not pos_videos:
            continue

        evaluated_queries += 1
        query_scores = similarity_matrix[:, qid]
        ranked_video_indices = np.argsort(-query_scores)

        # Multi-positive handling: find best rank among valid positive videos
        best_rank = None
        for rank_pos, vid in enumerate(ranked_video_indices, 1):
            if vid in pos_videos:
                best_rank = rank_pos
                break

        if best_rank is not None:
            ranks.append(best_rank)
            for k in recall_ks:
                if best_rank <= k:
                    recalls[k] += 1

    if evaluated_queries == 0:
        return {f"R@{k}": 0.0 for k in recall_ks} | {"MeanRank": 0.0, "MedianRank": 0.0}

    metrics = {
        f"R@{k}": round(recalls[k] / float(evaluated_queries), 4) for k in recall_ks
    }
    metrics["MeanRank"] = round(float(np.mean(ranks)), 2)
    metrics["MedianRank"] = float(np.median(ranks))
    return metrics


class SyntheticFeatureGenerator:
    """Generates deterministic unit-normalized pseudo-embeddings for fast local testing."""

    @staticmethod
    def generate_embedding(seed_text: str, dim: int = 512) -> np.ndarray:
        """Produces a deterministic pseudo-random normalized vector from text."""
        h = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(h)
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-12)
