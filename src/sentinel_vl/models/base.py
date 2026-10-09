"""Base abstractions and contracts for Sentinel-VL models.

Modular interfaces for:
- Visual encoders (frame spatial feature extraction)
- Temporal encoders / aggregators (temporal sequence modeling)
- Alignment heads (video-text contrastive scoring)
- Anomaly heads (multiple-instance learning scoring)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseVisualEncoder(ABC):
    """Abstract base class for visual feature extraction."""

    @abstractmethod
    def encode_frames(self, frames: Any) -> Any:
        """Extracts spatial representations for sampled frames."""
        pass


class BaseTemporalEncoder(ABC):
    """Abstract base class for temporal sequence aggregation."""

    @abstractmethod
    def forward(self, frame_features: Any) -> Any:
        """Aggregates frame features into clip/window temporal representations."""
        pass


class BaseAlignmentHead(ABC):
    """Abstract base class for video-text alignment."""

    @abstractmethod
    def score_similarity(self, video_embedding: Any, text_embedding: Any) -> float:
        """Computes normalized cosine similarity between video and text representations."""
        pass


class BaseAnomalyHead(ABC):
    """Abstract base class for weakly supervised multiple-instance learning anomaly head."""

    @abstractmethod
    def predict_window_scores(self, temporal_features: Any) -> List[float]:
        """Predicts anomaly scores for sampled temporal windows."""
        pass

    @abstractmethod
    def aggregate_video_score(self, window_scores: List[float], top_k: int = 3) -> float:
        """Aggregates window-level scores into a video-level bag prediction."""
        pass

