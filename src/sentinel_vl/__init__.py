"""Sentinel-VL: Uncertainty-Aware and Explainable Video-Language Understanding for Security Monitoring."""

__version__ = "0.1.0"
__author__ = "Sentinel-VL Research Team"

from sentinel_vl.inference.pipeline import InferenceDisabledError, SentinelInferencePipeline

__all__ = [
    "__version__",
    "SentinelInferencePipeline",
    "InferenceDisabledError",
]

