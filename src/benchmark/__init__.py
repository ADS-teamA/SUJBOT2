"""
Benchmark evaluation system for RAG pipeline.

Evaluates retrieval quality on standardized datasets (PrivacyQA, CUAD, MAUD).
"""

from .config import BenchmarkConfig
from .runner import BenchmarkRunner

__all__ = ["BenchmarkConfig", "BenchmarkRunner"]
