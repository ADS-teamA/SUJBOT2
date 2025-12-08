"""
Benchmark evaluation system for RAG pipeline.

Evaluates retrieval quality on standardized datasets (PrivacyQA, CUAD, MAUD).
"""

from .config import BenchmarkConfig
from .models import QueryResult, BenchmarkResult

__all__ = ["BenchmarkConfig", "QueryResult", "BenchmarkResult"]
