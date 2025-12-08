"""
HyDE + Expansion Fusion Retrieval System

This package provides a clean implementation of retrieval using:
- HyDE (Hypothetical Document Embeddings)
- Query Expansion (2 paraphrases)
- Weighted Fusion (w_hyde=0.6, w_exp=0.4)

All API calls go through DeepInfra (Qwen models).
"""

from .deepinfra_client import DeepInfraClient, DeepInfraConfig
from .hyde_expansion import HyDEExpansionGenerator, HyDEExpansionResult
from .fusion_retriever import FusionRetriever, FusionConfig

__all__ = [
    "DeepInfraClient",
    "DeepInfraConfig",
    "HyDEExpansionGenerator",
    "HyDEExpansionResult",
    "FusionRetriever",
    "FusionConfig",
]
