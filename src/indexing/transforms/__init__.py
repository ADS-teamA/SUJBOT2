"""
Custom LlamaIndex TransformComponents for the indexing pipeline.

Transforms:
    - GeminiEntityLabeler: Extract entities from chunks using Gemini 2.5 Flash
"""

from src.indexing.transforms.gemini_entity_labeler import GeminiEntityLabeler

__all__ = ["GeminiEntityLabeler"]
