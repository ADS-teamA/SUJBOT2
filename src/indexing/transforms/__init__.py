"""
Custom LlamaIndex TransformComponents for the indexing pipeline.

Transforms:
    - GeminiEntityLabeler: Extract entities from chunks using Gemini 2.5 Flash
    - BaseLabeler: Base class for document labeling transformers
    - DocumentCategoryExtractor: Extract document-level categories with dynamic taxonomy
    - SectionKeywordExtractor: Extract keywords at section level
    - ChunkQuestionGenerator: Generate synthetic questions for HyDE boost
    - LabelingPipeline: Orchestrate full labeling flow with smart propagation
    - LabelingBatchProcessor: Process labeling via OpenAI Batch API
"""

from src.indexing.transforms.gemini_entity_labeler import GeminiEntityLabeler
from src.indexing.transforms.base_labeler import BaseLabeler, SyncLabeler, AsyncLabeler
from src.indexing.transforms.category_extractor import (
    DocumentCategoryExtractor,
    DocumentTaxonomy,
)
from src.indexing.transforms.keyword_extractor import (
    SectionKeywordExtractor,
    SectionKeywords,
)
from src.indexing.transforms.question_generator import (
    ChunkQuestionGenerator,
    ChunkQuestions,
)
from src.indexing.transforms.labeling_pipeline import (
    LabelingPipeline,
    LabelingResult,
    create_labeling_pipeline,
)
from src.indexing.transforms.batch_processor import LabelingBatchProcessor

__all__ = [
    # Entity labeling
    "GeminiEntityLabeler",
    # Base classes
    "BaseLabeler",
    "SyncLabeler",
    "AsyncLabeler",
    # Category extraction
    "DocumentCategoryExtractor",
    "DocumentTaxonomy",
    # Keyword extraction
    "SectionKeywordExtractor",
    "SectionKeywords",
    # Question generation
    "ChunkQuestionGenerator",
    "ChunkQuestions",
    # Pipeline
    "LabelingPipeline",
    "LabelingResult",
    "create_labeling_pipeline",
    "LabelingBatchProcessor",
]
