"""
Document labeling pipeline orchestrator.

Orchestrates the full labeling flow with smart propagation:
1. Categories: Document-level (1 LLM call) → propagated to sections/chunks
2. Keywords: Section-level (~100 LLM calls) → propagated to chunks
3. Questions: Chunk-level (HyDE boost, per-chunk generation)

Supports both synchronous and Batch API modes.

Smart Propagation reduces LLM calls from 3000 to ~1100 for a 1000-chunk document.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config_schema import LabelingConfig
from src.extraction_models import DocumentSection, ExtractedDocument
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
from src.multi_layer_chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class LabelingResult:
    """Complete labeling result for a document."""

    taxonomy: DocumentTaxonomy
    section_keywords: Dict[str, SectionKeywords]  # section_id -> keywords
    chunk_questions: Dict[str, ChunkQuestions]  # chunk_id -> questions
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "taxonomy": self.taxonomy.to_dict(),
            "section_keywords": {
                k: v.to_dict() for k, v in self.section_keywords.items()
            },
            "chunk_questions": {
                k: v.to_dict() for k, v in self.chunk_questions.items()
            },
            "stats": self.stats,
        }


class LabelingPipeline:
    """
    Orchestrates document labeling with smart propagation.

    Pipeline flow:
    1. Extract taxonomy + classify document (1 LLM call)
    2. Extract keywords per section (~100 LLM calls)
    3. Generate questions per chunk (~1000 LLM calls via Batch API)
    4. Propagate labels hierarchically
    5. Optionally add questions to embedding_text (HyDE boost)

    Example:
        >>> config = LabelingConfig(enabled=True, use_batch_api=True)
        >>> pipeline = LabelingPipeline(config)
        >>> result = await pipeline.label_document(document, chunks)
        >>> labeled_chunks = pipeline.apply_labels_to_chunks(chunks, result)
    """

    def __init__(self, config: Optional[LabelingConfig] = None):
        """
        Initialize labeling pipeline.

        Args:
            config: Labeling configuration (uses defaults if None)
        """
        self.config = config or LabelingConfig()

        # Initialize extractors
        self.category_extractor = DocumentCategoryExtractor(
            model_name=self.config.model,
            use_dynamic_categories=self.config.use_dynamic_categories,
            fixed_categories=self.config.fixed_categories,
        )

        self.keyword_extractor = SectionKeywordExtractor(
            model_name=self.config.model,
            max_keywords=self.config.max_keywords_per_chunk,
        )

        self.question_generator = ChunkQuestionGenerator(
            model_name=self.config.model,
            max_questions=self.config.max_questions_per_chunk,
            cache_enabled=self.config.cache_enabled,
            cache_size=self.config.cache_size,
        )

        # Statistics
        self._stats = {
            "documents_processed": 0,
            "category_calls": 0,
            "keyword_calls": 0,
            "question_calls": 0,
        }

    def label_document_sync(
        self,
        document: ExtractedDocument,
        chunks: List[Chunk],
    ) -> LabelingResult:
        """
        Label document synchronously (real-time API).

        Args:
            document: Extracted document
            chunks: Document chunks

        Returns:
            Complete labeling result
        """
        logger.info(
            f"Labeling document: {len(document.sections)} sections, {len(chunks)} chunks"
        )

        # Step 1: Extract document taxonomy (1 LLM call)
        taxonomy = DocumentTaxonomy.default()
        if self.config.enable_categories:
            logger.info("Step 1: Extracting document taxonomy...")
            taxonomy = self.category_extractor.extract_taxonomy_sync(document)
            self._stats["category_calls"] += 1
            logger.info(f"  → Primary category: {taxonomy.primary_category}")

        # Step 2: Extract section keywords (~100 LLM calls)
        section_keywords: Dict[str, SectionKeywords] = {}
        if self.config.enable_keywords:
            logger.info(
                f"Step 2: Extracting keywords for {len(document.sections)} sections..."
            )
            section_keywords = self.keyword_extractor.extract_batch_sync(
                document.sections, taxonomy.primary_category
            )
            self._stats["keyword_calls"] += len(document.sections)
            logger.info(f"  → Extracted keywords for {len(section_keywords)} sections")

        # Step 3: Generate chunk questions (per-chunk)
        chunk_questions: Dict[str, ChunkQuestions] = {}
        if self.config.enable_questions:
            logger.info(f"Step 3: Generating questions for {len(chunks)} chunks...")
            for chunk in chunks:
                questions = self.question_generator.generate_sync(
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.raw_content or chunk.content,
                    document_title=document.title or "Unknown",
                    section_path=chunk.metadata.section_path or "",
                    category=taxonomy.primary_category,
                )
                chunk_questions[chunk.chunk_id] = questions
            self._stats["question_calls"] += len(chunks)
            logger.info(f"  → Generated questions for {len(chunk_questions)} chunks")

        self._stats["documents_processed"] += 1

        return LabelingResult(
            taxonomy=taxonomy,
            section_keywords=section_keywords,
            chunk_questions=chunk_questions,
            stats=self._stats.copy(),
        )

    async def label_document_async(
        self,
        document: ExtractedDocument,
        chunks: List[Chunk],
    ) -> LabelingResult:
        """
        Label document asynchronously.

        Uses asyncio for parallel processing where possible.

        Args:
            document: Extracted document
            chunks: Document chunks

        Returns:
            Complete labeling result
        """
        import asyncio

        logger.info(
            f"Labeling document (async): {len(document.sections)} sections, "
            f"{len(chunks)} chunks"
        )

        # Step 1: Extract document taxonomy (1 LLM call)
        taxonomy = DocumentTaxonomy.default()
        if self.config.enable_categories:
            logger.info("Step 1: Extracting document taxonomy...")
            taxonomy = await self.category_extractor.extract_taxonomy_async(document)
            self._stats["category_calls"] += 1
            logger.info(f"  → Primary category: {taxonomy.primary_category}")

        # Step 2: Extract section keywords (parallel)
        section_keywords: Dict[str, SectionKeywords] = {}
        if self.config.enable_keywords:
            logger.info(
                f"Step 2: Extracting keywords for {len(document.sections)} sections..."
            )
            tasks = [
                self.keyword_extractor.extract_async(section, taxonomy.primary_category)
                for section in document.sections
            ]
            results = await asyncio.gather(*tasks)
            for section, keywords in zip(document.sections, results):
                section_id = section.section_id or str(id(section))
                section_keywords[section_id] = keywords
            self._stats["keyword_calls"] += len(document.sections)
            logger.info(f"  → Extracted keywords for {len(section_keywords)} sections")

        # Step 3: Generate chunk questions (parallel)
        chunk_questions: Dict[str, ChunkQuestions] = {}
        if self.config.enable_questions:
            logger.info(f"Step 3: Generating questions for {len(chunks)} chunks...")
            tasks = [
                self.question_generator.generate_async(
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.raw_content or chunk.content,
                    document_title=document.title or "Unknown",
                    section_path=chunk.metadata.section_path or "",
                    category=taxonomy.primary_category,
                )
                for chunk in chunks
            ]
            results = await asyncio.gather(*tasks)
            for chunk, questions in zip(chunks, results):
                chunk_questions[chunk.chunk_id] = questions
            self._stats["question_calls"] += len(chunks)
            logger.info(f"  → Generated questions for {len(chunk_questions)} chunks")

        self._stats["documents_processed"] += 1

        return LabelingResult(
            taxonomy=taxonomy,
            section_keywords=section_keywords,
            chunk_questions=chunk_questions,
            stats=self._stats.copy(),
        )

    def apply_labels_to_chunks(
        self,
        chunks: List[Chunk],
        result: LabelingResult,
        augment_embedding: bool = True,
    ) -> List[Chunk]:
        """
        Apply labeling result to chunks.

        Propagates labels hierarchically and optionally augments embedding_text.

        Args:
            chunks: Chunks to label
            result: Labeling result
            augment_embedding: Add questions to embedding_text (HyDE boost)

        Returns:
            Labeled chunks (modified in place)
        """
        for chunk in chunks:
            # 1. Apply category (propagated from document)
            category_metadata = self.category_extractor.propagate_to_chunk(
                result.taxonomy, chunk.metadata.section_path
            )
            chunk.metadata.category = category_metadata.get("category")
            chunk.metadata.subcategory = category_metadata.get("subcategory")
            chunk.metadata.category_confidence = category_metadata.get(
                "category_confidence"
            )

            # 2. Apply keywords (propagated from section)
            section_id = chunk.metadata.section_id
            if section_id and section_id in result.section_keywords:
                section_kw = result.section_keywords[section_id]
                kw_metadata = self.keyword_extractor.propagate_to_chunk(
                    section_kw, chunk.raw_content or chunk.content
                )
                chunk.metadata.keywords = kw_metadata.get("keywords")
                chunk.metadata.key_phrases = kw_metadata.get("key_phrases")

            # 3. Apply questions (generated per-chunk)
            if chunk.chunk_id in result.chunk_questions:
                questions = result.chunk_questions[chunk.chunk_id]
                chunk.metadata.questions = questions.questions
                chunk.metadata.hyde_text = questions.hyde_text

                # 4. Augment embedding_text with questions (HyDE boost)
                if (
                    augment_embedding
                    and self.config.include_questions_in_embedding
                    and questions.hyde_text
                ):
                    chunk.embedding_text = self.question_generator.augment_embedding_text(
                        chunk.embedding_text, questions
                    )

            # Mark labeling source
            chunk.metadata.labels_source = "generated"

        logger.info(f"Applied labels to {len(chunks)} chunks")
        return chunks

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "question_generator": self.question_generator.get_stats(),
        }


def create_labeling_pipeline(config: Optional[LabelingConfig] = None) -> LabelingPipeline:
    """
    Factory function for creating labeling pipeline.

    Args:
        config: Optional configuration

    Returns:
        Configured LabelingPipeline
    """
    return LabelingPipeline(config)
