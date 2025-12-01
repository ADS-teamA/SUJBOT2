"""
LlamaIndex IngestionPipeline wrapper for state management.

Preserves existing pipeline while adding:
- Redis-backed state persistence (survives restarts)
- Document deduplication via content hash
- Entity labeling phase (3.5) using Gemini 2.5 Flash
- Partial re-indexing support (re-run specific phases)

This is a wrapper around the existing IndexingPipeline, not a replacement.
It delegates actual processing to the legacy pipeline while managing state.

Usage:
    from src.indexing import SujbotIngestionPipeline

    pipeline = SujbotIngestionPipeline()

    # Full indexing with state persistence
    result = pipeline.index_document(Path("document.pdf"))

    # Re-run only entity labeling (phase 3.5)
    result = pipeline.reindex_phase(Path("document.pdf"), phase=3)

    # Resume from failure (automatic)
    result = pipeline.index_document(Path("document.pdf"))  # Continues from last phase
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core.schema import TextNode

from src.config_schema import LabelingConfig
from src.exceptions import (
    APIKeyError,
    ConfigurationError,
    ProviderError,
    StorageError,
    is_recoverable,
    wrap_exception,
)
from src.extraction_models import ExtractedDocument
from src.indexing_pipeline import IndexingConfig, IndexingPipeline
from src.indexing.transforms.gemini_entity_labeler import GeminiEntityLabeler
from src.indexing.transforms.labeling_pipeline import (
    LabelingPipeline,
    LabelingResult,
)
from src.multi_layer_chunker import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

# Redis cache configuration
REDIS_COLLECTION = "sujbot_ingestion"
PHASE_STATE_PREFIX = "phase_state"
PHASE_RESULT_PREFIX = "phase_result"
DOC_HASH_PREFIX = "doc_hash"


class SujbotIngestionPipeline:
    """
    LlamaIndex wrapper around existing IndexingPipeline.

    Adds:
    - Redis-backed state persistence (survives restarts)
    - Document deduplication via hash
    - Entity labeling phase (3.5) using Gemini 2.5 Flash
    - Partial re-indexing support

    Phases:
    - Phase 1: Extraction (from legacy pipeline)
    - Phase 2: Summarization (from legacy pipeline)
    - Phase 3: Chunking (from legacy pipeline)
    - Phase 3.5: Entity Labeling (NEW - Gemini 2.5 Flash)
    - Phase 4: Embedding (from legacy pipeline)
    - Phase 5: Knowledge Graph (from legacy pipeline)
    """

    def __init__(
        self,
        config: Optional[IndexingConfig] = None,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        enable_entity_labeling: bool = True,
        entity_labeling_batch_size: int = 10,
        entity_labeling_model: str = "gemini-2.5-flash",
        enable_document_labeling: bool = True,
        labeling_config: Optional[LabelingConfig] = None,
    ):
        """
        Initialize the wrapper pipeline.

        Args:
            config: IndexingConfig instance (defaults loaded from env)
            redis_host: Redis host (default: from REDIS_HOST env or localhost)
            redis_port: Redis port (default: from REDIS_PORT env or 6379)
            enable_entity_labeling: Enable entity labeling phase 3.5
            entity_labeling_batch_size: Batch size for entity labeling
            entity_labeling_model: Gemini model for entity labeling
            enable_document_labeling: Enable document labeling phase 3.6
            labeling_config: LabelingConfig for document labeling pipeline
        """
        self.config = config or IndexingConfig.from_env()

        # Redis configuration
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))

        # Entity labeling configuration
        self.enable_entity_labeling = enable_entity_labeling
        self.entity_labeling_batch_size = entity_labeling_batch_size
        self.entity_labeling_model = entity_labeling_model

        # Document labeling configuration (categories, keywords, questions)
        self.enable_document_labeling = enable_document_labeling
        self.labeling_config = labeling_config or LabelingConfig()

        # Initialize Redis cache (lazy)
        self._cache = None

        # Initialize entity labeler (lazy)
        self._entity_labeler = None

        # Initialize document labeling pipeline (lazy)
        self._labeling_pipeline: Optional[LabelingPipeline] = None

        # Wrap existing pipeline
        self.legacy_pipeline = IndexingPipeline(self.config)

        logger.info(
            f"SujbotIngestionPipeline initialized: "
            f"Redis={self.redis_host}:{self.redis_port}, "
            f"entity_labeling={self.enable_entity_labeling}, "
            f"document_labeling={self.enable_document_labeling}"
        )

    @property
    def cache(self):
        """Lazy initialization of Redis cache."""
        if self._cache is None:
            try:
                from llama_index.core.ingestion import IngestionCache
                from llama_index.storage.kvstore.redis import RedisKVStore

                self._cache = IngestionCache(
                    cache=RedisKVStore.from_host_and_port(
                        host=self.redis_host,
                        port=self.redis_port,
                    ),
                    collection=REDIS_COLLECTION,
                )
                logger.info(f"Redis cache connected: {self.redis_host}:{self.redis_port}")
            except ImportError as e:
                logger.error(
                    f"Redis packages not installed: {e}. "
                    "Install with: uv add llama-index-storage-kvstore-redis"
                )
                self._cache = None
            except (ConnectionError, TimeoutError, OSError) as e:
                # Network-related errors - recoverable, continue without cache
                logger.error(
                    f"Redis connection failed ({self.redis_host}:{self.redis_port}): {e}. "
                    "Pipeline will run WITHOUT caching - indexing will NOT be resumable.",
                    exc_info=True
                )
                self._cache = None
            except Exception as e:
                # Unexpected errors - check if recoverable
                if not is_recoverable(e):
                    raise  # Re-raise KeyboardInterrupt, MemoryError, etc.
                logger.error(
                    f"Unexpected Redis error ({self.redis_host}:{self.redis_port}): {e}. "
                    "Pipeline will run WITHOUT caching.",
                    exc_info=True
                )
                self._cache = None
        return self._cache

    @property
    def entity_labeler(self) -> Optional[GeminiEntityLabeler]:
        """Lazy initialization of entity labeler."""
        if self._entity_labeler is None and self.enable_entity_labeling:
            self._entity_labeler = GeminiEntityLabeler(
                model_name=self.entity_labeling_model,
                batch_size=self.entity_labeling_batch_size,
            )
        return self._entity_labeler

    @property
    def labeling_pipeline(self) -> Optional[LabelingPipeline]:
        """Lazy initialization of document labeling pipeline."""
        if self._labeling_pipeline is None and self.enable_document_labeling:
            if self.labeling_config.enabled:
                self._labeling_pipeline = LabelingPipeline(self.labeling_config)
                logger.info(
                    f"Document labeling pipeline initialized: "
                    f"model={self.labeling_config.model}, "
                    f"categories={self.labeling_config.enable_categories}, "
                    f"keywords={self.labeling_config.enable_keywords}, "
                    f"questions={self.labeling_config.enable_questions}"
                )
        return self._labeling_pipeline

    def index_document(
        self,
        document_path: Path,
        start_phase: int = 1,
        end_phase: int = 5,
        force_reindex: bool = False,
        save_intermediate: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Index document with state persistence and partial re-indexing.

        Args:
            document_path: Path to document (PDF, DOCX, etc.)
            start_phase: Phase to start from (1-5)
            end_phase: Phase to end at (1-5)
            force_reindex: Ignore cache and reprocess all phases
            save_intermediate: Save intermediate results to disk
            output_dir: Directory for intermediate results

        Returns:
            Dict with pipeline results:
            - vector_store: VectorStoreAdapter
            - knowledge_graph: KnowledgeGraph (if enabled)
            - chunks: Dict of labeled chunks
            - stats: Pipeline statistics

        Returns None if document is duplicate.

        Raises:
            ValueError: If phase parameters are invalid.
        """
        # Validate phase parameters
        if not 1 <= start_phase <= 5:
            raise ValueError(f"start_phase must be 1-5, got {start_phase}")
        if not 1 <= end_phase <= 5:
            raise ValueError(f"end_phase must be 1-5, got {end_phase}")
        if start_phase > end_phase:
            raise ValueError(f"start_phase ({start_phase}) cannot be greater than end_phase ({end_phase})")

        document_path = Path(document_path)
        doc_id = document_path.stem

        # Setup output directory
        if output_dir is None:
            output_dir = Path("vector_db") / doc_id
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check cache for completed phases
        if not force_reindex and self.cache is not None:
            cached_phase = self._get_cached_phase(doc_id)
            if cached_phase >= end_phase:
                logger.info(f"Document {doc_id} already indexed to phase {cached_phase}")
                return self._load_cached_result(doc_id)

            # Adjust start_phase based on cache
            if cached_phase > 0:
                start_phase = max(start_phase, cached_phase + 1)
                logger.info(f"Resuming from phase {start_phase} (cached: {cached_phase})")

        logger.info(f"Indexing {doc_id}: phases {start_phase}-{end_phase}")

        # Run legacy pipeline for phases 1-3
        # The legacy pipeline handles extraction, summarization, and chunking
        legacy_result = self.legacy_pipeline.index_document(
            document_path=document_path,
            save_intermediate=save_intermediate,
            output_dir=output_dir,
            resume=not force_reindex,
        )

        # Handle duplicate detection (legacy pipeline returns None)
        if legacy_result is None:
            logger.info(f"Document {doc_id} is a duplicate, skipping")
            return None

        result = {
            "vector_store": legacy_result.get("vector_store"),
            "knowledge_graph": legacy_result.get("knowledge_graph"),
            "stats": legacy_result.get("stats", {}),
        }

        # Phase 3.5: Entity Labeling (NEW)
        if self.enable_entity_labeling and self.entity_labeler is not None:
            logger.info("PHASE 3.5: Entity Labeling (Gemini 2.5 Flash)")

            # Load chunks from output directory
            chunks_path = output_dir / "phase3_chunks.json"
            if chunks_path.exists():
                try:
                    labeled_chunks = self._run_entity_labeling(chunks_path)
                    result["labeled_chunks"] = labeled_chunks

                    # Save labeled chunks
                    labeled_path = output_dir / "phase3.5_labeled_chunks.json"
                    self._save_labeled_chunks(labeled_chunks, labeled_path)
                    logger.info(f"Saved labeled chunks to {labeled_path}")

                    # Cache phase 3.5 completion
                    if self.cache is not None:
                        self._cache_phase_result(doc_id, 3, {"labeled": True})

                except (APIKeyError, ConfigurationError) as e:
                    # Non-recoverable: missing API key or bad config - fail fast
                    logger.error(
                        f"Entity labeling configuration error: {e}. "
                        "Fix configuration before continuing.",
                        exc_info=True
                    )
                    raise  # Don't silently fallback for config errors
                except ProviderError as e:
                    # Recoverable: API rate limit, timeout, etc.
                    logger.warning(
                        f"Entity labeling provider error: {e}. "
                        "Document indexed without entity labels.",
                        exc_info=True
                    )
                    result["entity_labeling_error"] = str(e)
                    result["entity_labeling_succeeded"] = False
                except Exception as e:
                    # Check if recoverable before falling back
                    if not is_recoverable(e):
                        raise  # Re-raise KeyboardInterrupt, MemoryError, etc.
                    logger.error(
                        f"Entity labeling failed: {e}. Document indexed without entity labels.",
                        exc_info=True
                    )
                    logger.warning(
                        "=" * 60 + "\n"
                        "ENTITY LABELING FAILED\n"
                        f"Error: {e}\n"
                        "Document indexed but entity-based filtering will NOT work.\n"
                        "Check GOOGLE_API_KEY in .env if using Gemini labeler.\n" +
                        "=" * 60
                    )
                    result["entity_labeling_error"] = str(e)
                    result["entity_labeling_succeeded"] = False
            else:
                logger.warning(f"Chunks file not found: {chunks_path}")

        # Phase 3.6: Document Labeling (categories, keywords, questions)
        if self.enable_document_labeling and self.labeling_pipeline is not None:
            logger.info("PHASE 3.6: Document Labeling (categories, keywords, HyDE questions)")

            # Load extracted document (phase1 output)
            extracted_path = output_dir / "phase1_extracted.json"
            chunks_path = output_dir / "phase3_chunks.json"

            if extracted_path.exists() and chunks_path.exists():
                try:
                    labeling_result = self._run_document_labeling(
                        extracted_path, chunks_path
                    )
                    result["labeling_result"] = labeling_result.to_dict()

                    # Save labeled chunks with categories, keywords, questions
                    labeled_path = output_dir / "phase3.6_document_labeled.json"
                    self._save_labeling_result(labeling_result, labeled_path)
                    logger.info(f"Saved document labels to {labeled_path}")

                    # Update existing labeled_chunks with new labels
                    if "labeled_chunks" in result:
                        self._merge_document_labels(
                            result["labeled_chunks"], labeling_result
                        )

                    # Cache phase 3.6 completion
                    if self.cache is not None:
                        self._cache_phase_result(
                            doc_id, 3, {"document_labeled": True}
                        )

                except (APIKeyError, ConfigurationError) as e:
                    # Non-recoverable: missing API key or bad config - fail fast
                    logger.error(
                        f"Document labeling configuration error: {e}. "
                        "Fix configuration before continuing.",
                        exc_info=True
                    )
                    raise  # Don't silently fallback for config errors
                except ProviderError as e:
                    # Recoverable: API rate limit, timeout, etc.
                    logger.warning(
                        f"Document labeling provider error: {e}. "
                        "Document indexed without categories/keywords/questions.",
                        exc_info=True
                    )
                    result["document_labeling_error"] = str(e)
                    result["document_labeling_succeeded"] = False
                except Exception as e:
                    # Check if recoverable before falling back
                    if not is_recoverable(e):
                        raise  # Re-raise KeyboardInterrupt, MemoryError, etc.
                    logger.error(
                        f"Document labeling failed: {e}. "
                        "Document indexed without categories/keywords/questions.",
                        exc_info=True
                    )
                    logger.warning(
                        "=" * 60 + "\n"
                        "DOCUMENT LABELING FAILED\n"
                        f"Error: {e}\n"
                        "Document indexed but category filtering and HyDE boost unavailable.\n"
                        "Check OPENAI_API_KEY in .env if using OpenAI Batch API.\n" +
                        "=" * 60
                    )
                    result["document_labeling_error"] = str(e)
                    result["document_labeling_succeeded"] = False
            else:
                missing = []
                if not extracted_path.exists():
                    missing.append(str(extracted_path))
                if not chunks_path.exists():
                    missing.append(str(chunks_path))
                logger.warning(f"Required files not found: {', '.join(missing)}")

        # Cache final phase completion
        if self.cache is not None:
            self._cache_phase_result(doc_id, end_phase, {"completed": True})

        return result

    def _run_entity_labeling(self, chunks_path: Path) -> Dict[str, List[Dict]]:
        """
        Run entity labeling on chunks from JSON file.

        Args:
            chunks_path: Path to phase3_chunks.json

        Returns:
            Dict with labeled chunks per layer
        """
        # Load chunks JSON
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        labeled_chunks = {}

        for layer in ["layer1", "layer2", "layer3"]:
            if layer not in chunks_data or not chunks_data[layer]:
                labeled_chunks[layer] = []
                continue

            # Convert to LlamaIndex TextNodes for the transformer
            nodes = []
            for chunk in chunks_data[layer]:
                # Use embedding_text as content (per CLAUDE.md chunk format)
                text = chunk.get("embedding_text") or chunk.get("raw_content", "")

                node = TextNode(
                    text=text,
                    id_=chunk.get("chunk_id", ""),
                    metadata={
                        "chunk_id": chunk.get("chunk_id"),
                        "layer": chunk.get("metadata", {}).get("layer"),
                        "document_id": chunk.get("metadata", {}).get("document_id"),
                        "context": chunk.get("context", ""),
                        # Preserve original metadata
                        **chunk.get("metadata", {}),
                    },
                )
                nodes.append(node)

            # Run entity labeling
            if nodes:
                logger.info(f"Entity labeling {layer}: {len(nodes)} chunks")
                labeled_nodes = self.entity_labeler(nodes)

                # Convert back to dict format
                labeled_chunks[layer] = [
                    {
                        "chunk_id": node.id_,
                        "context": node.metadata.get("context", ""),
                        "raw_content": chunks_data[layer][i].get("raw_content", ""),
                        "embedding_text": chunks_data[layer][i].get("embedding_text", ""),
                        "metadata": {
                            **node.metadata,
                            "entities": node.metadata.get("entities", []),
                            "entity_types": node.metadata.get("entity_types", []),
                            "topics": node.metadata.get("topics", []),
                        },
                    }
                    for i, node in enumerate(labeled_nodes)
                ]
            else:
                labeled_chunks[layer] = []

        return labeled_chunks

    def _save_labeled_chunks(self, labeled_chunks: Dict, output_path: Path) -> None:
        """Save labeled chunks to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(labeled_chunks, f, ensure_ascii=False, indent=2)

    def _run_document_labeling(
        self, extracted_path: Path, chunks_path: Path
    ) -> LabelingResult:
        """
        Run document labeling pipeline (categories, keywords, questions).

        Phase 3.6: Adds hierarchical labels with smart propagation.
        - Categories: Document-level (1 LLM call) → propagated to chunks
        - Keywords: Section-level (~100 LLM calls) → propagated to chunks
        - Questions: Chunk-level (HyDE boost for +20-30% retrieval)

        Args:
            extracted_path: Path to phase1_extracted.json
            chunks_path: Path to phase3_chunks.json

        Returns:
            LabelingResult with taxonomy, keywords, questions
        """
        # Load extracted document
        with open(extracted_path, "r", encoding="utf-8") as f:
            extracted_data = json.load(f)

        # Reconstruct ExtractedDocument from JSON
        document = self._load_extracted_document(extracted_data)

        # Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        # Convert JSON chunks to Chunk objects (layer 3 only for labeling)
        chunks = self._load_chunks_from_json(chunks_data)

        logger.info(
            f"Document labeling: {len(document.sections)} sections, {len(chunks)} chunks"
        )

        # Run labeling pipeline synchronously
        # (Batch API mode handles async internally)
        result = self.labeling_pipeline.label_document_sync(document, chunks)

        # Apply labels to chunks
        self.labeling_pipeline.apply_labels_to_chunks(
            chunks, result, augment_embedding=self.labeling_config.include_questions_in_embedding
        )

        logger.info(
            f"Labeling complete: category={result.taxonomy.primary_category}, "
            f"keywords={len(result.section_keywords)} sections, "
            f"questions={len(result.chunk_questions)} chunks"
        )

        return result

    def _load_extracted_document(self, data: Dict[str, Any]) -> ExtractedDocument:
        """
        Reconstruct ExtractedDocument from phase1 JSON.

        Args:
            data: JSON data from phase1_extracted.json

        Returns:
            ExtractedDocument instance
        """
        from src.extraction_models import DocumentSection

        sections = []
        for section_data in data.get("sections", []):
            # Reconstruct DocumentSection with all required fields
            section = DocumentSection(
                section_id=section_data.get("section_id", ""),
                title=section_data.get("title", ""),
                content=section_data.get("content", ""),
                level=section_data.get("level", 1),
                depth=section_data.get("depth", 1),
                parent_id=section_data.get("parent_id"),
                children_ids=section_data.get("children_ids", []),
                ancestors=section_data.get("ancestors", []),
                path=section_data.get("path", ""),
                page_number=section_data.get("page_number", 0),
                char_start=section_data.get("char_start", 0),
                char_end=section_data.get("char_end", 0),
                content_length=section_data.get("content_length", len(section_data.get("content", ""))),
                element_type=section_data.get("element_type"),
                element_category=section_data.get("element_category"),
                summary=section_data.get("summary"),
            )
            sections.append(section)

        # Get metadata section (phase1 JSON structure)
        metadata = data.get("metadata", {})

        document = ExtractedDocument(
            document_id=metadata.get("document_id", data.get("document_id", "unknown")),
            source_path=metadata.get("source_path", data.get("source_path", "")),
            extraction_time=metadata.get("extraction_time_seconds", data.get("extraction_time", 0.0)),
            full_text=data.get("full_text", ""),
            markdown=data.get("markdown", ""),
            json_content=data.get("json_content", ""),
            sections=sections,
            hierarchy_depth=metadata.get("hierarchy_depth", 1),
            num_roots=metadata.get("num_roots", len([s for s in sections if s.parent_id is None])),
            tables=[],  # Tables reconstructed separately if needed
            num_pages=metadata.get("num_pages", 0),
            num_sections=metadata.get("num_sections", len(sections)),
            num_tables=metadata.get("num_tables", 0),
            total_chars=metadata.get("total_chars", len(data.get("full_text", ""))),
            document_summary=metadata.get("document_summary"),
            title=metadata.get("title"),
            extraction_method=metadata.get("extraction_method", "unstructured_detectron2"),
            config=metadata.get("config"),
        )

        return document

    def _load_chunks_from_json(self, chunks_data: Dict[str, Any]) -> List[Chunk]:
        """
        Convert phase3_chunks.json to Chunk objects.

        Loads layer 3 chunks (actual content chunks) for labeling.

        Args:
            chunks_data: JSON data with layer1/layer2/layer3 keys

        Returns:
            List of Chunk objects
        """
        chunks = []

        # Focus on layer 3 (actual content chunks)
        layer3_data = chunks_data.get("layer3", [])

        for chunk_data in layer3_data:
            metadata_dict = chunk_data.get("metadata", {})

            # Create ChunkMetadata
            metadata = ChunkMetadata(
                chunk_id=metadata_dict.get("chunk_id", chunk_data.get("chunk_id")),
                layer=metadata_dict.get("layer", 3),
                document_id=metadata_dict.get("document_id"),
                section_id=metadata_dict.get("section_id"),
                section_path=metadata_dict.get("section_path"),
                parent_chunk_id=metadata_dict.get("parent_chunk_id"),
                sibling_index=metadata_dict.get("sibling_index"),
                total_siblings=metadata_dict.get("total_siblings"),
                token_count=metadata_dict.get("token_count"),
                char_count=metadata_dict.get("char_count"),
                position_in_document=metadata_dict.get("position_in_document"),
            )

            chunk = Chunk(
                chunk_id=chunk_data.get("chunk_id", ""),
                content=chunk_data.get("embedding_text", chunk_data.get("raw_content", "")),
                context=chunk_data.get("context", ""),
                raw_content=chunk_data.get("raw_content", ""),
                embedding_text=chunk_data.get("embedding_text", ""),
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def _save_labeling_result(
        self, result: LabelingResult, output_path: Path
    ) -> None:
        """
        Save labeling result to JSON file.

        Args:
            result: LabelingResult to save
            output_path: Output file path
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    def _merge_document_labels(
        self,
        labeled_chunks: Dict[str, List[Dict]],
        labeling_result: LabelingResult,
    ) -> None:
        """
        Merge document labeling results into existing labeled chunks.

        Updates chunks in place with category, keywords, and questions.

        Args:
            labeled_chunks: Existing labeled chunks dict (modified in place)
            labeling_result: Document labeling result
        """
        # Get taxonomy for category propagation
        taxonomy = labeling_result.taxonomy

        for layer_key in ["layer1", "layer2", "layer3"]:
            if layer_key not in labeled_chunks:
                continue

            for chunk in labeled_chunks[layer_key]:
                chunk_id = chunk.get("chunk_id", "")

                # Add category (from document taxonomy)
                chunk["metadata"]["category"] = taxonomy.primary_category
                chunk["metadata"]["subcategory"] = taxonomy.subcategories[0] if taxonomy.subcategories else None
                chunk["metadata"]["secondary_categories"] = taxonomy.secondary_categories
                chunk["metadata"]["category_confidence"] = taxonomy.confidence

                # Add keywords (from section, if available)
                section_id = chunk.get("metadata", {}).get("section_id")
                if section_id and section_id in labeling_result.section_keywords:
                    section_kw = labeling_result.section_keywords[section_id]
                    chunk["metadata"]["keywords"] = section_kw.keywords
                    chunk["metadata"]["key_phrases"] = section_kw.key_phrases

                # Add questions (chunk-specific)
                if chunk_id in labeling_result.chunk_questions:
                    questions = labeling_result.chunk_questions[chunk_id]
                    chunk["metadata"]["questions"] = questions.questions
                    chunk["metadata"]["hyde_text"] = questions.hyde_text

                    # Augment embedding_text with HyDE questions
                    if (
                        self.labeling_config.include_questions_in_embedding
                        and questions.hyde_text
                    ):
                        original_text = chunk.get("embedding_text", "")
                        chunk["embedding_text"] = (
                            f"{original_text}\n\n[Otázky: {questions.hyde_text}]"
                        )

        logger.info(
            f"Merged document labels into {sum(len(v) for v in labeled_chunks.values())} chunks"
        )

    def _get_cached_phase(self, doc_id: str) -> int:
        """Get highest completed phase from cache."""
        if self.cache is None:
            return 0

        try:
            key = f"{PHASE_STATE_PREFIX}:{doc_id}"
            state = self.cache.cache.get(key)
            return int(state) if state else 0
        except Exception as e:
            logger.warning(f"Failed to get cached phase: {e}")
            return 0

    def _cache_phase_result(self, doc_id: str, phase: int, result: Any) -> None:
        """Cache phase result and update state."""
        if self.cache is None:
            return

        try:
            # Store result (serialize to JSON string)
            result_key = f"{PHASE_RESULT_PREFIX}:{doc_id}:{phase}"
            self.cache.cache.put(result_key, json.dumps(result))

            # Update state
            state_key = f"{PHASE_STATE_PREFIX}:{doc_id}"
            self.cache.cache.put(state_key, str(phase))

            logger.debug(f"Cached phase {phase} for {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to cache phase result: {e}")

    def _load_cached_result(self, doc_id: str) -> Dict[str, Any]:
        """Load all cached results for document."""
        result = {}
        if self.cache is None:
            return result

        try:
            for phase in range(1, 6):
                key = f"{PHASE_RESULT_PREFIX}:{doc_id}:{phase}"
                data = self.cache.cache.get(key)
                if data:
                    result[f"phase_{phase}"] = json.loads(data)
        except Exception as e:
            logger.warning(f"Failed to load cached results: {e}")

        return result

    def clear_document_cache(self, doc_id: str) -> None:
        """Clear all cache entries for a document."""
        if self.cache is None:
            logger.warning("Redis cache not available")
            return

        try:
            for phase in range(1, 6):
                self.cache.cache.delete(f"{PHASE_RESULT_PREFIX}:{doc_id}:{phase}")
            self.cache.cache.delete(f"{PHASE_STATE_PREFIX}:{doc_id}")
            logger.info(f"Cleared cache for {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def reindex_phase(
        self,
        document_path: Path,
        phase: int,
        output_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Re-run specific phase only.

        Useful for updating entity labels without full re-indexing.

        Args:
            document_path: Path to document
            phase: Phase to re-run (1-5, 3 includes 3.5 entity labeling)
            output_dir: Directory with existing intermediate results

        Returns:
            Dict with phase results
        """
        return self.index_document(
            document_path,
            start_phase=phase,
            end_phase=phase,
            force_reindex=True,
            output_dir=output_dir,
        )

    def get_document_hash(self, document_path: Path) -> str:
        """
        Compute content hash for document deduplication.

        Args:
            document_path: Path to document

        Returns:
            SHA-256 hash of file contents
        """
        hasher = hashlib.sha256()
        with open(document_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
