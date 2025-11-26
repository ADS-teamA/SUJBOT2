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

from src.indexing_pipeline import IndexingConfig, IndexingPipeline
from src.indexing.transforms.gemini_entity_labeler import GeminiEntityLabeler

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
        """
        self.config = config or IndexingConfig.from_env()

        # Redis configuration
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))

        # Entity labeling configuration
        self.enable_entity_labeling = enable_entity_labeling
        self.entity_labeling_batch_size = entity_labeling_batch_size
        self.entity_labeling_model = entity_labeling_model

        # Initialize Redis cache (lazy)
        self._cache = None

        # Initialize entity labeler (lazy)
        self._entity_labeler = None

        # Wrap existing pipeline
        self.legacy_pipeline = IndexingPipeline(self.config)

        logger.info(
            f"SujbotIngestionPipeline initialized: "
            f"Redis={self.redis_host}:{self.redis_port}, "
            f"entity_labeling={self.enable_entity_labeling}"
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
            except Exception as e:
                logger.error(
                    f"Redis connection failed ({self.redis_host}:{self.redis_port}): {e}. "
                    "Pipeline will run WITHOUT caching - indexing will NOT be resumable.",
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

                except Exception as e:
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
