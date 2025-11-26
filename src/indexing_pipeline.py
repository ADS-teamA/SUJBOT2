"""
PHASE 1-6: Complete Indexing Pipeline

Orchestrates:
1. PHASE 1-3: Extraction, hierarchy, summaries, chunking
2. PHASE 4: Embedding generation + PostgreSQL pgvector indexing
3. PHASE 5A: Knowledge Graph construction (optional)
4. PHASE 5C: Cross-Encoder Reranking (optional)
5. PHASE 5D: Graph-Vector Integration (optional)
6. PHASE 6: Context Assembly for LLM (optional)

Supported formats: PDF, DOCX, PPTX, XLSX, HTML

Based on research:
- LegalBench-RAG: Multi-Layer Embeddings (3 separate indexes)
- HyDE: Gao et al. (2022) - Hypothetical Document Embeddings
- Query Expansion: Vocabulary coverage via paraphrasing
- Weighted Fusion: w_hyde=0.6, w_exp=0.4 (empirically optimized)

Embedding model: Qwen3-Embedding-8B (4096 dims) via DeepInfra
Storage: PostgreSQL with pgvector extension

Usage:
    pipeline = IndexingPipeline(
        embedding_model="Qwen/Qwen3-Embedding-8B",
        enable_sac=True,
        enable_knowledge_graph=True,
    )

    result = pipeline.index_document("document.pdf")
    result["vector_store"].save("output/vector_store")
    result["knowledge_graph"].save_json("output/knowledge_graph.json")
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import numpy as np

from src.config import (
    ExtractionConfig,
    SummarizationConfig,
    ChunkingConfig,
    EmbeddingConfig,
)
from src.unified_extraction_pipeline import UnifiedDocumentPipeline
from src.multi_layer_chunker import MultiLayerChunker
from src.embedding_generator import EmbeddingGenerator
from src.storage import create_vector_store_adapter, load_vector_store_adapter
from src.cost_tracker import get_global_tracker, reset_global_tracker

# Knowledge Graph imports (optional)
try:
    from src.graph import (
        KnowledgeGraphConfig,
        EntityExtractionConfig as KGEntityConfig,
        RelationshipExtractionConfig as KGRelationshipConfig,
        GraphStorageConfig,
        GraphBackend,
    )
    from src.graph.graphiti_extractor import GraphitiExtractor, GraphitiExtractionResult

    KG_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False

logger = logging.getLogger(__name__)


def _make_relative_path(path: Path) -> str:
    """
    Convert path to relative path from current working directory.

    Ensures portability across different machines/environments.

    Args:
        path: Path object (can be absolute or relative)

    Returns:
        String representation of relative path
    """
    try:
        # Try to make path relative to current working directory
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        # Path is not under current working directory, return as-is
        return str(path)


def check_existing_components(document_id: str, connection_string: str = None) -> Dict:
    """
    Check what components already exist in PostgreSQL for this document.

    Args:
        document_id: Document ID to check
        connection_string: PostgreSQL connection string (from DATABASE_URL env var if not provided)

    Returns:
        Dict with:
            - exists: bool - Whether document exists in any component
            - has_dense: bool - Has vector embeddings in PostgreSQL
            - has_kg: bool - Has Knowledge Graph
            - existing_doc_id: str - Actual doc_id found (may differ slightly)
    """
    import asyncio

    result = {
        "exists": False,
        "has_dense": False,
        "has_kg": False,
        "existing_doc_id": None,
    }

    # Get connection string from environment if not provided
    if connection_string is None:
        connection_string = os.getenv("DATABASE_URL")

    if not connection_string:
        logger.warning("DATABASE_URL not set, cannot check existing components")
        return result

    # Check PostgreSQL for existing document
    try:
        import asyncpg

        async def check_postgres():
            conn = await asyncpg.connect(connection_string)
            try:
                # Check if document exists in chunks table (layer 3)
                query = """
                    SELECT DISTINCT document_id
                    FROM chunks
                    WHERE document_id = $1 OR document_id LIKE $2
                    LIMIT 1
                """
                row = await conn.fetchrow(query, document_id, f"{document_id}%")
                if row:
                    result["exists"] = True
                    result["has_dense"] = True
                    result["existing_doc_id"] = row["document_id"]
            finally:
                await conn.close()

        # Run async check
        asyncio.run(check_postgres())

    except ImportError:
        logger.warning("asyncpg not installed, cannot check PostgreSQL")
    except Exception as e:
        logger.warning(f"Failed to check PostgreSQL: {e}")

    # Check Knowledge Graph (in output directory)
    if result["existing_doc_id"]:
        output_dir = Path("output")
        kg_files = list(output_dir.glob(f"{result['existing_doc_id']}*/*/knowledge_graph.json"))
        if kg_files:
            result["has_kg"] = True

    return result


@dataclass
class IndexingConfig:
    """
    Configuration for complete indexing pipeline.

    Uses nested config objects for clean architecture. All sub-configs
    can be customized or loaded from environment via from_env().
    """

    # Speed/Cost Mode (determines Batch API usage)
    # "fast" = completions (2-3 min, full price) | "eco" = Batch API (15-30 min, 50% cheaper)
    speed_mode: str = "fast"  # "fast" or "eco"

    # Sub-configs (nested config objects)
    extraction_config: ExtractionConfig = field(default_factory=ExtractionConfig)
    summarization_config: SummarizationConfig = field(default_factory=SummarizationConfig)
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # PHASE 4.5: Semantic Clustering (optional)
    enable_semantic_clustering: bool = False  # Cluster chunks by semantic similarity
    clustering_config: Optional["ClusteringConfig"] = None

    # PHASE 5A: Knowledge Graph (enabled by default for SOTA 2025)
    enable_knowledge_graph: bool = True
    kg_config: Optional[KnowledgeGraphConfig] = None

    # PHASE 5C: Cross-Encoder Reranking (optional, not used with HyDE+Fusion)
    enable_reranking: bool = False  # Not used - HyDE+Fusion handles retrieval

    # PHASE 5D: Graph-Vector Integration (optional)
    enable_graph_retrieval: bool = False  # Graph-enhanced retrieval
    graph_boost_weight: float = 0.3  # Boost weight for graph matches
    enable_multi_hop: bool = False  # Multi-hop graph traversal

    # PHASE 6: Context Assembly (optional)
    enable_context_assembly: bool = False  # Assemble retrieved chunks for LLM
    citation_format: str = "inline"  # Citation style: inline, simple, detailed, footnote
    max_context_chunks: int = 6  # Max chunks to include in assembled context
    max_context_tokens: int = 4000  # Max tokens in assembled context (~16K chars)
    include_chunk_metadata: bool = True  # Include document/section/page metadata

    # Duplicate Detection (semantic similarity before indexing)
    enable_duplicate_detection: bool = True  # Detect semantic duplicates before indexing
    duplicate_similarity_threshold: float = 0.98  # 98% cosine similarity threshold
    duplicate_sample_pages: int = 1  # Pages to sample for detection (1 = first page)

    # Storage Backend Selection (PHASE 4) - PostgreSQL only
    storage_backend: str = "postgresql"  # PostgreSQL with pgvector (required)
    storage_layers: List[int] = field(default_factory=lambda: [1, 3])  # Which layers to embed/store

    def __post_init__(self):
        """Configure speed mode and validate settings."""
        # Validate speed mode
        if self.speed_mode not in ["fast", "eco"]:
            raise ValueError(f"speed_mode must be 'fast' or 'eco', got: {self.speed_mode}")

        # Configure summarization based on speed mode
        if self.speed_mode == "eco":
            # Eco mode: Use Batch API (50% cheaper, slower)
            self.summarization_config.use_batch_api = True
            self.summarization_config.batch_api_timeout = 43200  # 12 hours
            logger.info(f"💰 ECO MODE: Using Batch API (50% cost savings, 15-30 min latency)")
        else:  # fast mode
            # Fast mode: Use completions (full price, fast)
            self.summarization_config.use_batch_api = False
            logger.info(f"⚡ FAST MODE: Using completions (full price, 2-3 min latency)")

        # Initialize KG config if enabled
        if self.enable_knowledge_graph and self.kg_config is None:
            try:
                from src.graph.config import KnowledgeGraphConfig

                self.kg_config = KnowledgeGraphConfig.from_env()
            except ImportError:
                logger.warning("Knowledge Graph enabled but graph module not available")

        # Initialize clustering config if enabled
        if self.enable_semantic_clustering and self.clustering_config is None:
            try:
                from src.config import ClusteringConfig, get_config

                root_config = get_config()
                self.clustering_config = ClusteringConfig.from_config(root_config.clustering)
            except ImportError:
                logger.warning("Semantic clustering enabled but clustering module not available")

    @classmethod
    def from_env(cls, **overrides) -> "IndexingConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            SPEED_MODE: "fast" or "eco" (default: "fast")
            ENABLE_SEMANTIC_CLUSTERING: Enable semantic clustering (default: "false")
            ENABLE_KNOWLEDGE_GRAPH: Enable KG construction (default: "true")

        Args:
            **overrides: Override specific fields

        Returns:
            IndexingConfig instance loaded from environment
        """
        # Load validated config from config.json
        from src.config import get_config
        root_config = get_config()

        # PostgreSQL is the only supported backend
        storage_backend = "postgresql"

        # Load storage_layers from config.json (default: [1, 3] - skip sections)
        storage_layers = getattr(root_config.storage, 'storage_layers', [1, 3])

        # Create config with sub-configs loaded from config.json
        config = cls(
            speed_mode=root_config.summarization.speed_mode,
            extraction_config=ExtractionConfig.from_config(root_config.extraction),
            summarization_config=SummarizationConfig.from_config(root_config.summarization),
            chunking_config=ChunkingConfig.from_config(root_config.chunking),
            embedding_config=EmbeddingConfig.from_config(root_config.embedding),
            enable_semantic_clustering=root_config.clustering.enable_labels,
            enable_knowledge_graph=root_config.knowledge_graph.enable,
            enable_duplicate_detection=True,  # Always enabled
            duplicate_similarity_threshold=0.98,  # Default value
            duplicate_sample_pages=1,  # Default value
            storage_backend=storage_backend,
            storage_layers=storage_layers,
            **overrides,
        )

        return config


class IndexingPipeline:
    """
    Complete indexing pipeline for RAG system.

    Phases:
    1. Smart hierarchy extraction (font-size based)
    2. Generic summary generation (LLM)
    3. Multi-layer chunking + SAC (RCTS 512 tokens)
    4. Embedding + PostgreSQL pgvector indexing (3 separate indexes)
    5A. Knowledge Graph construction (optional)
    5D. Graph-vector integration (optional)
    6. Context assembly for LLM (optional)

    Retrieval (at query time):
    - HyDE: Hypothetical Document Embeddings
    - Query Expansion: 2 paraphrases for vocabulary coverage
    - Weighted Fusion: w_hyde=0.6, w_exp=0.4

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024)
    - Summary-Augmented Chunking (Reuter et al., 2024)
    - Multi-Layer Embeddings (Lima, 2024)
    - HyDE (Gao et al., 2022)
    - Context Assembly: SAC stripping + citations
    """

    def __init__(self, config: Optional[IndexingConfig] = None):
        """
        Initialize indexing pipeline.

        Args:
            config: IndexingConfig instance (defaults loaded from .env)
        """
        self.config = config or IndexingConfig.from_env()

        logger.info("Initializing IndexingPipeline...")

        # Initialize PHASE 1: Extraction (uses nested config)
        # UnifiedDocumentPipeline combines ToC retrieval + Unstructured extraction
        self.extractor = UnifiedDocumentPipeline(self.config.extraction_config)

        # Initialize PHASE 3: Chunking (uses nested config)
        self.chunker = MultiLayerChunker(config=self.config.chunking_config)

        # Initialize PHASE 4: Embedding (uses nested config)
        self.embedder = EmbeddingGenerator(self.config.embedding_config)

        # Initialize PHASE 5A: Knowledge Graph (optional, using Gemini)
        self.kg_extractor = None
        if self.config.enable_knowledge_graph:
            if not KG_AVAILABLE:
                logger.warning(
                    "Knowledge Graph requested but not available. "
                    "Install with: pip install google-generativeai"
                )
            else:
                self._initialize_kg_pipeline()

        logger.info(
            f"Pipeline initialized: "
            f"SAC={self.config.chunking_config.enable_contextual}, "
            f"model={self.config.embedding_config.model} "
            f"({self.embedder.dimensions}D), "
            f"KG={self.config.enable_knowledge_graph}, "
            f"Storage=PostgreSQL"
        )

    def _initialize_kg_pipeline(self):
        """Initialize Knowledge Graph extractor using Gemini 2.5 Pro."""
        if self.config.kg_config is None:
            logger.warning("Knowledge Graph enabled but no kg_config provided")
            self.kg_extractor = None
            return

        # Validate OpenAI API key for Graphiti (uses GPT-4o-mini)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found. Set OPENAI_API_KEY for Graphiti KG extraction.")
            self.kg_extractor = None
            return

        # Validate Neo4j password for Graphiti
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        if not neo4j_password:
            logger.warning("NEO4J_PASSWORD not found. Set NEO4J_PASSWORD for Graphiti KG extraction.")
            self.kg_extractor = None
            return

        # Initialize Graphiti extractor (uses GPT-4o-mini for extraction)
        try:
            model_name = os.getenv("GRAPHITI_MODEL", "gpt-4o-mini")
            batch_size = self.kg_config.batch_size if self.kg_config else 10
            self.kg_extractor = GraphitiExtractor(
                model_name=model_name,
                batch_size=batch_size,
            )
            logger.info(f"Knowledge Graph initialized: GraphitiExtractor ({model_name})")
        except Exception as e:
            logger.warning(f"Failed to initialize GraphitiExtractor: {e}")
            self.kg_extractor = None

    def index_document(
        self,
        document_path: Path,
        save_intermediate: bool = False,
        output_dir: Optional[Path] = None,
        resume: bool = True,
    ) -> Optional[Dict]:
        """
        Index a single document with automatic resume support.

        Supported formats: PDF, DOCX, PPTX, XLSX, HTML

        Complete pipeline:
        1. Extract with smart hierarchy (PHASE 1)
        2. Generate summaries (PHASE 2)
        3. Multi-layer chunking + SAC (PHASE 3)
        4. Embed + FAISS index (PHASE 4)
        5. Build knowledge graph (PHASE 5A - optional)

        Args:
            document_path: Path to document file (PDF, DOCX, PPTX, XLSX, HTML)
            save_intermediate: Save intermediate results (chunks, embeddings)
            output_dir: Directory for intermediate results
            resume: Enable resume from existing phases (default: True)

        Returns:
            Dict with pipeline results, or None if document is a duplicate and was skipped:
                - vector_store: VectorStoreAdapter (FAISS) with indexed document
                - knowledge_graph: KnowledgeGraph (if enabled, else None)
                - chunks: Dict of chunks (if save_intermediate)
                - stats: Pipeline statistics

            Returns None when duplicate detection is enabled and document is already indexed.
        """
        document_path = Path(document_path)

        if not document_path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        # Validate format
        supported_formats = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".txt"]
        if document_path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported format: {document_path.suffix}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        logger.info("=" * 80)
        logger.info(f"Indexing document: {document_path.name}")
        logger.info("=" * 80)

        # Reset cost tracker for this document
        reset_global_tracker()

        # RESUME DETECTION: Check for existing phases
        from src.phase_detector import PhaseDetector
        from src.phase_loaders import PhaseLoaders

        phase_status = None
        cached_extraction = None
        cached_chunks = None

        if resume and output_dir and output_dir.exists():
            phase_status = PhaseDetector.detect(output_dir)

            if phase_status.completed_phase > 0:
                logger.info("")
                logger.info("=" * 80)
                logger.info("RESUME MODE: Detected existing phases")
                logger.info("=" * 80)
                logger.info(f"Phase 1 (extraction): {'EXISTS' if 1 in phase_status.phase_files else 'MISSING'}")
                logger.info(f"Phase 2 (summaries):  {'EXISTS' if 2 in phase_status.phase_files else 'MISSING'}")
                logger.info(f"Phase 3 (chunks):     {'EXISTS' if 3 in phase_status.phase_files else 'MISSING'}")
                logger.info(f"Phase 4 (vectors):    {'EXISTS' if 4 in phase_status.phase_files else 'MISSING'}")
                logger.info("")
                logger.info(f"Will resume from Phase {phase_status.completed_phase + 1}")
                logger.info("=" * 80)
                logger.info("")

                # Load cached phases with specific error handling per phase
                try:
                    if phase_status.completed_phase >= 1:
                        logger.debug(f"Loading Phase 1 from {phase_status.phase_files[1]}")
                        cached_extraction = PhaseLoaders.load_phase1(phase_status.phase_files[1])

                    if phase_status.completed_phase >= 2:
                        logger.debug(f"Loading Phase 2 from {phase_status.phase_files[2]}")
                        cached_extraction = PhaseLoaders.load_phase2(
                            phase_status.phase_files[2],
                            cached_extraction
                        )

                    if phase_status.completed_phase >= 3:
                        logger.debug(f"Loading Phase 3 from {phase_status.phase_files[3]}")
                        cached_chunks = PhaseLoaders.load_phase3(phase_status.phase_files[3])

                except (FileNotFoundError, ValueError, KeyError, UnicodeDecodeError) as e:
                    logger.error(f"Failed to load cached phases: {e}")
                    logger.warning("Phase cache corrupted or incomplete - reprocessing from scratch")
                    logger.warning(f"Error details: {type(e).__name__}: {str(e)}")
                    cached_extraction = None
                    cached_chunks = None
                    phase_status = None

        # Semantic Duplicate Detection (before extraction to save time)
        if self.config.enable_duplicate_detection:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PRE-CHECK: Semantic Duplicate Detection")
            logger.info("=" * 80)

            try:
                from src.duplicate_detector import DuplicateDetector, DuplicateDetectionConfig

                # Create detector config
                dup_config = DuplicateDetectionConfig(
                    enabled=True,
                    similarity_threshold=self.config.duplicate_similarity_threshold,
                    sample_pages=self.config.duplicate_sample_pages,
                )

                # Initialize detector (lazy-loads embedder and vector store)
                # Uses DATABASE_URL from environment by default
                detector = DuplicateDetector(
                    config=dup_config,
                    connection_string=None,  # Uses DATABASE_URL env var
                )

                # Check for duplicate
                is_duplicate, similarity, match_doc_id = detector.check_duplicate(
                    file_path=str(document_path),
                    document_id=None,  # We don't have document_id yet (before extraction)
                )

                if is_duplicate:
                    logger.warning("")
                    logger.warning("[DUPLICATE]DUPLICATE DETECTED!")
                    logger.warning(f"   Document: {document_path.name}")
                    logger.warning(f"   Matches existing: {match_doc_id}")
                    logger.warning(f"   Similarity: {similarity:.1%}")
                    logger.warning(
                        f"   Threshold: {self.config.duplicate_similarity_threshold:.1%}"
                    )
                    logger.warning("")
                    logger.warning(
                        "[SKIPPED] indexing to prevent duplicates in vector_db/"
                    )
                    logger.info("=" * 80)
                    return None

                logger.info(f"No duplicate found (highest similarity: {similarity:.1%})")
                logger.info("=" * 80)
                logger.info("")

            except (ImportError, RuntimeError, PermissionError, OSError) as e:
                logger.error(f"Duplicate detection failed: {e}")
                logger.warning("Continuing with indexing...")
                import traceback

                logger.debug(traceback.format_exc())
            except KeyboardInterrupt:
                raise  # Always re-raise user interrupts
            except SystemExit:
                raise  # Always re-raise system exits

        # PHASE 1+2: Extract + Summaries (skip if cached)
        if cached_extraction is not None and phase_status and phase_status.completed_phase >= 2:
            logger.info("PHASE 1+2: [SKIPPED] - Loaded from cache")
            result = cached_extraction
            logger.info(
                f"Cached extraction: {result.num_sections} sections, "
                f"depth={result.hierarchy_depth}"
            )
        else:
            logger.info("PHASE 1+2: Extraction + Summaries")
            # UnifiedDocumentPipeline.process_document() returns ExtractedDocument
            # Automatically uses ToC extraction for PDFs when available
            result = self.extractor.process_document(Path(document_path))
            logger.info(
                f"Extracted: {result.num_sections} sections, "
                f"depth={result.hierarchy_depth}, "
                f"method={result.extraction_method}"
            )

        # Ensure document_id is never None (fallback to filename for non-legal documents)
        if not result.document_id:
            fallback_id = Path(document_path).stem
            logger.warning(
                f"document_id is None (non-legal document?), using filename: {fallback_id}"
            )
            result.document_id = fallback_id

        # Check existing components in vector_db AFTER extraction
        logger.info("")
        logger.info("=" * 80)
        logger.info("Checking existing components in vector_db/...")
        logger.info("=" * 80)

        existing = check_existing_components(result.document_id)

        if existing["exists"]:
            logger.info(f"Document '{existing['existing_doc_id']}' found in PostgreSQL:")
            logger.info(f"  - Vector embeddings: {'EXISTS' if existing['has_dense'] else 'MISSING'}")
            logger.info(f"  - Knowledge Graph: {'EXISTS' if existing['has_kg'] else 'MISSING'}")
            logger.info("")
            logger.info("Will skip existing components and add only missing ones...")
        else:
            logger.info(f"Document '{result.document_id}' NOT found in PostgreSQL")
            logger.info("Will create all components...")

        logger.info("=" * 80)
        logger.info("")

        # Determine what to skip
        skip_dense = existing["has_dense"]
        skip_kg = existing["has_kg"]

        # PHASE 3: Multi-layer chunking + SAC (skip if cached)
        if cached_chunks is not None and phase_status and phase_status.completed_phase >= 3:
            logger.info("PHASE 3: [SKIPPED] - Loaded from cache")
            chunks = cached_chunks
            chunking_stats = self.chunker.get_chunking_stats(chunks)
            logger.info(
                f"Cached chunks: L1={chunking_stats['layer1_count']}, "
                f"L2={chunking_stats['layer2_count']}, "
                f"L3={chunking_stats['layer3_count']} (PRIMARY)"
            )
        else:
            logger.info("PHASE 3: Multi-Layer Chunking + SAC")
            chunks = self.chunker.chunk_document(result)
            chunking_stats = self.chunker.get_chunking_stats(chunks)
            logger.info(
                f"Chunked: L1={chunking_stats['layer1_count']}, "
                f"L2={chunking_stats['layer2_count']}, "
                f"L3={chunking_stats['layer3_count']} (PRIMARY)"
            )

        # PHASE 4: Embedding & PostgreSQL pgvector (skip if exists)
        vector_store = None

        if skip_dense:
            logger.info("PHASE 4: [SKIPPED] - Vector embeddings already exist in PostgreSQL")
            # Load existing vector store adapter
            try:
                import asyncio
                connection_string = os.getenv("DATABASE_URL")
                if not connection_string:
                    raise ValueError("DATABASE_URL environment variable required")

                async def load_existing():
                    return await load_vector_store_adapter(
                        backend="postgresql",
                        connection_string=connection_string,
                        dimensions=self.embedder.dimensions,
                    )

                vector_store = asyncio.run(load_existing())
                store_stats = vector_store.get_stats()
                logger.info(
                    f"Loaded existing: {store_stats['total_vectors']} vectors "
                    f"({store_stats['documents']} documents)"
                )
            except (ValueError, RuntimeError, PermissionError) as e:
                logger.error(f"[ERROR] Failed to load existing PostgreSQL store: {e}")
                logger.info("Falling back to creating new embeddings...")
                skip_dense = False

        if not skip_dense:
            logger.info("PHASE 4: Embedding Generation")

            # Embed only layers configured in storage_layers (default: [1, 3])
            storage_layers = self.config.storage_layers
            logger.info(f"Embedding layers: {storage_layers}")

            embeddings = {}
            if 1 in storage_layers and chunks["layer1"]:
                embeddings["layer1"] = self.embedder.embed_chunks(chunks["layer1"], layer=1)
            else:
                embeddings["layer1"] = None

            if 2 in storage_layers and chunks["layer2"]:
                embeddings["layer2"] = self.embedder.embed_chunks(chunks["layer2"], layer=2)
            else:
                embeddings["layer2"] = None

            # Layer 3 if configured
            if 3 in storage_layers:
                embeddings["layer3"] = self.embedder.embed_chunks(chunks["layer3"], layer=3)
            else:
                embeddings["layer3"] = None

            # Log embedding stats
            embedded_counts = []
            for layer in [1, 2, 3]:
                if embeddings.get(f"layer{layer}") is not None:
                    embedded_counts.append(f"L{layer}:{embeddings[f'layer{layer}'].shape[0]}")
            logger.info(f"Embedded: {self.embedder.dimensions}D vectors ({', '.join(embedded_counts)})")

            # PHASE 4.5: Semantic Clustering (optional)
            if self.config.enable_semantic_clustering:
                logger.info("PHASE 4.5: Semantic Clustering")

                try:
                    from src.clustering import SemanticClusterer

                    clusterer = SemanticClusterer(self.config.clustering_config)

                    # Cluster each enabled layer
                    for layer in self.config.clustering_config.cluster_layers:
                        if layer == 1 and embeddings["layer1"] is not None:
                            chunk_list = chunks["layer1"]
                            embedding_array = embeddings["layer1"]
                        elif layer == 2 and embeddings["layer2"] is not None:
                            chunk_list = chunks["layer2"]
                            embedding_array = embeddings["layer2"]
                        elif layer == 3:
                            chunk_list = chunks["layer3"]
                            embedding_array = embeddings["layer3"]
                        else:
                            continue

                        # Perform clustering
                        chunk_ids = [c.chunk_id for c in chunk_list]
                        clustering_result = clusterer.cluster_embeddings(
                            embeddings=embedding_array,
                            chunk_ids=chunk_ids,
                        )

                        # Update chunk metadata with cluster assignments
                        for chunk in chunk_list:
                            cluster_info = clustering_result.get_chunk_cluster(chunk.chunk_id)
                            if cluster_info:
                                chunk.metadata.cluster_id = cluster_info.cluster_id
                                chunk.metadata.cluster_label = cluster_info.label
                                # Calculate confidence as distance to centroid
                                chunk_idx = chunk_ids.index(chunk.chunk_id)
                                chunk_embedding = embedding_array[chunk_idx]
                                centroid = cluster_info.centroid
                                if centroid is not None:
                                    # Cosine distance (0 = identical, 2 = opposite)
                                    distance = 1 - np.dot(chunk_embedding, centroid)
                                    chunk.metadata.cluster_confidence = float(distance)

                        logger.info(
                            f"Layer {layer}: Clustered into {clustering_result.n_clusters} clusters "
                            f"(noise: {clustering_result.noise_count})"
                        )

                        # Log quality metrics
                        if clustering_result.quality_metrics:
                            silhouette = clustering_result.quality_metrics.get("silhouette_score")
                            if silhouette:
                                logger.info(f"  Silhouette score: {silhouette:.3f}")

                except ImportError as e:
                    logger.warning(f"Clustering module not available: {e}")
                    logger.warning("Skipping semantic clustering")

            # PHASE 4: Vector Store Indexing (PostgreSQL pgvector)
            logger.info("PHASE 4: Vector Store Indexing (PostgreSQL pgvector)")

            import asyncio
            import nest_asyncio

            # Enable nested event loops (for running async in sync context)
            nest_asyncio.apply()

            # Get connection string from environment
            connection_string = os.getenv("DATABASE_URL")
            if not connection_string:
                raise ValueError(
                    "DATABASE_URL environment variable is required.\n"
                    "Example: export DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'"
                )

            # Create PostgreSQL adapter
            async def create_postgres_adapter():
                adapter = create_vector_store_adapter(
                    backend="postgresql",
                    connection_string=connection_string,
                    dimensions=self.embedder.dimensions
                )
                await adapter.initialize()
                return adapter

            vector_store = asyncio.run(create_postgres_adapter())
            vector_store.add_chunks(chunks, embeddings)  # Sync wrapper
            store_stats = vector_store.get_stats()
            logger.info(
                f"PostgreSQL indexed: {store_stats['total_vectors']} vectors "
                f"({store_stats['documents']} documents)"
            )

        # PHASE 5A: Knowledge Graph (optional, skip if exists)
        knowledge_graph = None
        kg_error = None

        if self.kg_extractor:
            if skip_kg:
                logger.info("PHASE 5A: [SKIPPED] - Knowledge Graph already exists")
                # Try to load existing KG for stats
                try:
                    kg_files = list(output_dir.glob("*_kg.json")) if output_dir else []
                    if kg_files:
                        with open(kg_files[0], 'r') as f:
                            kg_data = json.load(f)
                        logger.info(
                            f"Loaded existing KG: {len(kg_data.get('entities', []))} entities, "
                            f"{len(kg_data.get('relationships', []))} relationships"
                        )
                except Exception as e:
                    logger.warning(f"Could not load existing KG stats: {e}")
            else:
                logger.info("PHASE 5A: Knowledge Graph Extraction (Graphiti + GPT-4o-mini)")

                try:
                    # Graphiti uses phase3_chunks.json for episode-based ingestion
                    phase3_path = output_dir / "phase3_chunks.json" if output_dir else None

                    # Save phase3_chunks.json EARLY for entity-to-chunk mapping
                    if phase3_path and not phase3_path.exists() and chunks:
                        self._save_phase3(output_dir, result, chunks, chunking_stats)

                    if phase3_path and phase3_path.exists():
                        # Extract KG from phase3 chunks using async Graphiti
                        # Each chunk becomes a Graphiti episode for entity/relationship extraction
                        knowledge_graph = asyncio.run(
                            self.kg_extractor.extract_from_phase3(
                                phase3_path=phase3_path,
                                document_id=result.document_id,
                            )
                        )

                        logger.info(
                            f"Knowledge Graph: {knowledge_graph.total_entities} entities, "
                            f"{knowledge_graph.total_relationships} relationships "
                            f"({knowledge_graph.successful_chunks}/{knowledge_graph.total_chunks} chunks processed)"
                        )

                        # Save KG extraction result to output directory
                        if output_dir:
                            kg_output_path = output_dir / f"{result.document_id.replace('/', '_').replace(' ', '_')}_kg.json"
                            with open(kg_output_path, "w", encoding="utf-8") as f:
                                json.dump(knowledge_graph.to_dict(), f, indent=2, ensure_ascii=False)
                            logger.info(f"Knowledge Graph saved to: {kg_output_path}")
                    else:
                        logger.warning(f"phase3_chunks.json not found at {phase3_path}, skipping KG extraction")
                        knowledge_graph = None

                except (ValueError, RuntimeError, KeyError) as e:
                    # Expected KG construction errors (LLM API, data issues)
                    logger.error(f"[ERROR] Knowledge Graph extraction failed ({type(e).__name__}): {e}")
                    knowledge_graph = None
                    kg_error = str(e)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    # File system or parsing errors
                    logger.error(f"[ERROR] Knowledge Graph data error ({type(e).__name__}): {e}")
                    knowledge_graph = None
                    kg_error = str(e)
                except Exception as e:
                    # Unexpected errors - log with full traceback for debugging
                    logger.error(f"[ERROR] Knowledge Graph extraction failed unexpectedly: {e}", exc_info=True)
                    knowledge_graph = None
                    kg_error = str(e)

        # Save intermediate results
        if save_intermediate and output_dir:
            self._save_intermediate(
                output_dir=output_dir, result=result, chunks=chunks, chunking_stats=chunking_stats
            )

        # Display cost summary
        tracker = get_global_tracker()
        if tracker.get_total_cost() > 0:
            logger.info("\n" + tracker.get_summary())

        logger.info("=" * 80)
        logger.info(f"Indexing complete: {document_path.name}")
        logger.info("=" * 80)

        # vector_store is already a PostgresVectorStoreAdapter

        # Prepare result
        result_dict = {
            "vector_store": vector_store,
            "knowledge_graph": knowledge_graph,
            "stats": {
                "document_id": result.document_id,
                "source_path": _make_relative_path(document_path),
                "vector_store": store_stats,
                "chunking": chunking_stats,
                "storage_backend": "postgresql",
                "kg_enabled": self.config.enable_knowledge_graph,
                "kg_entities": knowledge_graph.total_entities if knowledge_graph else 0,
                "kg_relationships": knowledge_graph.total_relationships if knowledge_graph else 0,
                "kg_construction_failed": self.config.enable_knowledge_graph
                and knowledge_graph is None
                and kg_error is not None,
                "kg_error": kg_error if kg_error else None,
            },
        }

        if save_intermediate:
            result_dict["chunks"] = chunks

        return result_dict

    def index_batch(
        self, document_paths: list, output_dir: Path, save_per_document: bool = False
    ) -> Dict:
        """
        Index multiple documents into PostgreSQL.

        With PostgreSQL, all documents are indexed into the same database,
        so no merging is required. Each document is indexed independently.

        Supported formats: PDF, DOCX, PPTX, XLSX, HTML

        Args:
            document_paths: List of document file paths
            output_dir: Directory for intermediate results
            save_per_document: Save individual document intermediate results

        Returns:
            Dict with:
                - vector_store: PostgresVectorStoreAdapter
                - knowledge_graphs: List of KnowledgeGraphs (if enabled)
                - stats: Batch statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch indexing {len(document_paths)} documents to PostgreSQL...")

        # Collect knowledge graphs and track successful documents
        knowledge_graphs = []
        successful_docs = 0
        vector_store = None

        for i, document_path in enumerate(document_paths, 1):
            logger.info(f"\n[{i}/{len(document_paths)}] Processing: {Path(document_path).name}")

            try:
                # Index document (goes directly to PostgreSQL)
                result = self.index_document(
                    document_path=document_path,
                    save_intermediate=save_per_document,
                    output_dir=output_dir / Path(document_path).stem if save_per_document else None,
                )

                # Handle None return (duplicate document skipped)
                if result is None:
                    logger.info(f"[SKIPPED] Duplicate document: {document_path}")
                    continue

                # Track the vector store (all docs go to same PostgreSQL)
                vector_store = result["vector_store"]
                successful_docs += 1

                # Collect knowledge graph
                doc_kg = result.get("knowledge_graph")
                if doc_kg:
                    knowledge_graphs.append(doc_kg)

            except Exception as e:
                logger.error(f"[ERROR] Failed to index {document_path}: {e}")
                continue

        # Save combined knowledge graph (if any)
        if knowledge_graphs and self.kg_extractor:
            try:
                # Merge all KGs
                from src.graph import KnowledgeGraph

                combined_kg = KnowledgeGraph(
                    entities=[e for kg in knowledge_graphs for e in kg.entities],
                    relationships=[r for kg in knowledge_graphs for r in kg.relationships],
                )
                combined_kg.compute_stats()

                kg_output = output_dir / "combined_kg.json"
                combined_kg.save_json(str(kg_output))
                logger.info(f"Saved combined Knowledge Graph: {kg_output}")
                logger.info(
                    f"  Total: {len(combined_kg.entities)} entities, "
                    f"{len(combined_kg.relationships)} relationships"
                )
            except Exception as e:
                logger.error(f"[ERROR] Failed to save combined KG: {e}")

        # Get final stats from PostgreSQL
        store_stats = vector_store.get_stats() if vector_store else {}
        logger.info(f"\nBatch indexing complete: {store_stats}")

        return {
            "vector_store": vector_store,
            "knowledge_graphs": knowledge_graphs,
            "stats": {
                "total_documents": len(document_paths),
                "successful": successful_docs,
                "vector_store": store_stats,
                "storage_backend": "postgresql",
                "kg_enabled": self.config.enable_knowledge_graph,
                "total_entities": sum(len(kg.entities) for kg in knowledge_graphs),
                "total_relationships": sum(len(kg.relationships) for kg in knowledge_graphs),
            },
        }

    def _save_phase1(self, output_dir: Path, result) -> Path:
        """Save phase1 extraction results (for KG extraction)."""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        phase1_path = output_dir / "phase1_extraction.json"
        phase1_export = {
            "document_id": result.document_id,
            "source_path": _make_relative_path(Path(result.source_path)),
            "num_sections": result.num_sections,
            "hierarchy_depth": result.hierarchy_depth,
            "num_roots": result.num_roots,
            "num_tables": result.num_tables,
            "sections": [
                {
                    "section_id": s.section_id,
                    "title": s.title,
                    "content": s.content,
                    "level": s.level,
                    "depth": s.depth,
                    "path": s.path,
                    "page_number": s.page_number,
                    "content_length": len(s.content),
                }
                for s in result.sections
            ],
        }
        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(phase1_export, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved PHASE 1: {phase1_path}")
        return phase1_path

    def _save_phase3(self, output_dir: Path, result, chunks: Dict, chunking_stats: Dict) -> Path:
        """Save phase3 chunks early (for KG entity-to-chunk mapping)."""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        phase3_path = output_dir / "phase3_chunks.json"

        # Build L3 chunks list for KG mapping (simplified format with chunk_id, raw_content, context)
        l3_chunks = []
        for c in chunks.get("layer3", []):
            chunk_dict = c.to_dict()
            l3_chunks.append({
                "chunk_id": chunk_dict.get("chunk_id", ""),
                "raw_content": chunk_dict.get("raw_content", ""),
                "context": chunk_dict.get("context", ""),
            })

        phase3_export = {
            "document_id": result.document_id,
            "source_path": _make_relative_path(Path(result.source_path)),
            "chunking_stats": chunking_stats,
            "layer1": [c.to_dict() for c in chunks.get("layer1", [])],
            "layer2": [c.to_dict() for c in chunks.get("layer2", [])],
            "layer3": [c.to_dict() for c in chunks.get("layer3", [])],
            # Add simplified chunks list for KG mapping
            "chunks": l3_chunks,
        }
        with open(phase3_path, "w", encoding="utf-8") as f:
            json.dump(phase3_export, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved PHASE 3 (early, for KG mapping): {phase3_path}")
        return phase3_path

    def _save_intermediate(self, output_dir: Path, result, chunks: Dict, chunking_stats: Dict):
        """Save intermediate results from all phases (PHASE 1, 2, 3)."""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # PHASE 1: Save extraction results (structure & hierarchy)
        phase1_path = output_dir / "phase1_extraction.json"
        phase1_export = {
            "document_id": result.document_id,
            "source_path": _make_relative_path(Path(result.source_path)),
            "num_sections": result.num_sections,
            "hierarchy_depth": result.hierarchy_depth,
            "num_roots": result.num_roots,
            "num_tables": result.num_tables,
            "sections": [
                {
                    "section_id": s.section_id,
                    "title": s.title,
                    "content": s.content,
                    "level": s.level,
                    "depth": s.depth,
                    "path": s.path,
                    "page_number": s.page_number,
                    "content_length": len(s.content),
                }
                for s in result.sections
            ],
        }
        with open(phase1_path, "w", encoding="utf-8") as f:
            json.dump(phase1_export, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved PHASE 1: {phase1_path}")

        # PHASE 2: Save summaries
        phase2_path = output_dir / "phase2_summaries.json"
        phase2_export = {
            "document_id": result.document_id,
            "document_summary": result.document_summary,
            "section_summaries": [
                {"section_id": s.section_id, "title": s.title, "summary": s.summary}
                for s in result.sections
            ],
        }
        with open(phase2_path, "w", encoding="utf-8") as f:
            json.dump(phase2_export, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved PHASE 2: {phase2_path}")

        # PHASE 3: Save chunks
        phase3_path = output_dir / "phase3_chunks.json"
        phase3_export = {
            "document_id": result.document_id,
            "source_path": _make_relative_path(Path(result.source_path)),
            "chunking_stats": chunking_stats,
            "layer1": [c.to_dict() for c in chunks["layer1"]],
            "layer2": [c.to_dict() for c in chunks["layer2"]],
            "layer3": [c.to_dict() for c in chunks["layer3"]],
        }
        with open(phase3_path, "w", encoding="utf-8") as f:
            json.dump(phase3_export, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved PHASE 3: {phase3_path}")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline with research-optimal settings
    # Requires: DATABASE_URL environment variable for PostgreSQL
    # Requires: DEEPINFRA_API_KEY environment variable for embeddings
    config = IndexingConfig.from_env()

    pipeline = IndexingPipeline(config)

    # Index single document (stored in PostgreSQL)
    result = pipeline.index_document(
        document_path=Path("data/document.pdf"),
        save_intermediate=True,
        output_dir=Path("output/indexing"),
    )

    # Extract components
    vector_store = result["vector_store"]  # PostgresVectorStoreAdapter
    knowledge_graph = result["knowledge_graph"]

    # Save knowledge graph (vector store is already in PostgreSQL)
    if knowledge_graph:
        knowledge_graph.save_json("output/knowledge_graph.json")
        print(f"\nKnowledge Graph:")
        print(f"  Entities: {len(knowledge_graph.entities)}")
        print(f"  Relationships: {len(knowledge_graph.relationships)}")

    # Search example (using Layer 3 search)
    query_embedding = pipeline.embedder.embed_texts(["safety specification"])
    results = vector_store.search_layer3(query_embedding[0], k=6)

    print(f"\nTop results:")
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. {r['section_title']} (score: {r['score']:.4f})")
