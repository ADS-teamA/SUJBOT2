"""
PHASE 1-6: Complete Indexing Pipeline

Orchestrates:
1. PHASE 1-3: Extraction, hierarchy, summaries, chunking
2. PHASE 4: Embedding generation + FAISS indexing
3. PHASE 5A: Knowledge Graph construction (optional)
4. PHASE 5B: Hybrid Search with BM25 + RRF (optional)
5. PHASE 5C: Cross-Encoder Reranking (optional)
6. PHASE 5D: Graph-Vector Integration (optional)
7. PHASE 6: Context Assembly for LLM (optional)

Supported formats: PDF, DOCX, PPTX, XLSX, HTML

Based on research:
- LegalBench-RAG: text-embedding-3-large + RCTS
- Multi-Layer Embeddings: 3 separate indexes
- Hybrid Search: BM25 + Dense + RRF fusion
- Cross-Encoder Reranking: Two-stage retrieval
- Graph Integration: Entity-aware boosting
- Context Assembly: SAC stripping + citations

Usage:
    pipeline = IndexingPipeline(
        embedding_model="text-embedding-3-large",
        enable_sac=True,
        enable_knowledge_graph=True,
        enable_hybrid_search=True,
        enable_reranking=True,
        enable_context_assembly=True
    )

    result = pipeline.index_document("document.pdf")
    result["vector_store"].save("output/vector_store")
    result["knowledge_graph"].save_json("output/knowledge_graph.json")
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
import numpy as np

from src.config import (
    ExtractionConfig,
    SummarizationConfig,
    ChunkingConfig,
    EmbeddingConfig,
)
from src.docling_extractor_v2 import DoclingExtractorV2
from src.multi_layer_chunker import MultiLayerChunker
from src.embedding_generator import EmbeddingGenerator
from src.faiss_vector_store import FAISSVectorStore
from src.cost_tracker import get_global_tracker, reset_global_tracker

# Knowledge Graph imports (optional)
try:
    from src.graph import (
        KnowledgeGraphPipeline,
        KnowledgeGraphConfig,
        EntityExtractionConfig as KGEntityConfig,
        RelationshipExtractionConfig as KGRelationshipConfig,
        GraphStorageConfig,
        GraphBackend,
    )

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


def check_existing_components(document_id: str, vector_db_path: str = "vector_db") -> Dict:
    """
    Check what components already exist in vector_db for this document.

    Args:
        document_id: Document ID to check
        vector_db_path: Path to vector database directory

    Returns:
        Dict with:
            - exists: bool - Whether document exists in any component
            - has_dense: bool - Has FAISS embeddings
            - has_bm25: bool - Has BM25 index
            - has_kg: bool - Has Knowledge Graph
            - existing_doc_id: str - Actual doc_id found (may differ slightly)
    """
    import pickle

    result = {
        "exists": False,
        "has_dense": False,
        "has_bm25": False,
        "has_kg": False,
        "existing_doc_id": None,
    }

    vector_db_path = Path(vector_db_path)

    # Check if vector_db exists
    if not vector_db_path.exists():
        return result

    # Check FAISS metadata
    metadata_file = vector_db_path / "metadata.pkl"
    if metadata_file.exists():
        try:
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            # Check all layers for document_id match
            for layer in ["metadata_layer1", "metadata_layer2", "metadata_layer3"]:
                if layer in metadata:
                    for chunk_meta in metadata[layer]:
                        chunk_doc_id = chunk_meta.get("document_id", "")
                        # Match exact or prefix (e.g., "BZ_VR1" matches "BZ_VR1_sample")
                        if chunk_doc_id == document_id or chunk_doc_id.startswith(document_id):
                            result["exists"] = True
                            result["has_dense"] = True
                            result["existing_doc_id"] = chunk_doc_id
                            break

                if result["exists"]:
                    break

        except Exception as e:
            logger.warning(f"Failed to read metadata.pkl: {e}")

    # Check BM25 indexes
    bm25_files = [
        vector_db_path / "bm25_layer1.pkl",
        vector_db_path / "bm25_layer2.pkl",
        vector_db_path / "bm25_layer3.pkl",
    ]

    if all(f.exists() for f in bm25_files):
        try:
            # Check if BM25 has documents
            with open(bm25_files[0], "rb") as f:
                bm25_data = pickle.load(f)
                if hasattr(bm25_data, "corpus") and len(bm25_data.corpus) > 0:
                    result["has_bm25"] = True
        except Exception as e:
            logger.warning(f"Failed to read BM25 index: {e}")

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

    # PHASE 5B: Hybrid Search (enabled by default for SOTA 2025)
    enable_hybrid_search: bool = True  # BM25 + dense with RRF fusion (+23% precision)
    hybrid_fusion_k: int = 60  # RRF k parameter (research: k=60 optimal)

    # PHASE 5C: Cross-Encoder Reranking (optional)
    enable_reranking: bool = False  # Two-stage retrieval: hybrid → rerank
    reranker_model: str = "bge-reranker-large"  # SOTA accuracy (aliases: "accurate", "sota")
    reranker_candidates: int = 50  # Retrieve 50 via hybrid search
    reranker_top_k: int = 6  # Rerank to top 6

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
    vector_store_path: str = "vector_db"  # Path to vector store for duplicate checking

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
                from src.config import ClusteringConfig

                self.clustering_config = ClusteringConfig.from_env()
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
            ENABLE_HYBRID_SEARCH: Enable hybrid search (default: "true")

        Args:
            **overrides: Override specific fields

        Returns:
            IndexingConfig instance loaded from environment
        """
        # Load speed mode from env
        speed_mode = os.getenv("SPEED_MODE", "fast")

        # Create config with sub-configs loaded from env
        config = cls(
            speed_mode=speed_mode,
            extraction_config=ExtractionConfig.from_env(),
            summarization_config=SummarizationConfig.from_env(),
            chunking_config=ChunkingConfig.from_env(),
            embedding_config=EmbeddingConfig.from_env(),
            enable_semantic_clustering=os.getenv("ENABLE_SEMANTIC_CLUSTERING", "false").lower() == "true",
            enable_knowledge_graph=os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true",
            enable_hybrid_search=os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true",
            enable_duplicate_detection=(
                os.getenv("ENABLE_DUPLICATE_DETECTION", "true").lower() == "true"
            ),
            duplicate_similarity_threshold=float(
                os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.98")
            ),
            duplicate_sample_pages=int(os.getenv("DUPLICATE_SAMPLE_PAGES", "1")),
            vector_store_path=os.getenv("VECTOR_STORE_PATH", "vector_db"),
            **overrides,
        )

        return config


class IndexingPipeline:
    """
    Complete indexing pipeline for RAG system.

    Phases:
    1. Smart hierarchy extraction (font-size based)
    2. Generic summary generation (gpt-4o-mini)
    3. Multi-layer chunking + SAC (RCTS 500 chars)
    4. Embedding + FAISS indexing (3 separate indexes)
    5A. Knowledge Graph construction (optional)
    5B. Hybrid search with BM25 + RRF (optional)
    5C. Cross-encoder reranking (optional)
    5D. Graph-vector integration (optional)
    6. Context assembly for LLM (optional)

    Based on:
    - LegalBench-RAG (Pipitone & Alami, 2024)
    - Summary-Augmented Chunking (Reuter et al., 2024)
    - Multi-Layer Embeddings (Lima, 2024)
    - Contextual Retrieval (Anthropic, 2024)
    - HybridRAG (2024): Graph + Vector integration
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
        self.extractor = DoclingExtractorV2(self.config.extraction_config)

        # Initialize PHASE 3: Chunking (uses nested config)
        self.chunker = MultiLayerChunker(config=self.config.chunking_config)

        # Initialize PHASE 4: Embedding (uses nested config)
        self.embedder = EmbeddingGenerator(self.config.embedding_config)

        # Initialize PHASE 5A: Knowledge Graph (optional)
        self.kg_pipeline = None
        if self.config.enable_knowledge_graph:
            if not KG_AVAILABLE:
                logger.warning(
                    "Knowledge Graph requested but not available. "
                    "Install with: pip install openai anthropic"
                )
            else:
                self._initialize_kg_pipeline()

        logger.info(
            f"Pipeline initialized: "
            f"SAC={self.config.chunking_config.enable_contextual}, "
            f"model={self.config.embedding_config.model} "
            f"({self.embedder.dimensions}D), "
            f"KG={self.config.enable_knowledge_graph}, "
            f"Hybrid={self.config.enable_hybrid_search}, "
            f"Rerank={self.config.enable_reranking}, "
            f"GraphRetrieval={self.config.enable_graph_retrieval}"
        )

    def _initialize_kg_pipeline(self):
        """Initialize Knowledge Graph pipeline using nested kg_config."""
        if self.config.kg_config is None:
            logger.warning("Knowledge Graph enabled but no kg_config provided")
            self.kg_pipeline = None
            return

        # Use the kg_config directly (already initialized in IndexingConfig.__post_init__)
        kg_config = self.config.kg_config

        # Validate API key
        if kg_config.entity_extraction.llm_provider == "openai" and not kg_config.openai_api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.kg_pipeline = None
            return
        elif (
            kg_config.entity_extraction.llm_provider == "anthropic"
            and not kg_config.anthropic_api_key
        ):
            logger.warning(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )
            self.kg_pipeline = None
            return

        # Initialize pipeline with config
        self.kg_pipeline = KnowledgeGraphPipeline(kg_config)
        logger.info(
            f"Knowledge Graph initialized: "
            f"model={kg_config.entity_extraction.llm_model}, "
            f"backend={kg_config.graph_storage.backend.value}"
        )

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
                - vector_store: FAISSVectorStore with indexed document
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
                detector = DuplicateDetector(
                    config=dup_config,
                    vector_store_path=self.config.vector_store_path,
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
            result = self.extractor.extract(document_path)
            logger.info(
                f"Extracted: {result.num_sections} sections, " f"depth={result.hierarchy_depth}"
            )

        # Check existing components in vector_db AFTER extraction
        logger.info("")
        logger.info("=" * 80)
        logger.info("Checking existing components in vector_db/...")
        logger.info("=" * 80)

        existing = check_existing_components(result.document_id)

        if existing["exists"]:
            logger.info(f"Document '{existing['existing_doc_id']}' found in vector_db:")
            logger.info(f"  - Dense embeddings (FAISS): {'EXISTS' if existing['has_dense'] else 'MISSING'}")
            logger.info(f"  - BM25 index: {'EXISTS' if existing['has_bm25'] else 'MISSING'}")
            logger.info(f"  - Knowledge Graph: {'EXISTS' if existing['has_kg'] else 'MISSING'}")
            logger.info("")
            logger.info("Will skip existing components and add only missing ones...")
        else:
            logger.info(f"Document '{result.document_id}' NOT found in vector_db")
            logger.info("Will create all components...")

        logger.info("=" * 80)
        logger.info("")

        # Determine what to skip
        skip_dense = existing["has_dense"]
        skip_bm25 = existing["has_bm25"]
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

        # PHASE 4: Embedding & FAISS (skip if exists)
        vector_store = None

        if skip_dense:
            logger.info("PHASE 4: [SKIPPED] - Dense embeddings already exist")
            # Load existing vector store
            try:
                vector_store = FAISSVectorStore.load("vector_db")
                store_stats = vector_store.get_stats()
                logger.info(
                    f"Loaded existing: {store_stats['total_vectors']} vectors "
                    f"({store_stats['documents']} documents)"
                )
            except (FileNotFoundError, ValueError, RuntimeError, PermissionError) as e:
                logger.error(f"[ERROR] Failed to load existing vector_db: {e}")
                logger.info("Falling back to creating new embeddings...")
                skip_dense = False

        if not skip_dense:
            logger.info("PHASE 4: Embedding Generation")

            # Embed only non-empty layers (optimization for flat documents)
            embeddings = {}
            if chunks["layer1"]:
                embeddings["layer1"] = self.embedder.embed_chunks(chunks["layer1"], layer=1)
            else:
                embeddings["layer1"] = None

            if chunks["layer2"]:
                embeddings["layer2"] = self.embedder.embed_chunks(chunks["layer2"], layer=2)
            else:
                embeddings["layer2"] = None

            # Layer 3 always exists
            embeddings["layer3"] = self.embedder.embed_chunks(chunks["layer3"], layer=3)

            logger.info(
                f"Embedded: {self.embedder.dimensions}D vectors, "
                f"{embeddings['layer3'].shape[0]} Layer 3 chunks"
            )

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

            # PHASE 4: FAISS Indexing
            logger.info("PHASE 4: FAISS Indexing")
            vector_store = FAISSVectorStore(dimensions=self.embedder.dimensions)
            vector_store.add_chunks(chunks, embeddings)
            store_stats = vector_store.get_stats()
            logger.info(
                f"Indexed: {store_stats['total_vectors']} vectors "
                f"({store_stats['documents']} documents)"
            )

        # PHASE 5B: Hybrid Search (optional, skip if exists)
        if self.config.enable_hybrid_search:
            if skip_bm25 and existing["has_dense"]:
                logger.info("PHASE 5B: [SKIPPED] - BM25 index already exists")
                # vector_store should already be HybridVectorStore from loading
            else:
                logger.info("PHASE 5B: Hybrid Search (BM25 + Dense + RRF)")

                try:
                    from src.hybrid_search import BM25Store, HybridVectorStore

                    # Build BM25 indexes for all 3 layers
                    bm25_store = BM25Store()
                    bm25_store.build_from_chunks(chunks)

                    logger.info(
                        f"BM25 Indexed: "
                        f"L1={len(bm25_store.index_layer1.corpus)}, "
                        f"L2={len(bm25_store.index_layer2.corpus)}, "
                        f"L3={len(bm25_store.index_layer3.corpus)}"
                    )

                    # Wrap FAISS + BM25 into HybridVectorStore
                    hybrid_store = HybridVectorStore(
                        faiss_store=vector_store,
                        bm25_store=bm25_store,
                        fusion_k=self.config.hybrid_fusion_k,
                    )

                    # Replace vector_store with hybrid_store for return
                    vector_store = hybrid_store
                    store_stats = vector_store.get_stats()

                    logger.info(f"Hybrid Search enabled: RRF k={self.config.hybrid_fusion_k}")

                except (ImportError, RuntimeError, ValueError, MemoryError) as e:
                    logger.error(f"[ERROR] Hybrid Search failed: {e}")
                    logger.warning("Continuing with dense-only retrieval...")
                    import traceback

                    logger.debug(traceback.format_exc())

        # PHASE 5A: Knowledge Graph (optional, skip if exists)
        knowledge_graph = None
        kg_error = None

        if self.kg_pipeline:
            if skip_kg:
                logger.info("PHASE 5A: [SKIPPED] - Knowledge Graph already exists")
                # Try to load existing KG for stats
                try:
                    output_dir = Path("output")
                    kg_files = list(output_dir.glob(f"{existing['existing_doc_id']}*/*/knowledge_graph.json"))
                    if kg_files:
                        import json
                        with open(kg_files[0], 'r') as f:
                            kg_data = json.load(f)
                        logger.info(
                            f"Loaded existing KG: {len(kg_data.get('entities', []))} entities, "
                            f"{len(kg_data.get('relationships', []))} relationships"
                        )
                except Exception as e:
                    logger.warning(f"Could not load existing KG stats: {e}")
            else:
                logger.info("PHASE 5A: Knowledge Graph Construction")

                try:
                    # Prepare chunks for KG (use Layer 3 primary chunks)
                    kg_chunks = [
                        {
                            "id": chunk.chunk_id,  # Fixed: Use chunk_id instead of id
                            "content": chunk.content,
                            "raw_content": chunk.raw_content,
                            "metadata": {
                                "document_id": chunk.metadata.document_id,
                                "section_path": chunk.metadata.section_path,
                                "section_title": chunk.metadata.section_title,
                            },
                        }
                        for chunk in chunks["layer3"]
                    ]

                    # Build knowledge graph
                    knowledge_graph = self.kg_pipeline.build_from_chunks(
                        chunks=kg_chunks, document_id=result.document_id
                    )

                    logger.info(
                        f"Knowledge Graph: {len(knowledge_graph.entities)} entities, "
                        f"{len(knowledge_graph.relationships)} relationships"
                    )

                except (ValueError, RuntimeError, KeyError) as e:
                    # Expected KG construction errors (LLM API, data issues)
                    logger.error(f"[ERROR] Knowledge Graph construction failed: {e}", exc_info=True)
                    if self.config.enable_knowledge_graph:
                        logger.error(
                            f"ERROR: Knowledge Graph was enabled in config but construction failed.\n"
                            f"Error: {e}\n"
                            f"To disable KG: Set ENABLE_KNOWLEDGE_GRAPH=false in .env"
                        )
                    knowledge_graph = None
                    kg_error = str(e)
                except (AttributeError, TypeError) as e:
                    # Code bugs - should fail fast with diagnostics
                    logger.error(f"[CRITICAL] Knowledge Graph code bug detected: {e}", exc_info=True)
                    logger.error(f"Enabled entity types: {self.kg_config.entity_extraction.enabled_entity_types}")
                    logger.error(f"Enabled relationship types: {self.kg_config.relationship_extraction.enabled_relationship_types}")
                    raise  # Re-raise to stop execution
                except MemoryError as e:
                    # Resource exhaustion - specific guidance
                    logger.error(f"[CRITICAL] Out of memory during KG construction: {e}", exc_info=True)
                    logger.error("Try reducing batch_size in config or disabling KG for large documents")
                    raise

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

        # Prepare result
        result_dict = {
            "vector_store": vector_store,
            "knowledge_graph": knowledge_graph,
            "stats": {
                "document_id": result.document_id,
                "source_path": _make_relative_path(document_path),
                "vector_store": store_stats,
                "chunking": chunking_stats,
                "hybrid_enabled": self.config.enable_hybrid_search,
                "kg_enabled": self.config.enable_knowledge_graph,
                "kg_entities": len(knowledge_graph.entities) if knowledge_graph else 0,
                "kg_relationships": len(knowledge_graph.relationships) if knowledge_graph else 0,
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
        Index multiple documents into a single vector store.

        Supported formats: PDF, DOCX, PPTX, XLSX, HTML

        Args:
            document_paths: List of document file paths
            output_dir: Directory to save vector store
            save_per_document: Save individual document vector stores

        Returns:
            Dict with:
                - vector_store: Combined FAISSVectorStore
                - knowledge_graphs: List of KnowledgeGraphs (if enabled)
                - stats: Batch statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch indexing {len(document_paths)} documents...")

        # Create combined vector store (HybridVectorStore or FAISSVectorStore)
        if self.config.enable_hybrid_search:
            from src.hybrid_search import HybridVectorStore, BM25Store

            # Create empty FAISS and BM25 stores
            faiss_store = FAISSVectorStore(dimensions=self.embedder.dimensions)
            bm25_store = BM25Store()
            vector_store = HybridVectorStore(faiss_store, bm25_store)
            logger.info("Created HybridVectorStore for batch indexing")
        else:
            vector_store = FAISSVectorStore(dimensions=self.embedder.dimensions)
            logger.info("Created FAISSVectorStore for batch indexing")

        # Collect knowledge graphs
        knowledge_graphs = []

        for i, document_path in enumerate(document_paths, 1):
            logger.info(f"\n[{i}/{len(document_paths)}] Processing: {Path(document_path).name}")

            try:
                # Index document
                result = self.index_document(
                    document_path=document_path,
                    save_intermediate=False,
                    output_dir=output_dir if save_per_document else None,
                )

                # Handle None return (duplicate document skipped)
                if result is None:
                    logger.info(f"[SKIPPED] Duplicate document: {document_path}")
                    continue

                # Extract components
                doc_store = result["vector_store"]
                doc_kg = result.get("knowledge_graph")

                # Merge into combined store using proper merge() methods
                from src.hybrid_search import HybridVectorStore

                if isinstance(vector_store, HybridVectorStore):
                    # HybridVectorStore: Merge both FAISS and BM25 stores
                    if isinstance(doc_store, HybridVectorStore):
                        # Both hybrid: merge faiss_store and bm25_store
                        vector_store.faiss_store.merge(doc_store.faiss_store)
                        vector_store.bm25_store.merge(doc_store.bm25_store)
                        logger.debug("Merged HybridVectorStore into combined store")
                    else:
                        # doc_store is FAISSVectorStore (shouldn't happen, but handle gracefully)
                        vector_store.faiss_store.merge(doc_store)
                        logger.warning(
                            "Document store is FAISSVectorStore but combined is Hybrid - "
                            "BM25 index will be incomplete"
                        )
                else:
                    # FAISSVectorStore: Simple merge
                    if isinstance(doc_store, FAISSVectorStore):
                        vector_store.merge(doc_store)
                        logger.debug("Merged FAISSVectorStore into combined store")
                    else:
                        # doc_store is HybridVectorStore (shouldn't happen, but handle gracefully)
                        vector_store.merge(doc_store.faiss_store)
                        logger.warning(
                            "Document store is HybridVectorStore but combined is FAISS - "
                            "Hybrid features will be lost"
                        )

                # Collect knowledge graph
                if doc_kg:
                    knowledge_graphs.append(doc_kg)

                # Save per-document store
                if save_per_document:
                    doc_name = Path(document_path).stem
                    doc_output = output_dir / f"{doc_name}_store"
                    doc_store.save(doc_output)
                    logger.info(f"Saved individual store: {doc_output}")

            except Exception as e:
                logger.error(f"[ERROR] Failed to index {document_path}: {e}")
                continue

        # Save combined store
        combined_output = output_dir / "combined_store"
        vector_store.save(combined_output)

        # Save combined knowledge graph (if any)
        if knowledge_graphs and self.kg_pipeline:
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

        logger.info(f"\nBatch indexing complete: {vector_store.get_stats()}")
        logger.info(f"Saved to: {combined_output}")

        return {
            "vector_store": vector_store,
            "knowledge_graphs": knowledge_graphs,
            "stats": {
                "total_documents": len(document_paths),
                "successful": len(knowledge_graphs) if self.kg_pipeline else len(document_paths),
                "vector_store": vector_store.get_stats(),
                "kg_enabled": self.config.enable_knowledge_graph,
                "total_entities": sum(len(kg.entities) for kg in knowledge_graphs),
                "total_relationships": sum(len(kg.relationships) for kg in knowledge_graphs),
            },
        }

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
    # Knowledge Graph is now enabled by default (PHASE 5A)
    config = IndexingConfig(
        # PHASE 1-2
        enable_smart_hierarchy=True,
        generate_summaries=True,
        summary_model="gpt-4o-mini",
        # PHASE 3
        chunk_size=500,
        enable_sac=True,
        # PHASE 4
        embedding_model="text-embedding-3-large",
        normalize_embeddings=True,
        # PHASE 5A: Knowledge Graph (enabled by default)
        # enable_knowledge_graph=True,  # ✅ No need to set - default is True
        kg_llm_provider="openai",
        kg_llm_model="gpt-4o-mini",
        kg_backend="simple",  # simple, neo4j, or networkx
    )

    pipeline = IndexingPipeline(config)

    # Index single document
    result = pipeline.index_document(
        document_path=Path("data/document.pdf"),
        save_intermediate=True,
        output_dir=Path("output/indexing"),
    )

    # Extract components
    vector_store = result["vector_store"]
    knowledge_graph = result["knowledge_graph"]

    # Save vector store
    vector_store.save(Path("output/vector_store"))

    # Save knowledge graph
    if knowledge_graph:
        knowledge_graph.save_json("output/knowledge_graph.json")
        print(f"\nKnowledge Graph:")
        print(f"  Entities: {len(knowledge_graph.entities)}")
        print(f"  Relationships: {len(knowledge_graph.relationships)}")

    # Search example
    query_embedding = pipeline.embedder.embed_texts(["safety specification"])
    results = vector_store.hierarchical_search(query_embedding, k_layer3=6)

    print(f"\nTop results:")
    for i, result in enumerate(results["layer3"][:3], 1):
        print(f"{i}. {result['section_title']} (score: {result['score']:.4f})")
