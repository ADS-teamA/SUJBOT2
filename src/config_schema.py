"""
JSON Configuration Schema for MY_SUJBOT.

This module defines the complete configuration schema using Pydantic models.
ALL fields are REQUIRED unless explicitly marked Optional with a default value.
NO fallback values - if a required field is missing, validation will fail.

Migration from .env to config.json (2025-11-10):
- Removed fallback values and defaults
- Strict validation on startup
- Hierarchical JSON structure for better organization
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict, model_validator
from pathlib import Path
import json
import os
from dotenv import load_dotenv


class APIKeysConfig(BaseModel):
    """
    API Keys configuration - loaded from .env file.

    API keys are NEVER stored in config.json for security.
    Instead, they are loaded from environment variables.
    """

    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic Claude API key (loaded from ANTHROPIC_API_KEY env var)"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (loaded from OPENAI_API_KEY env var)"
    )
    voyage_api_key: Optional[str] = Field(
        default=None,
        description="Voyage AI API key (loaded from VOYAGE_API_KEY env var)"
    )
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key (loaded from GOOGLE_API_KEY env var)"
    )

    @model_validator(mode="after")
    def load_from_env(self):
        """Load API keys from environment variables (.env file)."""
        # Load .env file if it exists
        load_dotenv()

        # Load API keys from environment, overriding any values from JSON
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or self.anthropic_api_key
        self.openai_api_key = os.getenv("OPENAI_API_KEY") or self.openai_api_key
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY") or self.voyage_api_key
        self.google_api_key = os.getenv("GOOGLE_API_KEY") or self.google_api_key

        return self


class ModelsConfig(BaseModel):
    """Core model selection."""

    llm_model: str = Field(
        ...,
        description="LLM model for summaries and agent (e.g., claude-sonnet-4-5-20250929, gpt-4o-mini)"
    )
    llm_provider: Optional[str] = Field(
        None,
        description="LLM provider override (auto-detected from llm_model if not set): 'claude', 'openai', 'google'"
    )
    embedding_model: str = Field(
        ...,
        description="Embedding model for vector store (e.g., text-embedding-3-large, bge-m3)"
    )
    embedding_provider: Optional[str] = Field(
        None,
        description="Embedding provider override (auto-detected from embedding_model if not set): 'huggingface', 'openai', 'voyage'"
    )


class ExtractionConfig(BaseModel):
    """Document extraction configuration (PHASE 1)."""

    # Extraction backend selection
    backend: Literal["gemini", "unstructured", "auto"] = Field(
        ...,
        description="Extraction backend: 'gemini' (native PDF), 'unstructured' (with ToC), 'auto'"
    )
    gemini_model: str = Field(
        ...,
        description="Gemini model for extraction (e.g., 'gemini-2.5-flash')"
    )
    fallback_to_unstructured: bool = Field(
        ...,
        description="Fallback to Unstructured if Gemini fails"
    )

    # OCR settings
    enable_ocr: bool = Field(..., description="Enable OCR processing")
    ocr_engine: Literal["tesseract", "rapidocr"] = Field(
        ...,
        description="OCR engine: 'tesseract' (Czech support) or 'rapidocr' (3-5x faster)"
    )
    ocr_language: List[str] = Field(
        ...,
        description="OCR languages (e.g., ['ces', 'eng'])"
    )
    ocr_recognition: Literal["accurate", "fast"] = Field(
        ...,
        description="OCR recognition mode"
    )

    # Table extraction
    extract_tables: bool = Field(..., description="Extract tables from documents")
    table_mode: Literal["ACCURATE", "FAST"] = Field(..., description="Table extraction mode")

    # Hierarchy extraction (CRITICAL for hierarchical chunking)
    extract_hierarchy: bool = Field(
        ...,
        description="Extract document hierarchy (MUST be true for PHASE 3)"
    )
    enable_smart_hierarchy: bool = Field(
        ...,
        description="Font-size based hierarchy detection (CRITICAL - DO NOT DISABLE)"
    )
    hierarchy_tolerance: float = Field(
        ...,
        description="BBox height clustering tolerance in pixels (0.8 is research-optimal)",
        ge=0.0,
        le=2.0
    )

    # Watermark filtering
    filter_rotated_text: bool = Field(..., description="Remove diagonally oriented watermark text")
    rotation_min_angle: float = Field(
        ...,
        description="Minimum rotation angle in degrees",
        ge=0.0,
        le=90.0
    )
    rotation_max_angle: float = Field(
        ...,
        description="Maximum rotation angle in degrees",
        ge=0.0,
        le=90.0
    )

    # Summary generation (PHASE 2 integration)
    generate_summaries: bool = Field(..., description="Generate document/section summaries")
    summary_model: Optional[str] = Field(None, description="Override summary model (uses llm_model if None)")
    summary_max_chars: int = Field(
        ...,
        description="Summary length target for sections (research-optimal: 150 chars, valid range: 50-300)",
        ge=50,
        le=300
    )
    document_summary_max_chars: int = Field(
        ...,
        description="Document summary length target (100-1000 chars for describing document and sections)",
        ge=100,
        le=1000
    )
    summary_style: Literal["generic", "expert"] = Field(
        ...,
        description="Summary style (research-optimal: 'generic', proven better for legal docs)"
    )
    use_batch_api: bool = Field(..., description="Use OpenAI Batch API for summaries (50% cheaper)")
    batch_api_poll_interval: int = Field(
        ...,
        description="Seconds between batch status checks",
        ge=1
    )
    batch_api_timeout: int = Field(
        ...,
        description="Max wait for batch completion in seconds",
        ge=60
    )

    # Output formats
    generate_markdown: bool = Field(..., description="Generate markdown output")
    generate_json: bool = Field(..., description="Generate JSON output")


class UnstructuredConfig(BaseModel):
    """Unstructured.io extraction configuration."""

    strategy: Literal["hi_res", "fast", "ocr_only"] = Field(
        ...,
        description="Extraction strategy for PDF files"
    )
    model: Literal["yolox", "detectron2_mask_rcnn", "detectron2_onnx", "detectron2_quantized"] = Field(
        ...,
        description="Hi-res model for PDF extraction (yolox recommended from testing)"
    )
    languages: List[str] = Field(..., description="OCR languages (e.g., ['ces', 'eng'])")
    detect_language_per_element: bool = Field(..., description="Detect language per element")
    infer_table_structure: bool = Field(..., description="Extract table structure")
    extract_images: bool = Field(..., description="Extract images from PDF")
    include_page_breaks: bool = Field(..., description="Include page breaks in output")


class HierarchyDetectionConfig(BaseModel):
    """Generic hierarchy detection configuration."""

    enable_generic_hierarchy: bool = Field(..., description="Enable generic hierarchy detection")
    hierarchy_signals: List[str] = Field(
        ...,
        description="Signals for hierarchy detection: type, font_size, spacing, indentation, numbering, length, parent_id"
    )
    clustering_eps: float = Field(
        ...,
        description="DBSCAN clustering epsilon (lower = more granular levels)",
        ge=0.01,
        le=1.0
    )
    clustering_min_samples: int = Field(
        ...,
        description="Minimum samples per cluster",
        ge=1
    )


class SummarizationConfig(BaseModel):
    """Summarization configuration (PHASE 2)."""

    speed_mode: Literal["fast", "eco"] = Field(
        ...,
        description="Pipeline speed mode: 'fast' (immediate) or 'eco' (Batch API, 50% cheaper)"
    )
    temperature: float = Field(
        ...,
        description="LLM temperature for summaries (0.3 = consistent)",
        ge=0.0,
        le=2.0
    )
    max_tokens: int = Field(
        ...,
        description="Max output tokens for summary generation",
        ge=10
    )
    retry_on_exceed: bool = Field(..., description="Retry if summary exceeds max_chars")
    max_retries: int = Field(..., description="Max retry attempts", ge=0)
    max_workers: int = Field(
        ...,
        description="Parallel summary generation threads",
        ge=1
    )
    min_text_length: int = Field(
        ...,
        description="Minimum text length for summarization",
        ge=10
    )
    enable_batching: bool = Field(
        ...,
        description="Enable prompt batching (disabled by default due to JSON overhead)"
    )
    batch_size: int = Field(
        ...,
        description="Number of sections per batch if batching enabled",
        ge=1
    )


class ContextGenerationConfig(BaseModel):
    """Contextual Retrieval configuration (PHASE 3A)."""

    enable_contextual: bool = Field(
        ...,
        description="Enable contextual retrieval (-67% retrieval failures)"
    )
    temperature: float = Field(
        ...,
        description="Temperature for context generation",
        ge=0.0,
        le=2.0
    )
    max_tokens: int = Field(
        ...,
        description="Max tokens for generated context (50-100 words)",
        ge=10
    )
    include_surrounding: bool = Field(
        ...,
        description="Include neighboring chunks for context"
    )
    num_surrounding_chunks: int = Field(
        ...,
        description="Number of chunks before/after to include",
        ge=0
    )
    fallback_to_basic: bool = Field(
        ...,
        description="Fallback to basic chunking if context generation fails"
    )
    batch_size: int = Field(
        ...,
        description="Context generation batch size",
        ge=1
    )
    max_workers: int = Field(
        ...,
        description="Context generation parallel threads",
        ge=1
    )
    use_batch_api: bool = Field(..., description="Use OpenAI Batch API (50% cheaper)")
    batch_api_poll_interval: int = Field(
        ...,
        description="Batch API poll interval in seconds",
        ge=1
    )
    batch_api_timeout: int = Field(
        ...,
        description="Batch API timeout in seconds",
        ge=60
    )
    language: str = Field(
        default="ces",
        description="Language code for context generation ('ces' for Czech, 'eng' for English)"
    )


class ChunkingConfig(BaseModel):
    """Token-aware chunking configuration (PHASE 3)."""

    max_tokens: int = Field(
        ...,
        description="Max tokens per chunk (research-optimal: 512 for legal docs, valid range: 100-8192)",
        ge=100,
        le=8192
    )
    tokenizer_model: str = Field(
        ...,
        description="Tokenizer model (must match embedding_model)"
    )
    enable_sac: bool = Field(
        ...,
        description="Enable Summary-Augmented Chunking (-58% context drift)"
    )


class EmbeddingConfig(BaseModel):
    """Embedding configuration (PHASE 4)."""

    batch_size: int = Field(
        ...,
        description="Embedding batch size (larger = faster but more memory)",
        ge=1
    )
    cache_enabled: bool = Field(
        ...,
        description="Enable embedding cache (40-80% hit rate)"
    )
    cache_size: int = Field(
        ...,
        description="Max embedding cache entries",
        ge=1
    )
    normalize: bool = Field(
        ...,
        description="Normalize embeddings for cosine similarity (REQUIRED: must be true for FAISS indexing)"
    )


class ClusteringConfig(BaseModel):
    """Semantic clustering configuration (PHASE 4.5)."""

    algorithm: Literal["hdbscan", "agglomerative"] = Field(
        ...,
        description="Clustering algorithm"
    )
    n_clusters: Optional[int] = Field(
        None,
        description="Number of clusters for agglomerative (ignored for HDBSCAN)",
        ge=1
    )
    min_size: int = Field(
        ...,
        description="Minimum chunks per cluster for HDBSCAN",
        ge=1
    )
    max_clusters: int = Field(
        ...,
        description="Maximum clusters for auto-detection (agglomerative)",
        ge=1
    )
    min_clusters: int = Field(
        ...,
        description="Minimum clusters for auto-detection (agglomerative)",
        ge=1
    )
    layers: List[int] = Field(
        ...,
        description="Layers to cluster (e.g., [3] for chunks only)"
    )
    enable_labels: bool = Field(..., description="Generate semantic labels using LLM")
    enable_visualization: bool = Field(..., description="Generate UMAP visualization")
    visualization_dir: str = Field(..., description="Visualization output directory")


class HybridSearchConfig(BaseModel):
    """Hybrid search configuration (PHASE 5)."""

    enable: bool = Field(
        ...,
        description="Enable hybrid search (BM25 + Dense + RRF fusion, +23% precision)"
    )
    fusion_k: int = Field(
        ...,
        description="RRF fusion parameter (60 = optimal)",
        ge=1
    )


class KnowledgeGraphConfig(BaseModel):
    """
    Knowledge graph configuration (PHASE 5A) - config.json schema.

    NOTE: This is the CONFIG SCHEMA for validating config.json.
    For the internal pipeline configuration, see src/graph/config.py::KnowledgeGraphConfig.
    The two classes serve different purposes:
    - This class: Validates user-provided config.json (flat structure)
    - graph/config.py: Rich internal representation with nested sub-configs
    """

    enable: bool = Field(..., description="Enable knowledge graph extraction")
    llm_provider: str = Field(
        ...,
        description="LLM provider for entity/relationship extraction"
    )
    llm_model: str = Field(
        ...,
        description="LLM model for KG extraction (gpt-4o-mini recommended)"
    )
    backend: Literal["simple", "neo4j", "networkx"] = Field(
        ...,
        description="Graph storage backend"
    )
    export_path: str = Field(
        ...,
        description="JSON export path for simple backend"
    )
    verbose: bool = Field(..., description="Enable verbose logging")
    min_entity_confidence: float = Field(
        ...,
        description="Minimum entity confidence threshold",
        ge=0.0,
        le=1.0
    )
    min_relationship_confidence: float = Field(
        ...,
        description="Minimum relationship confidence threshold",
        ge=0.0,
        le=1.0
    )
    enable_entity_extraction: bool = Field(..., description="Enable entity extraction")
    enable_relationship_extraction: bool = Field(..., description="Enable relationship extraction")
    enable_cross_document_relationships: bool = Field(
        ...,
        description="Extract cross-document relationships (expensive)"
    )
    batch_size: int = Field(
        default=10,
        description="Batch size for Graphiti chunk processing (parallel)",
        ge=1,
        le=50
    )
    max_retries: int = Field(..., description="Max retry attempts", ge=0)
    retry_delay: float = Field(..., description="Delay between retries in seconds", ge=0.0)
    timeout: int = Field(..., description="Timeout per extraction batch in seconds", ge=1)
    log_path: str = Field(..., description="KG extraction log file path")


class EntityExtractionConfig(BaseModel):
    """Entity extraction configuration."""

    temperature: float = Field(
        ...,
        description="Temperature for extraction (0.0 for deterministic)",
        ge=0.0,
        le=2.0
    )
    min_confidence: float = Field(
        ...,
        description="Minimum confidence threshold",
        ge=0.0,
        le=1.0
    )
    extract_definitions: bool = Field(..., description="Extract entity definitions")
    normalize: bool = Field(..., description="Normalize entity values")
    batch_size: int = Field(..., description="Batch size for extraction", ge=1)
    max_workers: int = Field(..., description="Parallel extraction threads", ge=1)
    cache_results: bool = Field(..., description="Cache extraction results")
    include_examples: bool = Field(..., description="Include few-shot examples in prompt")
    max_per_chunk: int = Field(..., description="Max entities per chunk", ge=1)
    enabled_types: Optional[List[str]] = Field(
        None,
        description="Enabled entity types (None = all types)"
    )


class RelationshipExtractionConfig(BaseModel):
    """Relationship extraction configuration."""

    temperature: float = Field(
        ...,
        description="Temperature for extraction",
        ge=0.0,
        le=2.0
    )
    min_confidence: float = Field(
        ...,
        description="Minimum confidence threshold",
        ge=0.0,
        le=1.0
    )
    extract_evidence: bool = Field(..., description="Extract supporting evidence text")
    max_evidence_length: int = Field(..., description="Max characters for evidence snippets", ge=1)
    within_chunk: bool = Field(..., description="Extract relationships within single chunk")
    cross_chunk: bool = Field(..., description="Extract relationships across chunks")
    from_metadata: bool = Field(..., description="Extract relationships from chunk metadata")
    batch_size: int = Field(..., description="Batch size for extraction", ge=1)
    max_workers: int = Field(..., description="Parallel extraction threads", ge=1)
    cache_results: bool = Field(..., description="Cache extraction results")
    max_per_entity: int = Field(..., description="Max relationships per entity", ge=1)
    enabled_types: Optional[List[str]] = Field(
        None,
        description="Enabled relationship types (None = all types)"
    )


class Neo4jConfig(BaseModel):
    """Neo4j connection configuration."""

    uri: str = Field(..., description="Neo4j connection URI")
    username: str = Field(..., description="Neo4j username")
    password: str = Field(..., description="Neo4j password")
    database: str = Field(..., description="Neo4j database name")
    max_connection_lifetime: int = Field(
        ...,
        description="Max connection lifetime in seconds",
        ge=1
    )
    max_connection_pool_size: int = Field(
        ...,
        description="Max concurrent connections",
        ge=1
    )
    connection_timeout: int = Field(
        ...,
        description="Connection timeout in seconds",
        ge=1
    )
    create_indexes: bool = Field(..., description="Auto-create indexes for entity types")
    create_constraints: bool = Field(..., description="Auto-create uniqueness constraints")


class EntityDeduplicationConfig(BaseModel):
    """Entity deduplication configuration."""

    enable: bool = Field(
        ...,
        description="Master switch for entity deduplication (3-layer strategy)"
    )
    use_embeddings: bool = Field(
        ...,
        description="Layer 2: Use embedding-based semantic similarity (50-200ms overhead)"
    )
    similarity_threshold: float = Field(
        ...,
        description="Cosine similarity threshold for Layer 2",
        ge=0.0,
        le=1.0
    )
    use_acronym_expansion: bool = Field(
        ...,
        description="Layer 3: Enable acronym expansion (100-500ms overhead)"
    )
    acronym_fuzzy_threshold: float = Field(
        ...,
        description="Fuzzy match threshold for Layer 3",
        ge=0.0,
        le=1.0
    )
    custom_acronyms: Optional[str] = Field(
        None,
        description="Custom acronym mappings (format: 'ACRO1:expansion1,ACRO2:expansion2')"
    )
    apoc_enabled: bool = Field(
        ...,
        description="Use APOC for deduplication (~10-20ms), fallback to Cypher (~20-50ms)"
    )
    embedding_model: str = Field(
        ...,
        description="Embedding model for Layer 2 deduplication"
    )
    embedding_batch_size: int = Field(
        ...,
        description="Batch size for Layer 2 embeddings",
        ge=1
    )
    cache_embeddings: bool = Field(..., description="Cache embeddings for Layer 2")
    create_constraints: bool = Field(..., description="Create Neo4j uniqueness constraints")


class GraphStorageConfig(BaseModel):
    """Graph storage configuration."""

    backend: Literal["simple", "neo4j", "networkx"] = Field(
        ...,
        description="Graph storage backend"
    )
    simple_store_path: str = Field(
        ...,
        description="JSON store path for simple backend"
    )
    export_json: bool = Field(..., description="Export to JSON after construction")
    export_path: str = Field(..., description="JSON export path")
    deduplicate_entities: bool = Field(..., description="Deduplicate entities during graph construction")
    merge_similar_entities: bool = Field(
        ...,
        description="Merge similar entities (expensive)"
    )
    similarity_threshold: float = Field(
        ...,
        description="Similarity threshold for entity merging",
        ge=0.0,
        le=1.0
    )
    track_provenance: bool = Field(
        ...,
        description="Track provenance (chunk sources) for entities"
    )


class AgentConfig(BaseModel):
    """RAG agent configuration (PHASE 7)."""

    model: str = Field(
        ...,
        description="Agent model (can differ from llm_model)"
    )
    max_tokens: Optional[int] = Field(
        None,
        description="Max output tokens for agent responses (model-dependent)",
        ge=100
    )
    temperature: float = Field(
        ...,
        description="Agent temperature (0.3 = consistent)",
        ge=0.0,
        le=2.0
    )
    enable_tool_validation: bool = Field(..., description="Enable tool validation on startup")
    debug_mode: bool = Field(..., description="Enable debug mode")
    vector_store_path: str = Field(
        ...,
        description="Vector store path (must point to phase4_vector_store directory)"
    )
    knowledge_graph_path: Optional[str] = Field(
        None,
        description="Knowledge graph path (if using KG with agent)"
    )
    enable_prompt_caching: bool = Field(
        ...,
        description="Enable prompt caching (Anthropic only, 90% cost reduction)"
    )
    enable_context_management: bool = Field(
        ...,
        description="Enable context management (prevents quadratic cost growth)"
    )
    context_management_trigger: int = Field(
        ...,
        description="Token threshold for context pruning",
        ge=1000
    )
    context_management_keep: int = Field(
        ...,
        description="Keep last N messages with full tool context",
        ge=1
    )
    query_expansion_model: str = Field(
        ...,
        description="Query expansion LLM model (gpt-4o-mini recommended)"
    )


class AgentToolConfig(BaseModel):
    """Agent tool configuration."""

    default_k: int = Field(
        ...,
        description="Default number of results to retrieve",
        ge=1,
        le=100
    )
    enable_reranking: bool = Field(
        ...,
        description="Enable cross-encoder reranking (+25% accuracy)"
    )
    reranker_candidates: int = Field(
        ...,
        description="Number of candidates before reranking (must be >= default_k)",
        ge=1
    )
    reranker_model: Literal["bge-reranker-large", "bge-reranker-base", "ms-marco-mini"] = Field(
        ...,
        description="Reranker model"
    )
    enable_graph_boost: bool = Field(
        ...,
        description="Enable graph-based result boosting (+8% factual correctness, +200-500ms)"
    )
    graph_boost_weight: float = Field(
        ...,
        description="Weight for graph boosting",
        ge=0.0,
        le=1.0
    )
    max_document_compare: int = Field(
        ...,
        description="Max documents to compare",
        ge=1
    )
    compliance_threshold: float = Field(
        ...,
        description="Legal compliance confidence threshold",
        ge=0.0,
        le=1.0
    )
    context_window: int = Field(
        ...,
        description="Context window for expansion (chunks before/after)",
        ge=0
    )
    lazy_load_reranker: bool = Field(
        ...,
        description="Lazy load reranker (speeds up startup)"
    )
    lazy_load_graph: bool = Field(
        ...,
        description="Lazy load knowledge graph (speeds up startup)"
    )
    cache_embeddings: bool = Field(..., description="Cache embeddings")
    hyde_num_hypotheses: int = Field(
        ...,
        description="Number of hypothetical documents for HyDE (multi-hypothesis averaging)",
        ge=1,
        le=10
    )


class CLIConfig(BaseModel):
    """CLI configuration."""

    show_citations: bool = Field(..., description="Display citations in responses")
    citation_format: Literal["inline", "footnote", "detailed", "simple"] = Field(
        ...,
        description="Citation format"
    )
    show_tool_calls: bool = Field(..., description="Display tool calls in output")
    show_timing: bool = Field(..., description="Display timing information")
    enable_streaming: bool = Field(..., description="Enable streaming responses")
    save_history: bool = Field(..., description="Save conversation history")
    history_file: str = Field(..., description="History file path")
    max_history_items: int = Field(
        ...,
        description="Max history items to keep",
        ge=1
    )


class IndexingConfig(BaseModel):
    """
    Indexing pipeline configuration (optional section).

    Controls LlamaIndex wrapper, Redis caching, and entity labeling.
    Redis connection uses environment variables (REDIS_HOST, REDIS_PORT).
    """

    # LlamaIndex wrapper toggle
    use_llamaindex_wrapper: bool = Field(
        default=True,
        description="Use LlamaIndex wrapper for state persistence"
    )

    # Entity labeling settings
    enable_entity_labeling: bool = Field(
        default=True,
        description="Enable entity labeling phase (3.5) using Gemini"
    )
    entity_labeling_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for entity labeling"
    )
    entity_labeling_batch_size: int = Field(
        default=10,
        description="Batch size for entity labeling",
        ge=1,
        le=50
    )


class PipelineConfig(BaseModel):
    """General pipeline configuration."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        ...,
        description="Logging level"
    )
    log_file: str = Field(..., description="Log file path")
    data_dir: str = Field(..., description="Data directory for input documents")
    output_dir: str = Field(..., description="Output directory for pipeline results")


class RootConfig(BaseModel):
    """
    Root configuration - ALL sections REQUIRED.

    Strict validation enabled:
    - No automatic type coercion ("512" will not be converted to 512)
    - Unknown fields are rejected
    - All types must match exactly
    """

    model_config = ConfigDict(
        strict=True,  # Disable automatic type coercion
        extra="allow",  # Allow unknown fields (for backwards compatibility with multi_agent, storage, etc.)
    )

    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    models: ModelsConfig
    extraction: ExtractionConfig
    unstructured: UnstructuredConfig
    hierarchy_detection: HierarchyDetectionConfig
    summarization: SummarizationConfig
    context_generation: ContextGenerationConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    clustering: ClusteringConfig
    hybrid_search: HybridSearchConfig
    knowledge_graph: KnowledgeGraphConfig
    entity_extraction: EntityExtractionConfig
    relationship_extraction: RelationshipExtractionConfig
    neo4j: Neo4jConfig
    entity_deduplication: EntityDeduplicationConfig
    graph_storage: GraphStorageConfig
    agent: AgentConfig
    agent_tools: AgentToolConfig
    cli: CLIConfig
    pipeline: PipelineConfig
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)

    def _is_placeholder(self, value: Optional[str]) -> bool:
        """
        Check if API key is a placeholder value from config.json.example.

        Args:
            value: API key value to check

        Returns:
            True if value is empty, None, or contains placeholder text
        """
        if not value:
            return True

        # Common placeholder patterns
        PLACEHOLDER_PATTERNS = [
            "REQUIRED_IF",
            "YOUR_API_KEY",
            "YOUR_AURA_PASSWORD",
            "YOUR_INSTANCE_ID",
            "sk-placeholder",
            "example",
            "INSERT",
            "TODO",
            "CHANGEME",
        ]

        return any(pattern in value.upper() for pattern in PLACEHOLDER_PATTERNS)

    def validate_api_keys(self):
        """
        Validate that at least one API key is provided and matches model selection.
        API keys are loaded from .env file - this validation runs after loading.

        Raises:
            ValueError: If required API keys are missing
        """
        # Check LLM API keys
        if self.models.llm_model.startswith("claude-"):
            if self._is_placeholder(self.api_keys.anthropic_api_key):
                raise ValueError(
                    f"ANTHROPIC_API_KEY is REQUIRED for LLM model '{self.models.llm_model}'.\n"
                    f"Please set ANTHROPIC_API_KEY in your .env file.\n"
                    f"Example: ANTHROPIC_API_KEY=sk-ant-..."
                )
        elif self.models.llm_model.startswith(("gpt-", "o1-", "o3-", "gpt-5")):
            if self._is_placeholder(self.api_keys.openai_api_key):
                raise ValueError(
                    f"OPENAI_API_KEY is REQUIRED for LLM model '{self.models.llm_model}'.\n"
                    f"Please set OPENAI_API_KEY in your .env file.\n"
                    f"Example: OPENAI_API_KEY=sk-proj-..."
                )
        elif self.models.llm_model.startswith("gemini-"):
            if self._is_placeholder(self.api_keys.google_api_key):
                raise ValueError(
                    f"GOOGLE_API_KEY is REQUIRED for LLM model '{self.models.llm_model}'.\n"
                    f"Please set GOOGLE_API_KEY in your .env file.\n"
                    f"Example: GOOGLE_API_KEY=..."
                )

        # Check embedding API keys
        if self.models.embedding_provider == "openai":
            if self._is_placeholder(self.api_keys.openai_api_key):
                raise ValueError(
                    f"OPENAI_API_KEY is REQUIRED for EMBEDDING_PROVIDER 'openai'.\n"
                    f"Please set OPENAI_API_KEY in your .env file.\n"
                    f"Example: OPENAI_API_KEY=sk-proj-..."
                )
        elif self.models.embedding_provider == "voyage":
            if self._is_placeholder(self.api_keys.voyage_api_key):
                raise ValueError(
                    f"VOYAGE_API_KEY is REQUIRED for EMBEDDING_PROVIDER 'voyage'.\n"
                    f"Please set VOYAGE_API_KEY in your .env file.\n"
                    f"Example: VOYAGE_API_KEY=..."
                )

        # Validate at least one API key is set (not placeholder)
        valid_keys = [
            k for k in [
                self.api_keys.anthropic_api_key,
                self.api_keys.openai_api_key,
                self.api_keys.voyage_api_key,
                self.api_keys.google_api_key
            ] if not self._is_placeholder(k)
        ]

        if not valid_keys:
            raise ValueError(
                "At least ONE valid API key must be set in your .env file.\n"
                "Please set ANTHROPIC_API_KEY, OPENAI_API_KEY, VOYAGE_API_KEY, or GOOGLE_API_KEY.\n"
                "See .env.example for template."
            )

    @classmethod
    def from_json_file(cls, path: Path) -> "RootConfig":
        """
        Load configuration from JSON file with strict validation.

        Args:
            path: Path to config.json

        Returns:
            Validated RootConfig instance

        Raises:
            FileNotFoundError: If config.json does not exist
            ValueError: If validation fails or required fields are missing
            json.JSONDecodeError: If JSON is malformed
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}\n"
                f"Please create config.json from config.json.example:\n"
                f"  cp config.json.example config.json\n"
                f"  # Edit config.json with your settings"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in {path}:\n{e}\n"
                f"Please fix the JSON syntax error."
            ) from e
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Invalid UTF-8 encoding in {path}:\n{e}\n"
                f"Please save the file with UTF-8 encoding."
            ) from e

        try:
            config = cls(**data)
            # Run additional validation
            config.validate_api_keys()
            return config
        except ValidationError as e:
            # Pydantic validation errors - user config issues
            raise ValueError(
                f"Configuration validation failed:\n{e}\n\n"
                f"Please check config.json matches the required schema.\n"
                f"See config.json.example for reference."
            ) from e
        except ValueError as e:
            # API key validation errors or other value errors
            raise
        except (TypeError, AttributeError) as e:
            # Code bugs - don't hide these
            import logging
            logging.error(f"Internal validation error (this is a bug): {e}", exc_info=True)
            raise RuntimeError(
                f"Internal validation error. This is a bug in the validation code.\n"
                f"Please report this issue at https://github.com/ADS-teamA/SUJBOT2/issues\n"
                f"Include your config.json file (remove API keys) in the report."
            ) from e
        # DO NOT catch Exception - let unexpected errors propagate with full traceback


def load_config(config_path: Optional[Path] = None) -> RootConfig:
    """
    Load and validate configuration from config.json.

    This is the main entry point for loading configuration.

    Args:
        config_path: Optional path to config.json (default: ./config.json)

    Returns:
        Validated RootConfig instance

    Raises:
        FileNotFoundError: If config.json does not exist
        ValueError: If validation fails
    """
    if config_path is None:
        # Default: look for config.json in project root
        config_path = Path.cwd() / "config.json"

        # Fallback: if we're in src/, look in parent
        if not config_path.exists() and Path(__file__).parent.name == "src":
            config_path = Path(__file__).parent.parent / "config.json"

    return RootConfig.from_json_file(config_path)
