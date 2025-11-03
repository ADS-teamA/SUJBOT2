"""
Unified configuration system for RAG pipeline.

## Configuration Philosophy

**Single Source of Truth: .env file**
- API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, VOYAGE_API_KEY)
- Model selection (LLM_PROVIDER, LLM_MODEL, EMBEDDING_PROVIDER, EMBEDDING_MODEL)
- Optional overrides for advanced users

**config.py: Research-backed defaults**
- Chunk size: 500 chars (RCTS optimal from LegalBench-RAG)
- Summary length: 150 chars (generic summaries outperform expert)
- Temperature: 0.3 (low for consistency)
- Batch sizes, worker counts, etc.

## Pipeline Phases

- PHASE 1: Document Extraction (Docling)
- PHASE 2: Summarization (Generic summaries, 150 chars) → Model from .env
- PHASE 3: Chunking (Hierarchical with SAC, 500 chars)
- PHASE 4: Embedding (Multi-layer embeddings) → Model from .env

## Usage

All model selections are loaded from .env automatically:

```python
from src.config import SummarizationConfig, EmbeddingConfig

# Models loaded from .env automatically
summary_config = SummarizationConfig()  # Uses LLM_PROVIDER, LLM_MODEL from .env
embed_config = EmbeddingConfig()        # Uses EMBEDDING_PROVIDER, EMBEDDING_MODEL from .env

# Research parameters can be overridden if needed
custom_config = SummarizationConfig(max_chars=200, temperature=0.5)
```

All configuration classes can be imported by other modules.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def load_env():
    """Load environment variables from .env file with robust error handling."""
    # Primary path: project root (src/config.py -> parent.parent)
    env_path = Path(__file__).parent.parent / ".env"

    def _load_env_file(path: Path, is_fallback: bool = False):
        """Helper to load a single .env file with error handling."""
        try:
            if is_fallback:
                logger.warning(
                    f"Primary .env not found at {env_path}\n"
                    f"Using fallback: {path}\n"
                    f"This may cause unexpected configuration behavior."
                )
            else:
                logger.info(f"Loading configuration from {path}")

            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        try:
                            key, value = line.split("=", 1)
                            # Strip inline comments (text after #)
                            if "#" in value:
                                value = value.split("#")[0]
                            os.environ[key.strip()] = value.strip()
                        except ValueError as e:
                            logger.warning(f"Skipping malformed line {line_num} in {path}: {line}")
            return True

        except UnicodeDecodeError as e:
            logger.error(f"Failed to read .env file (encoding error): {e}")
            raise RuntimeError(
                f".env file contains invalid UTF-8 characters.\n"
                f"Please ensure .env is saved with UTF-8 encoding.\n"
                f"Path: {path}"
            )
        except PermissionError as e:
            logger.error(f"Permission denied reading .env file: {e}")
            raise RuntimeError(
                f"Cannot read .env file (permission denied).\n"
                f"Please check file permissions.\n"
                f"Path: {path}"
            )
        except Exception as e:
            logger.error(f"Failed to load .env file: {e}")
            raise RuntimeError(
                f"Failed to load configuration from .env file: {e}\n" f"Path: {path}"
            )

    # Try primary path first
    if env_path.exists():
        _load_env_file(env_path, is_fallback=False)
    else:
        # Fallback: current working directory
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            _load_env_file(cwd_env, is_fallback=True)
        else:
            logger.warning(
                f"No .env file found.\n"
                f"Expected: {env_path}\n"
                f"Fallback: {cwd_env}\n"
                f"Using default configuration values and environment variables."
            )


# Load .env on module import
load_env()


@dataclass
class ModelConfig:
    """Central model configuration loaded from environment variables."""

    # LLM Configuration
    llm_provider: str  # "claude" or "openai"
    llm_model: str  # e.g., "claude-sonnet-4.5", "gpt-4o-mini"

    # Embedding Configuration
    embedding_provider: str  # "voyage", "openai", "huggingface"
    embedding_model: str  # e.g., "kanon-2", "text-embedding-3-large", "bge-m3"

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            LLM_MODEL: Model name (default: "claude-sonnet-4-5-20250929")
            EMBEDDING_MODEL: Model name (default: "bge-m3")

            LLM_PROVIDER: Optional override (auto-detected from LLM_MODEL if not set)
            EMBEDDING_PROVIDER: Optional override (auto-detected from EMBEDDING_MODEL if not set)

            ANTHROPIC_API_KEY: Claude API key
            OPENAI_API_KEY: OpenAI API key
            VOYAGE_API_KEY: Voyage AI API key
        """
        # Load models first
        llm_model = os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929")
        embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3")

        # Auto-detect providers from model names (unless explicitly set)
        llm_provider = os.getenv("LLM_PROVIDER") or ModelRegistry.get_provider(llm_model, "llm")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER") or ModelRegistry.get_provider(embedding_model, "embedding")

        return cls(
            # LLM Configuration
            llm_provider=llm_provider,
            llm_model=llm_model,
            # Embedding Configuration
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            # API Keys
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        )

    def get_llm_config(self) -> dict:
        """Get LLM configuration for SummaryGenerator."""
        if self.llm_provider in ("claude", "anthropic"):
            return {
                "provider": "claude",
                "model": self.llm_model,
                "api_key": self.anthropic_api_key,
            }
        elif self.llm_provider == "openai":
            return {"provider": "openai", "model": self.llm_model, "api_key": self.openai_api_key}
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def get_embedding_config(self) -> dict:
        """Get embedding configuration for EmbeddingGenerator."""
        if self.embedding_provider == "voyage":
            return {
                "provider": "voyage",
                "model": self.embedding_model,
                "api_key": self.voyage_api_key,
            }
        elif self.embedding_provider == "openai":
            return {
                "provider": "openai",
                "model": self.embedding_model,
                "api_key": self.openai_api_key,
            }
        elif self.embedding_provider == "huggingface":
            return {
                "provider": "huggingface",
                "model": self.embedding_model,
                "api_key": None,  # Local models
            }
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")


# Import ModelRegistry for centralized model management
from src.utils.model_registry import ModelRegistry


def resolve_model_alias(model_name: str) -> str:
    """
    Resolve model alias to full model name using centralized ModelRegistry.

    This function is kept for backward compatibility. All model aliases are now
    managed in utils.model_registry.ModelRegistry.

    Args:
        model_name: Model name or alias (e.g., "haiku", "sonnet", "gpt-4o-mini")

    Returns:
        Full model name (e.g., "claude-haiku-4-5-20251001")
    """
    # Try LLM models first
    if model_name in ModelRegistry.LLM_MODELS:
        return ModelRegistry.resolve_llm(model_name)
    # Then try embedding models
    elif model_name in ModelRegistry.EMBEDDING_MODELS:
        return ModelRegistry.resolve_embedding(model_name)
    # Fallback: return as-is
    else:
        return model_name


@dataclass
class ExtractionConfig:
    """
    Configuration for Docling extraction (PHASE 1).

    To customize, create instance with your values:
        config = ExtractionConfig(
            hierarchy_tolerance=0.5,  # Stricter clustering
            enable_smart_hierarchy=True
        )
    """

    # OCR settings - Choice of OCR engine
    enable_ocr: bool = True
    # OCR engine: "tesseract" (best Czech support) or "rapidocr" (3-5× faster)
    # To use RapidOCR: pip install rapidocr_onnxruntime
    ocr_engine: str = "tesseract"  # "tesseract" or "rapidocr"
    # Tesseract language codes: ces=Czech, eng=English, deu=German, etc.
    # Use ["auto"] for automatic language detection
    ocr_language: List[str] = field(default_factory=lambda: ["ces", "eng"])
    ocr_recognition: str = "accurate"  # "accurate" or "fast" (deprecated for Tesseract)

    # Table extraction
    table_mode: str = "ACCURATE"  # Will be converted to TableFormerMode
    extract_tables: bool = True

    # Hierarchy extraction (CRITICAL for hierarchical chunking)
    extract_hierarchy: bool = True
    enable_smart_hierarchy: bool = True  # Font-size based classification
    hierarchy_tolerance: float = 0.8  # BBox height clustering tolerance (pixels, lower = stricter)

    # Summary generation (PHASE 2 integration)
    generate_summaries: bool = True  # Generate document/section summaries
    summary_model: Optional[str] = None  # Uses SummarizationConfig default when None
    summary_max_chars: int = 150  # Generic summary target length
    summary_style: str = "generic"  # Maintain parity with SummarizationConfig default
    use_batch_api: bool = True  # Mirror SummarizationConfig default behaviour
    batch_api_poll_interval: int = 5  # Seconds between batch status checks
    batch_api_timeout: int = 43200  # Max wait for batch completion (12h)

    # Output formats
    generate_markdown: bool = True
    generate_json: bool = True

    # Performance
    layout_model: str = "EGRET_XLARGE"  # Options: HERON, EGRET_LARGE, EGRET_XLARGE (recommended)

    @classmethod
    def from_env(cls) -> "ExtractionConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            OCR_LANGUAGE: Comma-separated language codes (default: "ces,eng")
            ENABLE_SMART_HIERARCHY: Enable font-size based hierarchy (default: "true")

        Returns:
            ExtractionConfig instance loaded from environment
        """
        ocr_lang_str = os.getenv("OCR_LANGUAGE", "ces,eng")
        ocr_languages = [lang.strip() for lang in ocr_lang_str.split(",")]

        return cls(
            ocr_language=ocr_languages,
            enable_smart_hierarchy=os.getenv("ENABLE_SMART_HIERARCHY", "true").lower() == "true",
        )


@dataclass
class SummarizationConfig:
    """
    Configuration for summarization (PHASE 2).

    Model selection is loaded from .env (LLM_PROVIDER, LLM_MODEL).
    Only research-backed parameters are configured here.
    """

    # Research-backed parameters (from LegalBench-RAG)
    max_chars: int = 150  # Summary length (research optimal)
    tolerance: int = 20  # Length tolerance
    style: str = "generic"  # Generic > Expert summaries
    temperature: float = 0.3  # Low temperature for consistency
    max_tokens: int = 100  # Max LLM output tokens (optimized: 150 chars ≈ 40-60 tokens)
    retry_on_exceed: bool = True  # Retry if exceeds max_chars
    max_retries: int = 3  # Max retry attempts
    # OPTIMIZED: Zvýšeno pro rychlejší zpracování (2× rychlejší)
    max_workers: int = 20  # Parallel summary generation
    min_text_length: int = 50  # Min text length for summarization

    # Prompt batching optimization (DISABLED - JSON overhead makes it slower)
    # For most LLMs, parallel mode with smaller max_tokens is faster than batching
    enable_prompt_batching: bool = False  # Batch multiple sections in one API call
    batch_size: int = 8  # Number of sections per API call (if enabled)

    # OpenAI Batch API optimization (NEW - 50% cost savings, async processing)
    use_batch_api: bool = True  # Use OpenAI Batch API for summaries (50% cheaper)
    batch_api_poll_interval: int = 5  # Seconds between status checks (faster response)
    batch_api_timeout: int = 43200  # Max wait time in seconds (12 hours default)

    # Model config loaded from .env (don't set here)
    provider: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self):
        """Load model config from environment if not provided."""
        if self.provider is None or self.model is None:
            model_config = ModelConfig.from_env()
            self.provider = model_config.llm_provider
            self.model = model_config.llm_model

    @classmethod
    def from_env(cls, **overrides) -> "SummarizationConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            LLM_PROVIDER: LLM provider (from ModelConfig)
            LLM_MODEL: Model name (from ModelConfig)
            SPEED_MODE: "fast" or "eco" (affects use_batch_api)

        Args:
            **overrides: Override specific fields

        Returns:
            SummarizationConfig instance loaded from environment
        """
        speed_mode = os.getenv("SPEED_MODE", "fast")
        config = cls(use_batch_api=(speed_mode == "eco"), **overrides)
        return config


@dataclass
class ContextGenerationConfig:
    """
    Configuration for Contextual Retrieval (Anthropic, Sept 2024).

    Generates LLM-based context for each chunk instead of generic summaries.
    Results in 67% reduction in retrieval failures (Anthropic research).

    Model selection is loaded from .env (LLM_PROVIDER, LLM_MODEL).
    """

    # Enable contextual retrieval
    enable_contextual: bool = True

    # Research-backed parameters
    temperature: float = 0.3  # Low temperature for consistency
    max_tokens: int = 150  # Context should be 50-100 words

    # Context window params
    include_surrounding_chunks: bool = True  # Include chunks above/below for better context
    num_surrounding_chunks: int = 1  # Number of chunks to include on each side

    # Fallback behavior
    fallback_to_basic: bool = True  # Use basic chunking if context generation fails

    # Batch processing (for performance)
    # OPTIMIZED: Zvýšeno pro rychlejší zpracování (2× rychlejší)
    batch_size: int = 20  # Generate contexts in batches
    max_workers: int = 10  # Parallel context generation

    # OpenAI Batch API optimization (NEW - 50% cost savings, async processing)
    use_batch_api: bool = True  # Use OpenAI Batch API for contexts (50% cheaper)
    batch_api_poll_interval: int = 5  # Seconds between status checks
    batch_api_timeout: int = 43200  # Max wait time in seconds (12 hours default)

    # Model config loaded from .env (don't set here)
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        """Load model config and API key from environment if not provided."""
        import logging

        logger = logging.getLogger(__name__)

        # Load model config from .env
        if self.provider is None or self.model is None:
            model_config = ModelConfig.from_env()
            self.provider = model_config.llm_provider
            self.model = model_config.llm_model

        # Load API key from .env based on provider
        if self.api_key is None:
            if self.provider in ("anthropic", "claude"):
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
                # Normalize provider to "anthropic" (contextual_retrieval.py expects "anthropic", not "claude")
                self.provider = "anthropic"
                if not self.api_key:
                    logger.warning(
                        "ANTHROPIC_API_KEY not set in environment. "
                        "Contextual retrieval will fail unless API key is provided during initialization."
                    )
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    logger.warning(
                        "OPENAI_API_KEY not set in environment. "
                        "Contextual retrieval will fail unless API key is provided during initialization."
                    )


@dataclass
class ChunkingConfig:
    """Configuration for chunking (PHASE 3)."""

    method: str = "RecursiveCharacterTextSplitter"
    chunk_size: int = 500
    chunk_overlap: int = 0

    # Chunking strategy
    enable_contextual: bool = True  # Contextual Retrieval (RECOMMENDED)
    enable_multi_layer: bool = True

    # Context generation config
    context_config: Optional["ContextGenerationConfig"] = None

    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", "; ", ", ", " ", ""])

    def __post_init__(self):
        """Initialize context_config if not provided."""
        if self.context_config is None and self.enable_contextual:
            # Default to standard config if not loading from environment
            self.context_config = ContextGenerationConfig()

    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            CHUNK_SIZE: Chunk size in characters (default: 500)
            ENABLE_SAC: Enable Summary-Augmented Chunking (default: "true")

        Returns:
            ChunkingConfig instance loaded from environment
        """
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            enable_contextual=os.getenv("ENABLE_SAC", "true").lower() == "true",
        )


@dataclass
class EmbeddingConfig:
    """
    Unified configuration for embedding generation (PHASE 4).

    This is the SINGLE source of truth for embedding configuration across
    the entire codebase (pipeline, agent, tools).

    Model selection is loaded from .env (EMBEDDING_PROVIDER, EMBEDDING_MODEL).
    Research-backed parameters (batch_size, normalize) are configured here.

    Supports:
    - Voyage AI: kanon-2, voyage-3-large, voyage-law-2
    - OpenAI: text-embedding-3-large, text-embedding-3-small
    - HuggingFace: BAAI/bge-m3 (local, multilingual)
    """

    # Model selection (loaded from .env)
    provider: Optional[str] = None  # "voyage", "openai", "huggingface"
    model: Optional[str] = None  # Model name

    # Research-backed parameters
    batch_size: int = 64  # Batch size for embedding generation (optimized)
    normalize: bool = True  # Normalize for cosine similarity (FAISS IndexFlatIP)

    # Multi-layer indexing
    enable_multi_layer: bool = True  # Enable multi-layer indexing (document, section, chunk)

    # Model metadata (auto-derived)
    dimensions: Optional[int] = None  # Auto-detected from model

    # Performance optimization
    cache_enabled: bool = True  # Enable embedding cache (40-80% hit rate)
    cache_max_size: int = 1000  # Max cache entries

    def __post_init__(self):
        """Load model config from environment if not provided and validate."""
        # Load provider and model from .env if not provided
        if self.provider is None or self.model is None:
            model_config = ModelConfig.from_env()
            self.provider = model_config.embedding_provider
            self.model = model_config.embedding_model

        # Validate parameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError(f"dimensions must be positive if specified, got {self.dimensions}")
        if self.cache_max_size <= 0:
            raise ValueError(f"cache_max_size must be positive, got {self.cache_max_size}")
        # Provider can be None (will be loaded from env in __post_init__) or one of the valid values
        if self.provider is not None and self.provider not in ["voyage", "openai", "huggingface"]:
            raise ValueError(
                f"provider must be 'voyage', 'openai', or 'huggingface', got {self.provider}"
            )

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            EMBEDDING_PROVIDER: "voyage", "openai", or "huggingface" (default: "huggingface")
            EMBEDDING_MODEL: Model name (default: "bge-m3")
            EMBEDDING_BATCH_SIZE: Batch size (default: 64)
            EMBEDDING_CACHE_SIZE: Cache max size (default: 1000)

        Returns:
            EmbeddingConfig instance loaded from environment
        """
        model_config = ModelConfig.from_env()

        return cls(
            provider=model_config.embedding_provider,
            model=model_config.embedding_model,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
            normalize=True,  # Always normalize for FAISS IndexFlatIP
            enable_multi_layer=True,  # Multi-layer indexing enabled by default
            cache_enabled=os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true",
            cache_max_size=int(os.getenv("EMBEDDING_CACHE_SIZE", "1000")),
        )


@dataclass
class PipelineConfig:
    """General pipeline configuration."""

    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/pipeline.log"


@dataclass
class RAGConfig:
    """
    Unified RAG pipeline configuration.

    Combines all configuration with sensible defaults from research.
    """

    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    models: ModelConfig = field(default_factory=ModelConfig.from_env)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """
        Load all sub-configs from environment variables.

        Returns:
            RAGConfig instance with all sub-configs loaded from environment
        """
        return cls(
            extraction=ExtractionConfig.from_env(),
            summarization=SummarizationConfig.from_env(),
            chunking=ChunkingConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            models=ModelConfig.from_env(),
        )

    def get_embedding_config(self) -> dict:
        """Get embedding configuration for EmbeddingGenerator."""
        if self.models.embedding_provider == "voyage":
            return {
                "provider": "voyage",
                "model": self.models.embedding_model,
                "api_key": self.models.voyage_api_key,
            }
        elif self.models.embedding_provider == "openai":
            return {
                "provider": "openai",
                "model": self.models.embedding_model,
                "api_key": self.models.openai_api_key,
            }
        elif self.models.embedding_provider == "huggingface":
            return {
                "provider": "huggingface",
                "model": self.models.embedding_model,
                "api_key": None,  # Local models
            }
        else:
            raise ValueError(f"Unknown embedding provider: {self.models.embedding_provider}")


def get_default_config() -> RAGConfig:
    """
    Get default RAG pipeline configuration.

    Uses sensible defaults from research + environment variables for API keys.

    Returns:
        RAGConfig instance with all default settings
    """
    return RAGConfig()


def get_model_config() -> ModelConfig:
    """
    Get model configuration from environment (legacy compatibility).

    Default models (optimized for M1 Mac):
    - LLM: Claude Sonnet 4.5 (balance of speed and quality)
    - Embeddings: BGE-M3-v2 (multilingual, runs locally on M1 with MPS acceleration)
    """
    return ModelConfig.from_env()


# Example usage
if __name__ == "__main__":
    # Load full pipeline config
    config = get_default_config()

    print("=== RAG Pipeline Configuration ===\n")

    print("PHASE 1: Extraction")
    print(f"  OCR: {config.extraction.enable_ocr}")
    print(f"  Smart Hierarchy: {config.extraction.enable_smart_hierarchy}")
    print(f"  Hierarchy Tolerance: {config.extraction.hierarchy_tolerance}")
    print(f"  Layout Model: {config.extraction.layout_model}")
    print()

    print("PHASE 2: Summarization")
    print(f"  Provider: {config.summarization.provider}")
    print(f"  Model: {config.summarization.model}")
    print(f"  Max Chars: {config.summarization.max_chars}")
    print()

    print("PHASE 3: Chunking")
    print(f"  Method: {config.chunking.method}")
    print(f"  Chunk Size: {config.chunking.chunk_size}")
    print(f"  Enable Contextual: {config.chunking.enable_contextual}")
    print()

    print("PHASE 4: Embedding")
    print(f"  Provider: {config.embedding.provider}")
    print(f"  Model: {config.embedding.model}")
    print()

    print("Models (from .env):")
    print(f"  LLM: {config.models.llm_provider}/{config.models.llm_model}")
    print(f"  Embedding: {config.models.embedding_provider}/{config.models.embedding_model}")


@dataclass
class ClusteringConfig:
    """
    Configuration for semantic clustering (PHASE 4.5).

    Clusters chunks based on embedding similarity to enable:
    - Topic-based retrieval
    - Diversity-aware search
    - Cluster-based analytics

    Supported algorithms:
    - HDBSCAN: Automatic cluster count, noise handling, density-based
    - Agglomerative: Hierarchical clustering with cosine distance

    All algorithms use cosine distance for consistency with normalized embeddings.
    """

    # Algorithm selection
    algorithm: str = "hdbscan"  # "hdbscan" or "agglomerative"

    # Cluster count (agglomerative only, None = auto-detect)
    n_clusters: Optional[int] = None

    # HDBSCAN parameters
    min_cluster_size: int = 5  # Minimum chunks per cluster

    # Agglomerative auto-detection parameters
    max_clusters: int = 50  # Maximum clusters for auto-detection
    min_clusters: int = 5  # Minimum clusters for auto-detection

    # Label generation
    enable_cluster_labels: bool = True  # Generate semantic labels using LLM

    # Which layers to cluster (default: [3] for chunk-level only)
    cluster_layers: List[int] = field(default_factory=lambda: [3])

    # Visualization
    enable_visualization: bool = False  # Generate UMAP visualization
    visualization_output_dir: str = "output/clusters"  # Output directory for plots

    @classmethod
    def from_env(cls) -> "ClusteringConfig":
        """
        Load clustering configuration from environment variables.

        Environment Variables:
            CLUSTERING_ALGORITHM: "hdbscan" or "agglomerative" (default: "hdbscan")
            CLUSTERING_N_CLUSTERS: Number of clusters for agglomerative (default: None = auto)
            CLUSTERING_MIN_SIZE: Minimum cluster size for HDBSCAN (default: 5)
            CLUSTERING_ENABLE_LABELS: Generate semantic labels (default: "true")
            CLUSTERING_ENABLE_VIZ: Generate UMAP visualization (default: "false")

        Returns:
            ClusteringConfig instance
        """
        return cls(
            algorithm=os.getenv("CLUSTERING_ALGORITHM", "hdbscan"),
            n_clusters=int(os.getenv("CLUSTERING_N_CLUSTERS")) if os.getenv("CLUSTERING_N_CLUSTERS") else None,
            min_cluster_size=int(os.getenv("CLUSTERING_MIN_SIZE", "5")),
            enable_cluster_labels=os.getenv("CLUSTERING_ENABLE_LABELS", "true").lower() == "true",
            enable_visualization=os.getenv("CLUSTERING_ENABLE_VIZ", "false").lower() == "true",
            visualization_output_dir=os.getenv("CLUSTERING_VIZ_DIR", "output/clusters"),
        )
