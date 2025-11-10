"""
Unified configuration system for RAG pipeline.

## Configuration Philosophy (UPDATED 2025-11-10)

**Single Source of Truth: config.json file**
- ALL configuration in one JSON file
- Strict validation on startup - NO fallbacks, NO defaults
- If any required parameter is missing, application exits with error
- Hierarchical structure for better organization

## Migration from .env to config.json

Previous system used .env with fallback values. New system:
1. Copy config.json.example to config.json
2. Fill in ALL required values
3. Application validates config on startup
4. Missing values = immediate error, application stops

## Pipeline Phases

- PHASE 1: Document Extraction (Docling)
- PHASE 2: Summarization (Generic summaries, 150 chars) → Model from config.json
- PHASE 3: Chunking (Hierarchical with SAC, 512 tokens)
- PHASE 4: Embedding (Multi-layer embeddings) → Model from config.json

## Usage

All configuration is loaded from config.json automatically:

```python
from src.config import get_config, SummarizationConfig, EmbeddingConfig

# Load validated config (fails if config.json is invalid or missing)
config = get_config()

# Access specific sections
summary_config = SummarizationConfig.from_config(config.summarization)
embed_config = EmbeddingConfig.from_config(config.embedding)
```

All configuration classes can be imported by other modules.
"""

import logging
from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass, field

# Import JSON config loader and schema
from src.config_schema import (
    RootConfig,
    load_config as load_json_config,
    APIKeysConfig as APIKeysSchema,
    ModelsConfig as ModelsSchema,
    ExtractionConfig as ExtractionSchema,
    SummarizationConfig as SummarizationSchema,
    ContextGenerationConfig as ContextGenerationSchema,
    ChunkingConfig as ChunkingSchema,
    EmbeddingConfig as EmbeddingSchema,
    ClusteringConfig as ClusteringSchema,
)

logger = logging.getLogger(__name__)

# Global config instance - loaded once at module import
_CONFIG: Optional[RootConfig] = None


def get_config(reload: bool = False) -> RootConfig:
    """
    Get validated configuration from config.json.

    This function loads and validates config.json on first call,
    then caches the result for subsequent calls.

    Args:
        reload: Force reload config.json (default: False)

    Returns:
        Validated RootConfig instance

    Raises:
        FileNotFoundError: If config.json does not exist
        ValueError: If validation fails or required fields are missing
    """
    global _CONFIG

    if _CONFIG is None or reload:
        try:
            _CONFIG = load_json_config()
            logger.info("Configuration loaded and validated successfully from config.json")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    return _CONFIG


@dataclass
class ModelConfig:
    """Central model configuration loaded from config.json."""

    # LLM Configuration
    llm_provider: str  # "claude" or "openai" or "google"
    llm_model: str  # e.g., "claude-sonnet-4.5", "gpt-4o-mini"

    # Embedding Configuration
    embedding_provider: str  # "voyage", "openai", "huggingface"
    embedding_model: str  # e.g., "kanon-2", "text-embedding-3-large", "bge-m3"

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    @classmethod
    def from_config(cls, config: RootConfig) -> "ModelConfig":
        """
        Load configuration from validated RootConfig.

        Args:
            config: Validated RootConfig instance

        Returns:
            ModelConfig instance
        """
        # Import ModelRegistry for provider detection
        from src.utils.model_registry import ModelRegistry

        # Auto-detect providers from model names if not explicitly set
        llm_provider = config.models.llm_provider
        if llm_provider is None:
            llm_provider = ModelRegistry.get_provider(config.models.llm_model, "llm")

        embedding_provider = config.models.embedding_provider
        if embedding_provider is None:
            embedding_provider = ModelRegistry.get_provider(config.models.embedding_model, "embedding")

        return cls(
            # LLM Configuration
            llm_provider=llm_provider,
            llm_model=config.models.llm_model,
            # Embedding Configuration
            embedding_provider=embedding_provider,
            embedding_model=config.models.embedding_model,
            # API Keys
            anthropic_api_key=config.api_keys.anthropic_api_key,
            openai_api_key=config.api_keys.openai_api_key,
            voyage_api_key=config.api_keys.voyage_api_key,
            google_api_key=config.api_keys.google_api_key,
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
        elif self.llm_provider == "google":
            return {"provider": "google", "model": self.llm_model, "api_key": self.google_api_key}
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
    """

    # OCR settings
    enable_ocr: bool
    ocr_engine: str  # "tesseract" or "rapidocr"
    ocr_language: List[str]
    ocr_recognition: str  # "accurate" or "fast"

    # Table extraction
    table_mode: str
    extract_tables: bool

    # Hierarchy extraction (CRITICAL for hierarchical chunking)
    extract_hierarchy: bool
    enable_smart_hierarchy: bool
    hierarchy_tolerance: float

    # Watermark / rotated text filtering
    filter_rotated_text: bool
    rotation_min_angle: float
    rotation_max_angle: float

    # Summary generation (PHASE 2 integration)
    generate_summaries: bool
    summary_model: Optional[str]
    summary_max_chars: int
    summary_style: str
    use_batch_api: bool
    batch_api_poll_interval: int
    batch_api_timeout: int

    # Output formats
    generate_markdown: bool
    generate_json: bool

    @classmethod
    def from_config(cls, extraction_config: ExtractionSchema) -> "ExtractionConfig":
        """
        Load configuration from validated ExtractionSchema.

        Args:
            extraction_config: Validated ExtractionSchema from RootConfig

        Returns:
            ExtractionConfig instance
        """
        return cls(
            enable_ocr=extraction_config.enable_ocr,
            ocr_engine=extraction_config.ocr_engine,
            ocr_language=extraction_config.ocr_language,
            ocr_recognition=extraction_config.ocr_recognition,
            table_mode=extraction_config.table_mode,
            extract_tables=extraction_config.extract_tables,
            extract_hierarchy=extraction_config.extract_hierarchy,
            enable_smart_hierarchy=extraction_config.enable_smart_hierarchy,
            hierarchy_tolerance=extraction_config.hierarchy_tolerance,
            filter_rotated_text=extraction_config.filter_rotated_text,
            rotation_min_angle=extraction_config.rotation_min_angle,
            rotation_max_angle=extraction_config.rotation_max_angle,
            generate_summaries=extraction_config.generate_summaries,
            summary_model=extraction_config.summary_model,
            summary_max_chars=extraction_config.summary_max_chars,
            summary_style=extraction_config.summary_style,
            use_batch_api=extraction_config.use_batch_api,
            batch_api_poll_interval=extraction_config.batch_api_poll_interval,
            batch_api_timeout=extraction_config.batch_api_timeout,
            generate_markdown=extraction_config.generate_markdown,
            generate_json=extraction_config.generate_json,
        )


@dataclass
class SummarizationConfig:
    """
    Configuration for summarization (PHASE 2).
    """

    # Research-backed parameters (from LegalBench-RAG) - required fields first
    max_chars: int
    style: str
    temperature: float
    max_tokens: int
    retry_on_exceed: bool
    max_retries: int
    max_workers: int
    min_text_length: int

    # Prompt batching optimization
    enable_prompt_batching: bool
    batch_size: int

    # OpenAI Batch API optimization
    use_batch_api: bool
    batch_api_poll_interval: int
    batch_api_timeout: int

    # Fields with defaults must come last
    tolerance: int = 20
    provider: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self):
        """Load model config from global config if not provided."""
        if self.provider is None or self.model is None:
            config = get_config()
            model_config = ModelConfig.from_config(config)
            self.provider = model_config.llm_provider
            self.model = model_config.llm_model

    @classmethod
    def from_config(cls, summarization_config: SummarizationSchema, **overrides) -> "SummarizationConfig":
        """
        Load configuration from validated SummarizationSchema.

        Args:
            summarization_config: Validated SummarizationSchema from RootConfig
            **overrides: Override specific fields

        Returns:
            SummarizationConfig instance
        """
        # Get global config for extraction config (for max_chars, style)
        root_config = get_config()
        extraction = root_config.extraction

        config_dict = {
            "max_chars": extraction.summary_max_chars,
            "style": extraction.summary_style,
            "temperature": summarization_config.temperature,
            "max_tokens": summarization_config.max_tokens,
            "retry_on_exceed": summarization_config.retry_on_exceed,
            "max_retries": summarization_config.max_retries,
            "max_workers": summarization_config.max_workers,
            "min_text_length": summarization_config.min_text_length,
            "enable_prompt_batching": summarization_config.enable_batching,
            "batch_size": summarization_config.batch_size,
            "use_batch_api": (summarization_config.speed_mode == "eco"),
            "batch_api_poll_interval": extraction.batch_api_poll_interval,
            "batch_api_timeout": extraction.batch_api_timeout,
        }

        # Apply overrides
        config_dict.update(overrides)

        return cls(**config_dict)


@dataclass
class ContextGenerationConfig:
    """
    Configuration for Contextual Retrieval (Anthropic, Sept 2024).
    """

    # Enable contextual retrieval
    enable_contextual: bool

    # Research-backed parameters
    temperature: float
    max_tokens: int

    # Context window params
    include_surrounding_chunks: bool
    num_surrounding_chunks: int

    # Fallback behavior
    fallback_to_basic: bool

    # Batch processing
    batch_size: int
    max_workers: int

    # OpenAI Batch API optimization
    use_batch_api: bool
    batch_api_poll_interval: int
    batch_api_timeout: int

    # Model config
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        """Load model config and API key from global config if not provided."""
        if self.provider is None or self.model is None:
            config = get_config()
            model_config = ModelConfig.from_config(config)
            self.provider = model_config.llm_provider
            self.model = model_config.llm_model

            # Load API key based on provider
            if self.api_key is None:
                if self.provider in ("anthropic", "claude"):
                    self.api_key = model_config.anthropic_api_key
                    # Normalize provider to "anthropic"
                    self.provider = "anthropic"
                elif self.provider == "openai":
                    self.api_key = model_config.openai_api_key
                elif self.provider == "google":
                    self.api_key = model_config.google_api_key

    @classmethod
    def from_config(cls, context_config: ContextGenerationSchema, **overrides) -> "ContextGenerationConfig":
        """
        Load configuration from validated ContextGenerationSchema.

        Args:
            context_config: Validated ContextGenerationSchema from RootConfig
            **overrides: Override specific fields

        Returns:
            ContextGenerationConfig instance
        """
        # Get global config for summarization speed_mode
        root_config = get_config()
        speed_mode = root_config.summarization.speed_mode

        config_dict = {
            "enable_contextual": context_config.enable_contextual,
            "temperature": context_config.temperature,
            "max_tokens": context_config.max_tokens,
            "include_surrounding_chunks": context_config.include_surrounding,
            "num_surrounding_chunks": context_config.num_surrounding_chunks,
            "fallback_to_basic": context_config.fallback_to_basic,
            "batch_size": context_config.batch_size,
            "max_workers": context_config.max_workers,
            "use_batch_api": (speed_mode == "eco"),
            "batch_api_poll_interval": context_config.batch_api_poll_interval,
            "batch_api_timeout": context_config.batch_api_timeout,
        }

        # Apply overrides
        config_dict.update(overrides)

        return cls(**config_dict)


@dataclass
class ChunkingConfig:
    """
    Configuration for token-aware chunking (PHASE 3).
    """

    # Token-aware chunking (IMMUTABLE - research-backed)
    max_tokens: int
    tokenizer_model: str

    # Chunking strategy
    enable_contextual: bool
    enable_multi_layer: bool = True

    # Context generation config
    context_config: Optional["ContextGenerationConfig"] = None

    def __post_init__(self):
        """Initialize context_config."""
        if self.context_config is None and self.enable_contextual:
            config = get_config()
            self.context_config = ContextGenerationConfig.from_config(config.context_generation)

    @classmethod
    def from_config(cls, chunking_config: ChunkingSchema) -> "ChunkingConfig":
        """
        Load configuration from validated ChunkingSchema.

        Args:
            chunking_config: Validated ChunkingSchema from RootConfig

        Returns:
            ChunkingConfig instance
        """
        # Get global config for context generation
        root_config = get_config()
        context_config = None
        if chunking_config.enable_sac:
            context_config = ContextGenerationConfig.from_config(root_config.context_generation)

        return cls(
            max_tokens=chunking_config.max_tokens,
            tokenizer_model=chunking_config.tokenizer_model,
            enable_contextual=chunking_config.enable_sac,
            context_config=context_config,
        )


@dataclass
class EmbeddingConfig:
    """
    Unified configuration for embedding generation (PHASE 4).
    """

    # Model selection
    provider: Optional[str] = None
    model: Optional[str] = None

    # Research-backed parameters
    batch_size: int = 64
    normalize: bool = True

    # Multi-layer indexing
    enable_multi_layer: bool = True

    # Model metadata
    dimensions: Optional[int] = None

    # Performance optimization
    cache_enabled: bool = True
    cache_max_size: int = 1000

    def __post_init__(self):
        """Load model config from global config if not provided and validate."""
        if self.provider is None or self.model is None:
            config = get_config()
            model_config = ModelConfig.from_config(config)
            self.provider = model_config.embedding_provider
            self.model = model_config.embedding_model

        # Validate parameters
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError(f"dimensions must be positive if specified, got {self.dimensions}")
        if self.cache_max_size <= 0:
            raise ValueError(f"cache_max_size must be positive, got {self.cache_max_size}")
        if self.provider is not None and self.provider not in ["voyage", "openai", "huggingface"]:
            raise ValueError(
                f"provider must be 'voyage', 'openai', or 'huggingface', got {self.provider}"
            )

    @classmethod
    def from_config(cls, embedding_config: EmbeddingSchema) -> "EmbeddingConfig":
        """
        Load configuration from validated EmbeddingSchema.

        Args:
            embedding_config: Validated EmbeddingSchema from RootConfig

        Returns:
            EmbeddingConfig instance
        """
        return cls(
            batch_size=embedding_config.batch_size,
            normalize=embedding_config.normalize,
            enable_multi_layer=True,  # Always enabled
            cache_enabled=embedding_config.cache_enabled,
            cache_max_size=embedding_config.cache_size,
        )


@dataclass
class PipelineConfig:
    """General pipeline configuration."""

    log_level: str
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/pipeline.log"

    @classmethod
    def from_config(cls, pipeline_schema) -> "PipelineConfig":
        """
        Load configuration from validated pipeline schema.

        Args:
            pipeline_schema: Validated pipeline config from RootConfig

        Returns:
            PipelineConfig instance
        """
        return cls(
            log_level=pipeline_schema.log_level,
            log_file=pipeline_schema.log_file,
        )


@dataclass
class RAGConfig:
    """
    Unified RAG pipeline configuration.
    """

    extraction: ExtractionConfig
    summarization: SummarizationConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    pipeline: PipelineConfig
    models: ModelConfig

    @classmethod
    def from_config(cls, config: RootConfig) -> "RAGConfig":
        """
        Load all sub-configs from validated RootConfig.

        Args:
            config: Validated RootConfig instance

        Returns:
            RAGConfig instance with all sub-configs
        """
        return cls(
            extraction=ExtractionConfig.from_config(config.extraction),
            summarization=SummarizationConfig.from_config(config.summarization),
            chunking=ChunkingConfig.from_config(config.chunking),
            embedding=EmbeddingConfig.from_config(config.embedding),
            pipeline=PipelineConfig.from_config(config.pipeline),
            models=ModelConfig.from_config(config),
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
    Get default RAG pipeline configuration from config.json.

    Returns:
        RAGConfig instance loaded from config.json

    Raises:
        FileNotFoundError: If config.json does not exist
        ValueError: If validation fails
    """
    config = get_config()
    return RAGConfig.from_config(config)


def get_model_config() -> ModelConfig:
    """
    Get model configuration from config.json.

    Returns:
        ModelConfig instance
    """
    config = get_config()
    return ModelConfig.from_config(config)


@dataclass
class ClusteringConfig:
    """
    Configuration for semantic clustering (PHASE 4.5).
    """

    # Algorithm selection
    algorithm: str
    n_clusters: Optional[int]

    # HDBSCAN parameters
    min_cluster_size: int

    # Agglomerative auto-detection parameters
    max_clusters: int
    min_clusters: int

    # Label generation
    enable_cluster_labels: bool

    # Which layers to cluster
    cluster_layers: List[int]

    # Visualization
    enable_visualization: bool
    visualization_output_dir: str

    @classmethod
    def from_config(cls, clustering_config: ClusteringSchema) -> "ClusteringConfig":
        """
        Load clustering configuration from validated ClusteringSchema.

        Args:
            clustering_config: Validated ClusteringSchema from RootConfig

        Returns:
            ClusteringConfig instance
        """
        return cls(
            algorithm=clustering_config.algorithm,
            n_clusters=clustering_config.n_clusters,
            min_cluster_size=clustering_config.min_size,
            max_clusters=clustering_config.max_clusters,
            min_clusters=clustering_config.min_clusters,
            enable_cluster_labels=clustering_config.enable_labels,
            cluster_layers=clustering_config.layers,
            enable_visualization=clustering_config.enable_visualization,
            visualization_output_dir=clustering_config.visualization_dir,
        )


def validate_config_on_startup():
    """
    Validate config.json at startup and exit with clear message if invalid.

    Call this at the top of every entrypoint script (run_pipeline.py, agent CLI, etc.)
    to ensure configuration is valid before any operations begin.

    Returns:
        RootConfig: Validated configuration if successful

    Exits:
        sys.exit(1) with clear error message if config is invalid

    Example:
        >>> from src.config import validate_config_on_startup
        >>> config = validate_config_on_startup()
        >>> # Now safe to proceed with pipeline operations
    """
    import sys

    try:
        return get_config()
    except FileNotFoundError:
        print("\n❌ ERROR: config.json not found!")
        print("\nPlease create config.json from config.json.example:")
        print("  cp config.json.example config.json")
        print("  # Edit config.json with your settings")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ ERROR: Invalid configuration in config.json!")
        print(f"\n{e}")
        print("\nPlease fix the errors in config.json")
        print("See config.json.example for reference")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load configuration!")
        print(f"\n{e}")
        print("\nUnexpected error - please report this issue:")
        print("https://github.com/ADS-teamA/SUJBOT2/issues")
        sys.exit(1)


# Example usage
if __name__ == "__main__":
    # Load full pipeline config from config.json
    try:
        config = get_default_config()

        print("=== RAG Pipeline Configuration ===\n")

        print("PHASE 1: Extraction")
        print(f"  OCR: {config.extraction.enable_ocr}")
        print(f"  Smart Hierarchy: {config.extraction.enable_smart_hierarchy}")
        print(f"  Hierarchy Tolerance: {config.extraction.hierarchy_tolerance}")
        print()

        print("PHASE 2: Summarization")
        print(f"  Provider: {config.summarization.provider}")
        print(f"  Model: {config.summarization.model}")
        print(f"  Max Chars: {config.summarization.max_chars}")
        print()

        print("PHASE 3: Chunking")
        print(f"  Max Tokens: {config.chunking.max_tokens}")
        print(f"  Enable Contextual: {config.chunking.enable_contextual}")
        print()

        print("PHASE 4: Embedding")
        print(f"  Provider: {config.embedding.provider}")
        print(f"  Model: {config.embedding.model}")
        print()

        print("Models (from config.json):")
        print(f"  LLM: {config.models.llm_provider}/{config.models.llm_model}")
        print(f"  Embedding: {config.models.embedding_provider}/{config.models.embedding_model}")

    except Exception as e:
        print(f"ERROR: Failed to load configuration")
        print(f"{e}")
        print("\nPlease create config.json from config.json.example:")
        print("  cp config.json.example config.json")
        print("  # Edit config.json with your settings")
