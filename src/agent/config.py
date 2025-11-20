"""
Agent Configuration

All agent settings are config-driven (no hardcoded values).
Supports environment variable overrides.
"""

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _load_agent_base_prompt() -> str:
    """
    Load base agent prompt from prompts/ directory.

    This loads the universal RAG search strategist instructions.
    Task-specific prompts (chat, benchmark) should be appended separately.
    """
    try:
        # Use centralized prompt loader from multi_agent module
        from pathlib import Path
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "base_agent_prompt.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8").strip()
        else:
            logger.warning(f"Prompt file not found: {prompt_file}")
            return "You are a RAG-powered assistant for legal and technical documents."
    except Exception as e:
        logger.error(f"Failed to load base agent prompt: {e}")
        # Fallback to minimal prompt
        return "You are a RAG-powered assistant for legal and technical documents."


def _detect_optimal_embedding_model() -> str:
    """
    Detect optimal embedding model based on platform.

    Cross-platform compatibility (per CLAUDE.md):
    - Apple Silicon (MPS): Use bge-m3 (local, FREE, GPU-accelerated)
    - Linux with NVIDIA GPU: Use bge-m3 (local, FREE, GPU-accelerated)
    - Windows or CPU-only: Use text-embedding-3-large (cloud, avoids PyTorch DLL issues)

    Can be overridden via EMBEDDING_MODEL environment variable.

    Returns:
        str: Optimal embedding model identifier
    """
    # Check environment variable override first
    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model:
        logger.info(f"Using EMBEDDING_MODEL from environment: {env_model}")
        return env_model

    system = platform.system()

    try:
        import torch

        # Check for GPU acceleration (Apple Silicon MPS or Linux CUDA)
        has_gpu = (
            system == "Darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ) or (system == "Linux" and torch.cuda.is_available())

        if has_gpu:
            gpu_type = "MPS" if system == "Darwin" else "CUDA"
            logger.info(
                f"Detected {system} with {gpu_type} - using bge-m3 (local, GPU-accelerated)"
            )
            return "bge-m3"

    except ImportError:
        # PyTorch not available, fallback to cloud
        logger.info("PyTorch not available - using cloud embeddings")

    # Default: Windows or CPU-only
    logger.info(f"Platform: {system} (CPU) - using text-embedding-3-large (cloud)")
    return "text-embedding-3-large"


@dataclass
class ToolConfig:
    """Configuration for RAG tools."""

    # Retrieval settings
    default_k: int = 6
    enable_reranking: bool = True
    reranker_candidates: int = 50
    reranker_model: str = "bge-reranker-large"  # SOTA accuracy (was: ms-marco-mini)

    # Graph settings
    enable_graph_boost: bool = True
    graph_boost_weight: float = 0.3

    # Analysis settings
    max_document_compare: int = 3
    compliance_threshold: float = 0.7

    # Context expansion settings (for get_chunk_context tool)
    context_window: int = 2  # Number of chunks before/after for context expansion

    # Query expansion settings (for unified search tool)
    query_expansion_provider: str = "openai"  # "openai" or "anthropic"
    query_expansion_model: str = "gpt-4o-mini"  # Stable, fast model for expansion

    # Performance
    lazy_load_reranker: bool = False  # Load reranker at startup for better tool availability
    lazy_load_graph: bool = True
    cache_embeddings: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.default_k <= 0:
            raise ValueError(f"default_k must be positive, got {self.default_k}")
        if self.reranker_candidates < self.default_k:
            raise ValueError(
                f"reranker_candidates ({self.reranker_candidates}) must be >= default_k ({self.default_k})"
            )
        if not 0.0 <= self.graph_boost_weight <= 1.0:
            raise ValueError(f"graph_boost_weight must be in [0, 1], got {self.graph_boost_weight}")
        if not 0.0 <= self.compliance_threshold <= 1.0:
            raise ValueError(
                f"compliance_threshold must be in [0, 1], got {self.compliance_threshold}"
            )
        if self.max_document_compare <= 0:
            raise ValueError(
                f"max_document_compare must be positive, got {self.max_document_compare}"
            )
        if self.context_window < 0:
            raise ValueError(f"context_window must be non-negative, got {self.context_window}")
        if self.query_expansion_provider.lower() not in ["openai", "anthropic"]:
            raise ValueError(
                f"query_expansion_provider must be 'openai' or 'anthropic', got '{self.query_expansion_provider}'"
            )


@dataclass
class CLIConfig:
    """CLI-specific configuration."""

    # Display settings
    show_citations: bool = True
    citation_format: str = "inline"  # "inline", "footnote", "detailed"
    show_tool_calls: bool = True
    show_timing: bool = True

    # Streaming
    enable_streaming: bool = True

    # History
    save_history: bool = True
    history_file: Path = field(default_factory=lambda: Path(".agent_history"))
    max_history_items: int = 1000

    def __post_init__(self):
        """Validate configuration values."""
        valid_formats = ["inline", "footnote", "detailed", "simple"]
        if self.citation_format not in valid_formats:
            raise ValueError(
                f"citation_format must be one of {valid_formats}, got '{self.citation_format}'"
            )
        if self.max_history_items <= 0:
            raise ValueError(f"max_history_items must be positive, got {self.max_history_items}")


@dataclass
class AgentConfig:
    """
    Main agent configuration.

    All settings are configurable via environment variables or constructor.
    """

    # === Core Settings ===
    # API Keys (provider-specific)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))

    # Model selection (user can override via CLI)
    model: str = field(
        default_factory=lambda: os.getenv("AGENT_MODEL", "claude-sonnet-4-5-20250929")
    )
    max_tokens: int = 4096
    temperature: float = 0.3

    # === Paths ===
    vector_store_path: Path = field(default_factory=lambda: Path("vector_db"))
    knowledge_graph_path: Optional[Path] = None

    # === Embedding Configuration ===
    # Platform-aware embedding model selection (can override via EMBEDDING_MODEL env var)
    embedding_model: str = field(default_factory=_detect_optimal_embedding_model)

    # === Feature Flags ===
    enable_tool_validation: bool = True
    enable_knowledge_graph: bool = False

    # === Prompt Caching (Anthropic) ===
    # Enable prompt caching to reduce costs by 90% and improve latency
    # Caches: system prompt, tools, and initialization messages
    enable_prompt_caching: bool = field(
        default_factory=lambda: os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
    )

    # === Context Management (Anthropic) ===
    # Automatically prune old tool results to prevent quadratic cost growth
    # Uses Anthropic's native Context Management API (server-side)
    enable_context_management: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CONTEXT_MANAGEMENT", "true").lower() == "true"
    )
    # Trigger threshold: Start pruning when input exceeds this token count
    context_management_trigger: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_MANAGEMENT_TRIGGER", "50000"))
    )
    # Keep last N tool uses (older ones are removed first)
    context_management_keep: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_MANAGEMENT_KEEP", "3"))
    )

    # === Debug Mode ===
    debug_mode: bool = False

    # === Sub-Configs ===
    tool_config: ToolConfig = field(default_factory=ToolConfig)
    cli_config: CLIConfig = field(default_factory=CLIConfig)

    # === System Prompt ===
    # Loaded from prompts/base_agent_prompt.txt (universal RAG instructions)
    # Task-specific prompts (chat/benchmark) should be appended by the caller
    system_prompt: str = field(default_factory=lambda: _load_agent_base_prompt())

    def validate(self) -> None:
        """
        Validate configuration.

        Checks:
        - API key presence and format
        - Numeric ranges (max_tokens, temperature)
        - Path existence
        - Model name validity
        - Sub-config validation (automatically via __post_init__)
        """
        # API key validation - require at least one API key
        if not self.anthropic_api_key and not self.openai_api_key and not self.google_api_key:
            raise ValueError(
                "No API keys set. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY environment variable.\n"
                "Example:\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "  export GOOGLE_API_KEY=AIza..."
            )

        # Validate format if keys are present
        if self.anthropic_api_key and not self.anthropic_api_key.startswith("sk-ant-"):
            raise ValueError("Anthropic API key has invalid format (should start with sk-ant-)")

        if self.openai_api_key and not (self.openai_api_key.startswith("sk-") or self.openai_api_key.startswith("sk-proj-")):
            raise ValueError("OpenAI API key has invalid format (should start with sk- or sk-proj-)")

        # Google API keys: Legacy keys start with "AIza", modern keys vary - check length instead
        if self.google_api_key and len(self.google_api_key) < 30:
            raise ValueError("Google API key appears too short (expected 39 chars)")

        # Numeric range validation
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.max_tokens > 200000:  # Claude's context window limit
            raise ValueError(f"max_tokens exceeds maximum (200000), got {self.max_tokens}")

        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be in [0.0, 1.0], got {self.temperature}")

        # Model name validation - accept Claude, GPT, and Gemini models
        model_lower = self.model.lower()
        is_valid_model = any(
            keyword in model_lower
            for keyword in ["claude", "haiku", "sonnet", "opus", "gpt-", "o1", "o3", "gemini"]
        )

        if not is_valid_model:
            raise ValueError(
                f"Invalid model name: {self.model}\n"
                f"Supported models: Claude (haiku/sonnet/opus), GPT-5 (gpt-5-mini/gpt-5-nano), "
                f"Gemini (gemini-2.5-flash/gemini-2.5-pro)"
            )

        # Path validation (skip for PostgreSQL backend)
        # Storage backend is determined at runtime from config.json, not here
        # The multi-agent runner will handle vector store initialization
        # This validation only applies if explicitly using FAISS backend
        import os
        storage_backend = os.getenv("STORAGE_BACKEND", "faiss")
        if storage_backend == "faiss" and not self.vector_store_path.exists():
            raise FileNotFoundError(
                f"Vector store not found: {self.vector_store_path}. "
                f"Run indexing pipeline first."
            )

        # Knowledge graph validation
        if self.enable_knowledge_graph and not self.knowledge_graph_path:
            raise ValueError(
                "Knowledge graph enabled but path not specified. "
                "Set knowledge_graph_path in config."
            )

        # Sub-configs are automatically validated via their __post_init__ methods

    @classmethod
    def from_env(cls, **overrides) -> "AgentConfig":
        """
        Create config from environment variables with optional overrides.

        Environment variables:
        - ANTHROPIC_API_KEY: Required
        - AGENT_MODEL: Model to use (default: claude-sonnet-4-5-20250929)
        - AGENT_MAX_TOKENS: Max output tokens (default: 4096, Gemini max: 8192)
        - VECTOR_STORE_PATH: Path to hybrid store
        - KNOWLEDGE_GRAPH_PATH: Path to KG JSON (optional)
        - QUERY_EXPANSION_MODEL: LLM model for query expansion (default: gpt-4o-mini)

        Args:
            **overrides: Override specific config values

        Returns:
            AgentConfig instance
        """
        # Create ToolConfig with environment variable support
        query_expansion_model_env = os.getenv("QUERY_EXPANSION_MODEL", "gpt-4o-mini")

        # Detect provider from model name
        query_expansion_provider = "openai"
        if "claude" in query_expansion_model_env.lower() or "haiku" in query_expansion_model_env.lower() or "sonnet" in query_expansion_model_env.lower():
            query_expansion_provider = "anthropic"

        tool_config = ToolConfig(
            query_expansion_provider=query_expansion_provider,
            query_expansion_model=query_expansion_model_env,
        )

        # Load enable_knowledge_graph from environment
        enable_kg_str = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "false").lower()
        enable_kg = enable_kg_str in ("true", "1", "yes")

        # Load max_tokens from environment (NEW - for Gemini compatibility)
        max_tokens = int(os.getenv("AGENT_MAX_TOKENS", "4096"))

        config = cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("AGENT_MODEL", "claude-sonnet-4-5-20250929"),
            max_tokens=max_tokens,
            vector_store_path=Path(os.getenv("VECTOR_STORE_PATH", "vector_db")),
            enable_knowledge_graph=enable_kg,
            tool_config=tool_config,
        )

        # Auto-detect KG path BEFORE applying overrides
        kg_path_from_env = None
        kg_path_str = os.getenv("KNOWLEDGE_GRAPH_PATH")
        if kg_path_str:
            kg_path = Path(kg_path_str)
            if kg_path.exists():
                kg_path_from_env = kg_path

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Set KG path after overrides
        if kg_path_from_env:
            config.knowledge_graph_path = kg_path_from_env
            config.enable_knowledge_graph = True
        elif config.enable_knowledge_graph and not config.knowledge_graph_path:
            # Default to vector_store_path if KG enabled but no path specified
            if config.vector_store_path:
                default_kg_path = config.vector_store_path
                if default_kg_path.exists():
                    config.knowledge_graph_path = default_kg_path
                    logger.info(f"Using vector store path for knowledge graphs: {default_kg_path}")

        return config

    @classmethod
    def from_config(cls, root_config, **overrides) -> "AgentConfig":
        """
        Create config from validated RootConfig (config.json).

        Args:
            root_config: Validated RootConfig from src.config
            **overrides: Override specific config values (e.g., from CLI args)

        Returns:
            AgentConfig instance
        """
        from src.config import ModelConfig

        # Get model config for API keys
        model_config = ModelConfig.from_config(root_config)

        # Detect provider for query expansion model
        query_expansion_model = root_config.agent.query_expansion_model
        query_expansion_provider = "openai"
        if "claude" in query_expansion_model.lower() or "haiku" in query_expansion_model.lower() or "sonnet" in query_expansion_model.lower():
            query_expansion_provider = "anthropic"
        elif "gemini" in query_expansion_model.lower():
            query_expansion_provider = "google"

        tool_config = ToolConfig(
            query_expansion_provider=query_expansion_provider,
            query_expansion_model=query_expansion_model,
        )

        # Create config from JSON
        config = cls(
            anthropic_api_key=model_config.anthropic_api_key or "",
            openai_api_key=model_config.openai_api_key or "",
            google_api_key=model_config.google_api_key or "",
            model=root_config.agent.model,
            max_tokens=root_config.agent.max_tokens or 4096,
            temperature=root_config.agent.temperature,
            vector_store_path=Path(root_config.agent.vector_store_path),
            knowledge_graph_path=Path(root_config.agent.knowledge_graph_path) if root_config.agent.knowledge_graph_path else None,
            enable_knowledge_graph=root_config.knowledge_graph.enable,
            enable_tool_validation=root_config.agent.enable_tool_validation,
            enable_prompt_caching=root_config.agent.enable_prompt_caching,
            enable_context_management=root_config.agent.enable_context_management,
            context_management_trigger=root_config.agent.context_management_trigger,
            context_management_keep=root_config.agent.context_management_keep,
            debug_mode=root_config.agent.debug_mode,
            tool_config=tool_config,
        )

        # Apply overrides (e.g., from CLI args)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config
