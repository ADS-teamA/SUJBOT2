"""
Centralized constants for backend configuration.

This module provides a Single Source of Truth (SSOT) for constants
used across multiple modules. Values are loaded from config.json's
agent_variants section with built-in fallbacks for backward compatibility.

SSOT Architecture:
- Primary source: config.json -> agent_variants section
- Fallback: Built-in defaults (for backward compatibility)
- Pattern: Lazy loading with _ensure_config_loaded()
"""

from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Valid agent variants type hint
AgentVariant = Literal["premium", "cheap", "local"]

# Module-level state (lazy loaded from config)
_config_loaded = False
_OPUS_TIER_AGENTS: frozenset[str] = frozenset()
_VARIANT_CONFIG: dict[str, dict[str, str]] = {}
_DEFAULT_VARIANT: str = "cheap"
_DEEPINFRA_SUPPORTED_MODELS: frozenset[str] = frozenset()


# =============================================================================
# Built-in Defaults (fallback when config unavailable)
# =============================================================================

_BUILTIN_OPUS_TIER_AGENTS = frozenset({
    "orchestrator",           # Critical routing + final synthesis
    "compliance",             # Complex legal verification
    "extractor",              # Core information retrieval
    "requirement_extractor",  # Highest complexity (15 iterations)
    "gap_synthesizer",        # Final actionable recommendations
})

_BUILTIN_VARIANT_CONFIG = {
    "premium": {
        "display_name": "Premium (Opus + Sonnet)",
        "opus_model": "claude-opus-4-5-20251101",
        "default_model": "claude-sonnet-4-5-20250929",
    },
    "cheap": {
        "display_name": "Cheap (Haiku 4.5)",
        "opus_model": "claude-haiku-4-5-20251001",
        "default_model": "claude-haiku-4-5-20251001",
    },
    "local": {
        "display_name": "Local (Llama 3.1 70B)",
        "opus_model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "default_model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    },
}

_BUILTIN_DEFAULT_VARIANT = "cheap"

_BUILTIN_DEEPINFRA_SUPPORTED_MODELS = frozenset({
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
})


# =============================================================================
# Config Loading Functions
# =============================================================================

def _load_builtin_defaults() -> None:
    """Load built-in fallback defaults (when config is unavailable)."""
    global _OPUS_TIER_AGENTS, _VARIANT_CONFIG, _DEFAULT_VARIANT, _DEEPINFRA_SUPPORTED_MODELS
    _OPUS_TIER_AGENTS = _BUILTIN_OPUS_TIER_AGENTS
    _VARIANT_CONFIG = _BUILTIN_VARIANT_CONFIG.copy()
    _DEFAULT_VARIANT = _BUILTIN_DEFAULT_VARIANT
    _DEEPINFRA_SUPPORTED_MODELS = _BUILTIN_DEEPINFRA_SUPPORTED_MODELS


def _ensure_config_loaded() -> None:
    """
    Lazy load agent variant configuration from config.json.

    This function is called automatically by accessor functions.
    Uses built-in defaults as fallback when config is unavailable.

    SSOT Source: config.json -> agent_variants section
    """
    global _config_loaded, _OPUS_TIER_AGENTS, _VARIANT_CONFIG, _DEFAULT_VARIANT
    global _DEEPINFRA_SUPPORTED_MODELS

    if _config_loaded:
        return

    try:
        from src.config import get_config
        config = get_config()

        if config.agent_variants is not None:
            # Load opus_tier_agents
            _OPUS_TIER_AGENTS = frozenset(config.agent_variants.opus_tier_agents)

            # Load variant configurations
            _VARIANT_CONFIG = {}
            for variant_name, variant_config in config.agent_variants.variants.items():
                _VARIANT_CONFIG[variant_name] = {
                    "display_name": variant_config.display_name,
                    "opus_model": variant_config.opus_model,
                    "default_model": variant_config.default_model,
                }

            # Load default variant
            _DEFAULT_VARIANT = config.agent_variants.default_variant

            logger.debug(
                "Agent variants loaded from config.json: %d variants, %d opus-tier agents",
                len(_VARIANT_CONFIG),
                len(_OPUS_TIER_AGENTS),
            )
        else:
            logger.debug("No agent_variants in config, using built-in defaults")
            _load_builtin_defaults()

        # DeepInfra models are not in config.json yet, use built-in
        _DEEPINFRA_SUPPORTED_MODELS = _BUILTIN_DEEPINFRA_SUPPORTED_MODELS

        _config_loaded = True

    except ImportError as e:
        # Config module not available - expected during testing or isolated execution
        logger.info(
            "Config module not available: %s. Using built-in defaults.",
            e,
        )
        _load_builtin_defaults()
        _config_loaded = True

    except (KeyError, AttributeError, TypeError) as e:
        # Config schema mismatch - log with traceback for debugging
        logger.warning(
            "Config schema error in agent_variants section: %s. Using built-in defaults.",
            e,
            exc_info=True,
        )
        _load_builtin_defaults()
        _config_loaded = True


def reload_constants() -> None:
    """
    Force reload of constants from config.json.

    Useful for testing or when config file is updated at runtime.
    """
    global _config_loaded
    _config_loaded = False
    _ensure_config_loaded()


# =============================================================================
# Public Getter Functions
# =============================================================================

def get_opus_tier_agents() -> frozenset[str]:
    """Get the set of opus-tier agents."""
    _ensure_config_loaded()
    return _OPUS_TIER_AGENTS


def get_variant_config() -> dict[str, dict[str, str]]:
    """Get the variant configuration dictionary."""
    _ensure_config_loaded()
    return _VARIANT_CONFIG


def get_default_variant() -> str:
    """Get the default variant name."""
    _ensure_config_loaded()
    return _DEFAULT_VARIANT


def get_deepinfra_supported_models() -> frozenset[str]:
    """Get set of supported DeepInfra models."""
    _ensure_config_loaded()
    return _DEEPINFRA_SUPPORTED_MODELS


# =============================================================================
# Public Functions (main API)
# =============================================================================

def get_agent_model(variant: str, agent_name: str) -> str:
    """
    Get model identifier for a specific agent within a variant.

    In Premium mode, OPUS_TIER_AGENTS use opus_model (Opus 4.5),
    while other agents use default_model (Sonnet 4.5).
    In Cheap/Local modes, all agents use the same model.

    Args:
        variant: Agent variant ('premium', 'cheap', or 'local')
        agent_name: Name of the agent

    Returns:
        Model identifier string

    Raises:
        KeyError: If variant is not found
    """
    _ensure_config_loaded()
    config = _VARIANT_CONFIG[variant]
    if agent_name in _OPUS_TIER_AGENTS:
        return config["opus_model"]
    return config["default_model"]


def get_variant_model(variant: str) -> str:
    """
    Get default model identifier for a variant (backward compatibility).

    Prefer get_agent_model() for new code - it handles tiered model selection.

    Args:
        variant: Agent variant ('premium', 'cheap', or 'local')

    Returns:
        Default model identifier string

    Raises:
        KeyError: If variant is not found
    """
    _ensure_config_loaded()
    return _VARIANT_CONFIG[variant]["default_model"]


def is_valid_variant(variant: str) -> bool:
    """Check if variant is valid."""
    _ensure_config_loaded()
    return variant in _VARIANT_CONFIG


def get_all_variants() -> list[str]:
    """Get list of all available variant names."""
    _ensure_config_loaded()
    return list(_VARIANT_CONFIG.keys())


def get_variant_display_name(variant: str) -> str:
    """Get human-readable display name for a variant."""
    _ensure_config_loaded()
    return _VARIANT_CONFIG.get(variant, {}).get("display_name", variant)


# =============================================================================
# Backward Compatibility - Direct module-level access
# =============================================================================
# These are kept for backward compatibility with existing code that imports:
#   from backend.constants import OPUS_TIER_AGENTS, VARIANT_CONFIG, DEFAULT_VARIANT
#
# New code should use the getter functions instead.

# Initialize on first import to support direct attribute access
_ensure_config_loaded()

# Re-export for backward compatibility
OPUS_TIER_AGENTS = _OPUS_TIER_AGENTS
VARIANT_CONFIG = _VARIANT_CONFIG
DEFAULT_VARIANT = _DEFAULT_VARIANT
DEEPINFRA_SUPPORTED_MODELS = _DEEPINFRA_SUPPORTED_MODELS
