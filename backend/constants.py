"""
Centralized constants for backend configuration.

This module provides a Single Source of Truth (SSOT) for constants
used across multiple modules.
"""

from typing import Literal

# Valid agent variants
AgentVariant = Literal["premium", "cheap", "local"]

# Agents that use the premium-tier model (Opus 4.5 in Premium mode)
# These are the most critical agents requiring highest reasoning capability
OPUS_TIER_AGENTS = frozenset({
    "orchestrator",       # Critical routing + final synthesis
    "compliance",         # Complex legal verification
    "extractor",          # Core information retrieval
    "requirement_extractor",  # Highest complexity (15 iterations)
    "gap_synthesizer",    # Final actionable recommendations
})

# Variant configuration - SINGLE SOURCE OF TRUTH
# Used by: settings.py, agent_adapter.py
# Structure: opus_model for OPUS_TIER_AGENTS, default_model for others
VARIANT_CONFIG = {
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

# Default variant when lookup fails or user has no preference
DEFAULT_VARIANT: AgentVariant = "cheap"

# Supported DeepInfra models for validation
DEEPINFRA_SUPPORTED_MODELS = frozenset({
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
})


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
    config = VARIANT_CONFIG[variant]
    if agent_name in OPUS_TIER_AGENTS:
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
    return VARIANT_CONFIG[variant]["default_model"]


def is_valid_variant(variant: str) -> bool:
    """Check if variant is valid."""
    return variant in VARIANT_CONFIG
