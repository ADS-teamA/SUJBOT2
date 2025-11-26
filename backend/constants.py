"""
Centralized constants for backend configuration.

This module provides a Single Source of Truth (SSOT) for constants
used across multiple modules.
"""

from typing import Literal

# Valid agent variants
AgentVariant = Literal["premium", "local"]

# Variant configuration - SINGLE SOURCE OF TRUTH
# Used by: settings.py, agent_adapter.py
VARIANT_CONFIG = {
    "premium": {
        "display_name": "Premium (Claude Haiku)",
        "model": "claude-haiku-4-5"
    },
    "local": {
        "display_name": "Local (Llama 3.1 70B)",
        "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"
    }
}

# Default variant when lookup fails or user has no preference
DEFAULT_VARIANT: AgentVariant = "premium"

# Supported DeepInfra models for validation
DEEPINFRA_SUPPORTED_MODELS = frozenset({
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
})


def get_variant_model(variant: str) -> str:
    """
    Get model identifier for a variant.

    Args:
        variant: Agent variant ('premium' or 'local')

    Returns:
        Model identifier string

    Raises:
        KeyError: If variant is not found
    """
    return VARIANT_CONFIG[variant]["model"]


def is_valid_variant(variant: str) -> bool:
    """Check if variant is valid."""
    return variant in VARIANT_CONFIG
