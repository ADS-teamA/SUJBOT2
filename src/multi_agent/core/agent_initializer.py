"""
Agent initialization utilities.

Provides centralized initialization for agent components (provider, prompts, tools)
to ensure SSOT (Single Source of Truth) and reduce code duplication.
"""

from __future__ import annotations

__all__ = ["initialize_agent", "AgentComponents"]

import logging
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from src.exceptions import AgentInitializationError, APIKeyError

if TYPE_CHECKING:
    from src.agent.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass
class AgentComponents:
    """Container for initialized agent components."""

    provider: Any  # BaseProvider - using Any to avoid circular import
    system_prompt: str
    tool_adapter: Any


def initialize_agent(
    config: Any,
    agent_name: str,
    prompt_name: Optional[str] = None
) -> AgentComponents:
    """
    Initialize common agent components (provider, prompts, tools).

    This is the SSOT for agent initialization. All agents should use this
    function instead of duplicating initialization code.

    Args:
        config: Agent configuration with model name
        agent_name: Name of the agent (for logging)
        prompt_name: Optional prompt name (defaults to agent_name)

    Returns:
        AgentComponents with provider, system_prompt, and tool_adapter

    Raises:
        AgentInitializationError: If provider initialization fails
        APIKeyError: If API key is missing for the provider
    """
    from src.agent.providers.factory import create_provider
    from ..prompts.loader import get_prompt_loader
    from ..tools.adapter import get_tool_adapter

    # Initialize provider (auto-detects from model name: claude/gpt/gemini)
    try:
        provider = create_provider(model=config.model)
        logger.info(f"Initialized provider for {agent_name} with model: {config.model}")
    except KeyError as e:
        # Missing API key
        raise APIKeyError(
            f"Missing API key for model {config.model}. "
            f"Ensure the appropriate API key is set in environment variables.",
            details={"model": config.model, "agent": agent_name},
            cause=e
        )
    except (ValueError, RuntimeError) as e:
        # Invalid model name or provider error
        raise AgentInitializationError(
            f"Failed to initialize LLM provider for model {config.model}. "
            f"Ensure model name is valid and API keys are configured.",
            details={"model": config.model, "agent": agent_name},
            cause=e
        )

    # Load system prompt
    prompt_loader = get_prompt_loader()
    effective_prompt_name = prompt_name or agent_name
    system_prompt = prompt_loader.get_prompt(effective_prompt_name)

    # Initialize tool adapter
    tool_adapter = get_tool_adapter()

    logger.info(f"{agent_name.capitalize()}Agent initialized with model: {config.model}")

    return AgentComponents(
        provider=provider,
        system_prompt=system_prompt,
        tool_adapter=tool_adapter
    )
