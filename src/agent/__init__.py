"""
Agent shared utilities and configuration.

This module provides shared configuration, tools, and providers
used by both the legacy single-agent system (deprecated) and the
new multi-agent system.

Modules:
- config: Shared configuration (AgentConfig)
- tools: 17 RAG tools organized in 3 tiers
- providers: LLM provider abstractions (Anthropic, OpenAI, Google)
- graph_loader/graph_adapter: Knowledge graph integration

Note: AgentCore is deprecated. Use src.multi_agent.runner.MultiAgentRunner instead.
"""

from .config import AgentConfig

__version__ = "2.0.0"  # Multi-agent system
__all__ = ["AgentConfig"]
