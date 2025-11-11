"""
Agent Registry - Central registry for all agents.

Provides:
- Agent discovery and instantiation
- Configuration loading
- Tool validation
- Agent lifecycle management

Pattern: Registry + Factory
"""

import logging
from typing import Dict, List, Optional, Set, Type

from .agent_base import BaseAgent, AgentConfig, AgentRole, AgentTier

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for all multi-agent system agents.

    Manages agent lifecycle:
    1. Registration (at import time via decorator)
    2. Configuration (from config.json)
    3. Instantiation (lazy, on-demand)
    4. Validation (tools, config)
    """

    def __init__(self):
        """Initialize empty registry."""
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        self._agent_instances: Dict[str, BaseAgent] = {}
        self._configs: Dict[str, AgentConfig] = {}

    def register_agent_class(
        self,
        agent_name: str,
        agent_class: Type[BaseAgent]
    ) -> None:
        """
        Register an agent class.

        Args:
            agent_name: Unique agent identifier
            agent_class: Agent class to register
        """
        if agent_name in self._agent_classes:
            logger.warning(f"Agent {agent_name} already registered, overwriting")

        self._agent_classes[agent_name] = agent_class
        logger.debug(f"Registered agent class: {agent_name}")

    def register_config(self, agent_name: str, config: AgentConfig) -> None:
        """
        Register agent configuration.

        Args:
            agent_name: Agent identifier
            config: Agent configuration
        """
        config.validate()
        self._configs[agent_name] = config
        logger.debug(f"Registered config for agent: {agent_name}")

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get agent instance (creates if not exists).

        Args:
            agent_name: Agent identifier

        Returns:
            Agent instance or None if not found
        """
        # Return cached instance if exists
        if agent_name in self._agent_instances:
            return self._agent_instances[agent_name]

        # Check if agent class registered
        if agent_name not in self._agent_classes:
            logger.error(f"Agent {agent_name} not registered")
            return None

        # Check if config exists
        if agent_name not in self._configs:
            logger.error(f"No config for agent {agent_name}")
            return None

        # Instantiate agent
        try:
            agent_class = self._agent_classes[agent_name]
            config = self._configs[agent_name]
            agent_instance = agent_class(config)

            # Cache instance
            self._agent_instances[agent_name] = agent_instance
            logger.info(f"Instantiated agent: {agent_name}")

            return agent_instance

        except Exception as e:
            logger.error(f"Failed to instantiate agent {agent_name}: {e}")
            return None

    def get_all_agents(self) -> List[BaseAgent]:
        """
        Get all registered agent instances.

        Returns:
            List of all agent instances
        """
        agents = []
        for agent_name in self._agent_classes.keys():
            agent = self.get_agent(agent_name)
            if agent:
                agents.append(agent)
        return agents

    def get_agents_by_role(self, role: AgentRole) -> List[BaseAgent]:
        """
        Get agents by role.

        Args:
            role: Agent role

        Returns:
            List of agents with specified role
        """
        return [
            agent
            for agent in self.get_all_agents()
            if agent.config.role == role
        ]

    def get_agents_by_tier(self, tier: AgentTier) -> List[BaseAgent]:
        """
        Get agents by tier.

        Args:
            tier: Agent tier

        Returns:
            List of agents in specified tier
        """
        return [
            agent
            for agent in self.get_all_agents()
            if agent.config.tier == tier
        ]

    def validate_all_agents(self, available_tools: Set[str]) -> bool:
        """
        Validate all agents have their required tools available.

        Args:
            available_tools: Set of available tool names

        Returns:
            True if all agents valid, False otherwise
        """
        all_valid = True
        for agent in self.get_all_agents():
            if not agent.validate_tools(available_tools):
                all_valid = False
                logger.error(
                    f"Agent {agent.config.name} validation failed "
                    f"(missing tools)"
                )

        return all_valid

    def get_stats(self) -> Dict[str, any]:
        """
        Get registry statistics.

        Returns:
            Dict with registry stats
        """
        agents = self.get_all_agents()

        return {
            "total_registered": len(self._agent_classes),
            "total_configured": len(self._configs),
            "total_instantiated": len(self._agent_instances),
            "agents": [agent.get_stats() for agent in agents],
            "by_tier": {
                tier.value: len(self.get_agents_by_tier(tier))
                for tier in AgentTier
            },
            "by_role": {
                role.value: len(self.get_agents_by_role(role))
                for role in AgentRole
            },
        }

    def clear(self) -> None:
        """Clear all cached agent instances (for testing)."""
        self._agent_instances.clear()
        logger.info("Cleared all agent instances")

    def __len__(self) -> int:
        """Number of registered agent classes."""
        return len(self._agent_classes)

    def __contains__(self, agent_name: str) -> bool:
        """Check if agent is registered."""
        return agent_name in self._agent_classes


# Global registry instance
_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """Get global agent registry instance."""
    return _registry


def register_agent(agent_name: str):
    """
    Decorator to register an agent class.

    Usage:
        @register_agent("extractor")
        class ExtractorAgent(BaseAgent):
            ...
    """
    def decorator(agent_class: Type[BaseAgent]):
        _registry.register_agent_class(agent_name, agent_class)
        return agent_class
    return decorator
