"""
System Cache (Level 3) - Caches system prompts.

Very high reuse rate, long TTL.
Caches agent system prompts that rarely change.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SystemCache:
    """
    Level 3 cache for system prompts.

    Caches very high-reuse system prompts for agents.
    """

    def __init__(self, ttl_hours: int = 24):
        """
        Initialize system cache.

        Args:
            ttl_hours: Cache TTL in hours
        """
        self.ttl_hours = ttl_hours
        self._cached_prompts: Dict[str, str] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._hit_count = 0
        self._miss_count = 0

        logger.info(f"SystemCache initialized (ttl={ttl_hours}h)")

    def get_cached_prompt(self, agent_name: str, prompt: str) -> str:
        """
        Get or cache system prompt.

        Args:
            agent_name: Agent name
            prompt: System prompt text

        Returns:
            Prompt text (possibly from cache)
        """
        # Check if prompt is cached and valid
        if self._is_prompt_cached(agent_name):
            self._hit_count += 1
            return self._cached_prompts[agent_name]

        # Cache miss - store new prompt
        self._miss_count += 1
        self._cached_prompts[agent_name] = prompt
        self._cache_timestamps[agent_name] = datetime.now()

        return prompt

    def _is_prompt_cached(self, agent_name: str) -> bool:
        """Check if prompt is cached and valid."""
        if agent_name not in self._cached_prompts:
            return False

        timestamp = self._cache_timestamps.get(agent_name)
        if timestamp is None:
            return False

        age = datetime.now() - timestamp
        return age < timedelta(hours=self.ttl_hours)

    def get_cached_content(self) -> Optional[str]:
        """
        Get all cached system prompts concatenated.

        Returns:
            Concatenated cached prompts or None
        """
        if not self._cached_prompts:
            return None

        # Concatenate all cached prompts
        content_parts = [
            f"# {agent_name} System Prompt\n\n{prompt}"
            for agent_name, prompt in self._cached_prompts.items()
        ]

        return "\n\n".join(content_parts)

    def invalidate(self, agent_name: Optional[str] = None) -> None:
        """
        Invalidate cache.

        Args:
            agent_name: Specific agent to invalidate, or None for all
        """
        if agent_name:
            self._cached_prompts.pop(agent_name, None)
            self._cache_timestamps.pop(agent_name, None)

            logger.info(f"System cache invalidated for {agent_name}")
        else:
            self._cached_prompts.clear()
            self._cache_timestamps.clear()

            logger.info("System cache invalidated for all agents")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100) if total > 0 else 0

        total_size = sum(len(prompt) for prompt in self._cached_prompts.values())

        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": round(hit_rate, 1),
            "cached_agents": len(self._cached_prompts),
            "total_cache_size": total_size,
        }
