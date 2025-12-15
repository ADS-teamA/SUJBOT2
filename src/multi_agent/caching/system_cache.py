"""
System Cache (Level 3) - Caches system prompts.

Very high reuse rate, long TTL.
Caches agent system prompts that rarely change.

Uses TTLCache from src/utils/cache.py as internal storage (SSOT pattern).
"""

import logging
from typing import Optional, Dict, Any

from src.utils.cache import TTLCache

logger = logging.getLogger(__name__)


class SystemCache:
    """
    Level 3 cache for system prompts.

    Caches very high-reuse system prompts for agents.
    Uses TTLCache internally for SSOT-compliant caching.
    """

    def __init__(self, ttl_hours: int = 24):
        """
        Initialize system cache.

        Args:
            ttl_hours: Cache TTL in hours
        """
        self.ttl_hours = ttl_hours
        self._cache = TTLCache[str](
            ttl_seconds=ttl_hours * 3600,
            name="SystemCache"
        )

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
        cached = self._cache.get(agent_name)
        if cached is not None:
            return cached

        # Cache miss - store new prompt
        self._cache.set(agent_name, prompt)
        return prompt

    def get_cached_content(self) -> Optional[str]:
        """
        Get all cached system prompts concatenated.

        Returns:
            Concatenated cached prompts or None
        """
        # Use public items() API for thread-safe iteration over non-expired entries
        cached_items = self._cache.items()
        if not cached_items:
            return None

        # Build content from all cached entries
        content_parts = [
            f"# {agent_name} System Prompt\n\n{prompt}"
            for agent_name, prompt in cached_items
        ]

        return "\n\n".join(content_parts) if content_parts else None

    def invalidate(self, agent_name: Optional[str] = None) -> None:
        """
        Invalidate cache.

        Args:
            agent_name: Specific agent to invalidate, or None for all
        """
        if agent_name:
            self._cache.delete(agent_name)
            logger.info(f"System cache invalidated for {agent_name}")
        else:
            self._cache.clear()
            logger.info("System cache invalidated for all agents")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        base_stats = self._cache.get_stats()

        # Calculate total size using public API
        total_size = self._cache.size_bytes()

        return {
            "hits": base_stats["hits"],
            "misses": base_stats["misses"],
            "hit_rate": base_stats["hit_rate"] * 100,  # Convert to percentage
            "cached_agents": base_stats["size"],
            "total_cache_size": total_size,
        }
