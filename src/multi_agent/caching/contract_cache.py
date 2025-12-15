"""
Contract Cache (Level 2) - Caches contract templates.

Medium reuse rate, medium TTL.
Examples: Standard contract clauses, boilerplate text, template sections.

Uses TTLCache from src/utils/cache.py as internal storage (SSOT pattern).
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.cache import TTLCache

logger = logging.getLogger(__name__)

# Cache key for single-value storage
_CONTENT_KEY = "content"


class ContractCache:
    """
    Level 2 cache for contract templates.

    Caches medium-reuse contract templates and boilerplate.
    Uses TTLCache internally for SSOT-compliant caching.
    """

    def __init__(self, contract_path: Path, ttl_hours: int = 24):
        """
        Initialize contract cache.

        Args:
            contract_path: Path to contract templates
            ttl_hours: Cache TTL in hours
        """
        self.contract_path = contract_path
        self.ttl_hours = ttl_hours
        self._cache = TTLCache[str](
            ttl_seconds=ttl_hours * 3600,
            max_size=1,  # Single document cache
            name="ContractCache"
        )

        logger.info(f"ContractCache initialized (path={contract_path}, ttl={ttl_hours}h)")

    def get_cached_content(self) -> Optional[str]:
        """
        Get cached contract content.

        Returns:
            Cached content string or None
        """
        # Check if content is cached and valid
        cached = self._cache.get(_CONTENT_KEY)
        if cached is not None:
            return cached

        # Cache miss - load from disk and cache
        content = self._load_content()
        if content:
            self._cache.set(_CONTENT_KEY, content)

        return content

    def _load_content(self) -> Optional[str]:
        """Load contract content from disk."""
        if not self.contract_path.exists():
            logger.warning(f"Contract path does not exist: {self.contract_path}")
            return None

        try:
            content_parts = []

            for file_path in sorted(self.contract_path.glob("*.txt")):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        content_parts.append(f"## {file_path.stem}\n\n{content}")

                except Exception as e:
                    logger.warning(
                        f"Failed to load contract template {file_path.name}: {e}. "
                        f"This template will NOT be included in prompts. "
                        f"Check file encoding (must be UTF-8) and read permissions."
                    )

            if content_parts:
                loaded_content = "\n\n".join(content_parts)
                logger.info(
                    f"Loaded {len(content_parts)} contract templates "
                    f"({len(loaded_content)} chars)"
                )
                return loaded_content
            else:
                logger.warning("No contract templates found")
                return None

        except Exception as e:
            logger.error(f"Failed to load contract content: {e}", exc_info=True)
            return None

    def invalidate(self) -> None:
        """Invalidate cache."""
        self._cache.clear()
        logger.info("Contract cache invalidated")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        base_stats = self._cache.get_stats()

        # Get cached content size
        cached_content = self._cache.get(_CONTENT_KEY)
        cache_size = len(cached_content) if cached_content else 0

        return {
            "hits": base_stats["hits"],
            "misses": base_stats["misses"],
            "hit_rate": base_stats["hit_rate"] * 100,  # Convert to percentage
            "cached": base_stats["size"] > 0,
            "cache_size": cache_size,
        }
