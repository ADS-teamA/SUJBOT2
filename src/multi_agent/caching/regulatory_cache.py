"""
Regulatory Cache (Level 1) - Caches regulatory documents.

Highest reuse rate, longest TTL.
Examples: GDPR text, CCPA text, HIPAA regulations, ISO standards.

Uses TTLCache from src/utils/cache.py as internal storage (SSOT pattern).
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.cache import TTLCache

logger = logging.getLogger(__name__)

# Cache key for single-value storage
_CONTENT_KEY = "content"


class RegulatoryCache:
    """
    Level 1 cache for regulatory documents.

    Caches high-reuse regulatory content like GDPR, CCPA, HIPAA.
    Uses TTLCache internally for SSOT-compliant caching.
    """

    def __init__(self, regulatory_path: Path, ttl_hours: int = 24):
        """
        Initialize regulatory cache.

        Args:
            regulatory_path: Path to regulatory documents
            ttl_hours: Cache TTL in hours
        """
        self.regulatory_path = regulatory_path
        self.ttl_hours = ttl_hours
        self._cache = TTLCache[str](
            ttl_seconds=ttl_hours * 3600,
            max_size=1,  # Single document cache
            name="RegulatoryCache"
        )

        logger.info(f"RegulatoryCache initialized (path={regulatory_path}, ttl={ttl_hours}h)")

    def get_cached_content(self) -> Optional[str]:
        """
        Get cached regulatory content.

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
        """Load regulatory content from disk."""
        if not self.regulatory_path.exists():
            logger.warning(f"Regulatory path does not exist: {self.regulatory_path}")
            return None

        try:
            # Load all .txt files from regulatory directory
            content_parts = []

            for file_path in sorted(self.regulatory_path.glob("*.txt")):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        content_parts.append(f"## {file_path.stem}\n\n{content}")

                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            if content_parts:
                loaded_content = "\n\n".join(content_parts)
                logger.info(
                    f"Loaded {len(content_parts)} regulatory documents "
                    f"({len(loaded_content)} chars)"
                )
                return loaded_content
            else:
                logger.warning("No regulatory documents found")
                return None

        except Exception as e:
            logger.error(f"Failed to load regulatory content: {e}", exc_info=True)
            return None

    def invalidate(self) -> None:
        """Invalidate cache."""
        self._cache.clear()
        logger.info("Regulatory cache invalidated")

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
