"""
Contract Cache (Level 2) - Caches contract templates.

Medium reuse rate, medium TTL.
Examples: Standard contract clauses, boilerplate text, template sections.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ContractCache:
    """
    Level 2 cache for contract templates.

    Caches medium-reuse contract templates and boilerplate.
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
        self._cached_content: Optional[str] = None
        self._cache_timestamp: Optional[datetime] = None
        self._hit_count = 0
        self._miss_count = 0

        logger.info(f"ContractCache initialized (path={contract_path}, ttl={ttl_hours}h)")

    def get_cached_content(self) -> Optional[str]:
        """
        Get cached contract content.

        Returns:
            Cached content string or None
        """
        if self._is_cache_valid():
            self._hit_count += 1
            return self._cached_content

        self._miss_count += 1
        self._load_content()

        return self._cached_content

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cached_content is None or self._cache_timestamp is None:
            return False

        age = datetime.now() - self._cache_timestamp
        return age < timedelta(hours=self.ttl_hours)

    def _load_content(self) -> None:
        """Load contract content from disk."""
        if not self.contract_path.exists():
            logger.warning(f"Contract path does not exist: {self.contract_path}")
            self._cached_content = None
            return

        try:
            content_parts = []

            for file_path in sorted(self.contract_path.glob("*.txt")):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        content_parts.append(f"## {file_path.stem}\n\n{content}")

                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            if content_parts:
                self._cached_content = "\n\n".join(content_parts)
                self._cache_timestamp = datetime.now()

                logger.info(
                    f"Loaded {len(content_parts)} contract templates "
                    f"({len(self._cached_content)} chars)"
                )
            else:
                logger.warning("No contract templates found")
                self._cached_content = None

        except Exception as e:
            logger.error(f"Failed to load contract content: {e}", exc_info=True)
            self._cached_content = None

    def invalidate(self) -> None:
        """Invalidate cache."""
        self._cached_content = None
        self._cache_timestamp = None

        logger.info("Contract cache invalidated")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total * 100) if total > 0 else 0

        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": round(hit_rate, 1),
            "cached": self._cached_content is not None,
            "cache_size": len(self._cached_content) if self._cached_content else 0,
        }
