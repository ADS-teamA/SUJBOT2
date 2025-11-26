"""
Unified cache abstractions for SUJBOT2.

Provides SSOT (Single Source of Truth) caching implementations:
- LRUCache: Least Recently Used cache with size limit
- TTLCache: Time-To-Live cache with expiration

Usage:
    from src.utils.cache import LRUCache, TTLCache

    # LRU cache for embeddings
    embedding_cache = LRUCache[List[float]](max_size=1000)
    embedding_cache.set("key", [0.1, 0.2, 0.3])
    result = embedding_cache.get("key")

    # TTL cache for regulatory documents
    doc_cache = TTLCache[dict](ttl_seconds=3600)
    doc_cache.set("doc_id", {"content": "..."})
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, Generic, Optional, TypeVar, Tuple
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseCache(ABC, Generic[T]):
    """Abstract base class for all cache implementations."""

    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get value by key, returns None if not found or expired."""
        pass

    @abstractmethod
    def set(self, key: str, value: T) -> None:
        """Set value by key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key, returns True if key existed."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of items in cache."""
        pass

    def __contains__(self, key: str) -> bool:
        """Check if key exists (and is not expired)."""
        return self.get(key) is not None


class LRUCache(BaseCache[T]):
    """
    Least Recently Used (LRU) cache with max size limit.

    Thread-safe implementation using OrderedDict.
    When cache is full, evicts least recently used items.

    Args:
        max_size: Maximum number of items to store (default: 500)
        name: Optional name for logging
    """

    def __init__(self, max_size: int = 500, name: Optional[str] = None):
        self.max_size = max_size
        self.name = name or "LRUCache"
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value and move to end (most recently used)."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: T) -> None:
        """Set value, evicting LRU items if needed."""
        with self._lock:
            if key in self._cache:
                # Update existing and move to end
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                # Add new item
                self._cache[key] = value
                # Evict oldest if over limit
                while len(self._cache) > self.max_size:
                    evicted_key, _ = self._cache.popitem(last=False)
                    logger.debug(f"{self.name}: Evicted LRU key: {evicted_key[:50]}...")

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            logger.debug(f"{self.name}: Cleared all entries")

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "name": self.name,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self.hit_rate, 3),
            }


@dataclass
class _TTLEntry(Generic[T]):
    """Internal entry for TTL cache with expiration time."""
    value: T
    expires_at: datetime


class TTLCache(BaseCache[T]):
    """
    Time-To-Live (TTL) cache with automatic expiration.

    Thread-safe implementation. Expired entries are lazily removed
    on access or during periodic cleanup.

    Args:
        ttl_seconds: Time-to-live in seconds (default: 3600 = 1 hour)
        max_size: Optional maximum size (default: None = unlimited)
        name: Optional name for logging
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: Optional[int] = None,
        name: Optional[str] = None
    ):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.name = name or "TTLCache"
        self._cache: Dict[str, _TTLEntry[T]] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value if not expired."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if datetime.now() < entry.expires_at:
                    self._hits += 1
                    return entry.value
                else:
                    # Expired - remove it
                    del self._cache[key]
                    logger.debug(f"{self.name}: Key expired: {key[:50]}...")
            self._misses += 1
            return None

    def set(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> None:
        """Set value with TTL (uses default TTL if not specified)."""
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl)

        with self._lock:
            # Evict if over max_size
            if self.max_size and len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_expired_or_oldest()

            self._cache[key] = _TTLEntry(value=value, expires_at=expires_at)

    def _evict_expired_or_oldest(self) -> None:
        """Evict expired entries, or oldest if none expired (called with lock held)."""
        now = datetime.now()

        # First try to evict expired entries
        expired_keys = [k for k, v in self._cache.items() if v.expires_at <= now]
        for key in expired_keys:
            del self._cache[key]
            logger.debug(f"{self.name}: Evicted expired key: {key[:50]}...")

        # If still over limit, evict oldest (by expiration time)
        if self.max_size and len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].expires_at)
            del self._cache[oldest_key]
            logger.debug(f"{self.name}: Evicted oldest key: {oldest_key[:50]}...")

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            logger.debug(f"{self.name}: Cleared all entries")

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        now = datetime.now()
        with self._lock:
            expired_keys = [k for k, v in self._cache.items() if v.expires_at <= now]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                logger.debug(f"{self.name}: Cleaned up {len(expired_keys)} expired entries")
            return len(expired_keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        with self._lock:
            now = datetime.now()
            active_count = sum(1 for v in self._cache.values() if v.expires_at > now)
            expired_count = len(self._cache) - active_count

            return {
                "name": self.name,
                "size": len(self._cache),
                "active": active_count,
                "expired": expired_count,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self.hit_rate, 3),
            }


# Convenience function for creating caches
def create_cache(
    cache_type: str = "lru",
    max_size: int = 500,
    ttl_seconds: int = 3600,
    name: Optional[str] = None
) -> BaseCache:
    """
    Factory function for creating caches.

    Args:
        cache_type: "lru" or "ttl"
        max_size: Maximum size (for LRU, optional for TTL)
        ttl_seconds: TTL in seconds (for TTL cache)
        name: Optional name for logging

    Returns:
        BaseCache instance
    """
    if cache_type == "lru":
        return LRUCache(max_size=max_size, name=name)
    elif cache_type == "ttl":
        return TTLCache(ttl_seconds=ttl_seconds, max_size=max_size, name=name)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}. Use 'lru' or 'ttl'.")
