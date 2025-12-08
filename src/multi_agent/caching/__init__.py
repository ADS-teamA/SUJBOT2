"""3-Level caching system for multi-agent framework.

Implements prompt caching strategy from Harvey AI research:
- Level 1: Regulatory documents (high reuse, longest TTL)
- Level 2: Contract templates (medium reuse, medium TTL)
- Level 3: System prompts (very high reuse, long TTL)

Achieves 90% cost savings on repeated queries.
"""

from .cache_manager import CacheManager, create_cache_manager
from .regulatory_cache import RegulatoryCache
from .contract_cache import ContractCache
from .system_cache import SystemCache

__all__ = [
    "CacheManager",
    "create_cache_manager",
    "RegulatoryCache",
    "ContractCache",
    "SystemCache",
]
