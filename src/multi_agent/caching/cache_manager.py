"""
Cache Manager - Orchestrates 3-level caching strategy.

Coordinates:
1. Regulatory Cache (Level 1) - Regulatory documents
2. Contract Cache (Level 2) - Contract templates
3. System Cache (Level 3) - System prompts

All caches use TTLCache from src/utils/cache.py as internal storage (SSOT pattern).
Implements Harvey AI's prompt caching strategy for 90% cost savings.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from .regulatory_cache import RegulatoryCache
from .contract_cache import ContractCache
from .system_cache import SystemCache

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages 3-level caching strategy for prompt optimization.

    Provides unified interface for all cache levels and coordinates
    cache updates, invalidation, and statistics.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize cache manager with configuration.

        Args:
            config: Caching configuration dict
        """
        self.config = config

        # Cache TTL (hours)
        self.cache_ttl_hours = config.get("cache_ttl_hours", 24)

        # Paths to cached content
        self.regulatory_docs_path = Path(
            config.get("regulatory_docs_path", "data/regulatory_templates")
        )
        self.contract_templates_path = Path(
            config.get("contract_templates_path", "data/contract_templates")
        )

        # Initialize cache levels
        self.regulatory_cache = (
            RegulatoryCache(self.regulatory_docs_path, self.cache_ttl_hours)
            if config.get("enable_regulatory_cache", True)
            else None
        )

        self.contract_cache = (
            ContractCache(self.contract_templates_path, self.cache_ttl_hours)
            if config.get("enable_contract_cache", True)
            else None
        )

        self.system_cache = (
            SystemCache(self.cache_ttl_hours) if config.get("enable_system_cache", True) else None
        )

        logger.info(
            f"CacheManager initialized: "
            f"regulatory={'enabled' if self.regulatory_cache else 'disabled'}, "
            f"contract={'enabled' if self.contract_cache else 'disabled'}, "
            f"system={'enabled' if self.system_cache else 'disabled'}"
        )

    def prepare_cached_content(self) -> Dict[str, Any]:
        """
        Prepare all cached content for prompt caching.

        Returns:
            Dict with cached content organized by cache level
        """
        cached_content = {}

        # Level 1: Regulatory documents
        if self.regulatory_cache:
            regulatory_content = self.regulatory_cache.get_cached_content()
            if regulatory_content:
                cached_content["regulatory"] = {
                    "type": "text",
                    "text": regulatory_content,
                    "cache_control": {"type": "ephemeral"},
                }

        # Level 2: Contract templates
        if self.contract_cache:
            contract_content = self.contract_cache.get_cached_content()
            if contract_content:
                cached_content["contract"] = {
                    "type": "text",
                    "text": contract_content,
                    "cache_control": {"type": "ephemeral"},
                }

        # Level 3: System prompts
        if self.system_cache:
            system_content = self.system_cache.get_cached_content()
            if system_content:
                cached_content["system"] = {
                    "type": "text",
                    "text": system_content,
                    "cache_control": {"type": "ephemeral"},
                }

        logger.debug(f"Prepared cached content: {list(cached_content.keys())}")

        return cached_content

    def build_cached_system_prompt(
        self, base_prompt: str, include_regulatory: bool = True, include_contracts: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Build system prompt with cached content.

        Args:
            base_prompt: Base agent system prompt
            include_regulatory: Include regulatory cache
            include_contracts: Include contract cache

        Returns:
            List of system message parts with cache control
        """
        system_parts = []

        # Add cached regulatory documents (if enabled)
        if include_regulatory and self.regulatory_cache:
            regulatory_content = self.regulatory_cache.get_cached_content()
            if regulatory_content:
                system_parts.append(
                    {
                        "type": "text",
                        "text": f"# Regulatory Framework\n\n{regulatory_content}",
                        "cache_control": {"type": "ephemeral"},
                    }
                )

        # Add cached contract templates (if enabled)
        if include_contracts and self.contract_cache:
            contract_content = self.contract_cache.get_cached_content()
            if contract_content:
                system_parts.append(
                    {
                        "type": "text",
                        "text": f"# Contract Templates\n\n{contract_content}",
                        "cache_control": {"type": "ephemeral"},
                    }
                )

        # Add base prompt with cache control
        system_parts.append(
            {"type": "text", "text": base_prompt, "cache_control": {"type": "ephemeral"}}
        )

        return system_parts

    def invalidate_all(self) -> None:
        """Invalidate all caches."""
        if self.regulatory_cache:
            self.regulatory_cache.invalidate()

        if self.contract_cache:
            self.contract_cache.invalidate()

        if self.system_cache:
            self.system_cache.invalidate()

        logger.info("All caches invalidated")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get caching statistics.

        Returns:
            Dict with cache stats for all levels
        """
        stats = {
            "regulatory_cache": (
                self.regulatory_cache.get_stats() if self.regulatory_cache else None
            ),
            "contract_cache": self.contract_cache.get_stats() if self.contract_cache else None,
            "system_cache": self.system_cache.get_stats() if self.system_cache else None,
        }

        return stats


def create_cache_manager(config: Dict[str, Any]) -> Optional[CacheManager]:
    """
    Create cache manager from configuration.

    Args:
        config: Multi-agent config dict

    Returns:
        CacheManager instance or None if caching disabled
    """
    caching_config = config.get("caching", {})

    # Check if any caching is enabled
    if not any(
        [
            caching_config.get("enable_regulatory_cache", True),
            caching_config.get("enable_contract_cache", True),
            caching_config.get("enable_system_cache", True),
        ]
    ):
        logger.info("All caching disabled")
        return None

    try:
        cache_manager = CacheManager(caching_config)

        logger.info("Cache manager created successfully")

        return cache_manager

    except Exception as e:
        # Provide actionable error context for debugging
        regulatory_path = caching_config.get("regulatory_docs_path", "data/regulatory_templates")
        contract_path = caching_config.get("contract_templates_path", "data/contract_templates")

        logger.error(
            f"Failed to create cache manager: {e}. "
            f"Caching will be DISABLED for this session. "
            f"Troubleshooting: (1) Check regulatory path exists: {regulatory_path}, "
            f"(2) Check contract path exists: {contract_path}, "
            f"(3) Verify file read permissions, "
            f"(4) Ensure UTF-8 encoding for all template files.",
            exc_info=True,
        )
        return None
