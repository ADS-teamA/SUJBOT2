"""
Cost tracking for API usage (LLM and Embeddings).

Tracks token usage and calculates costs for:
- Anthropic Claude (Haiku, Sonnet, Opus)
- OpenAI (GPT-4o, o-series, embeddings)
- Voyage AI (embeddings)
- Local models (free)

Usage:
    from src.cost_tracker import CostTracker

    tracker = CostTracker()

    # Track LLM usage
    tracker.track_llm(
        provider="anthropic",
        model="claude-haiku-4-5",
        input_tokens=1000,
        output_tokens=500
    )

    # Track embedding usage
    tracker.track_embedding(
        provider="openai",
        model="text-embedding-3-large",
        tokens=10000
    )

    # Get total cost
    cost = tracker.get_total_cost()
    print(f"Total cost: ${cost:.4f}")
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

from src.exceptions import StorageError

logger = logging.getLogger(__name__)


# ====================================================================
# PRICING DATA (2025)
# ====================================================================
# Source: https://docs.anthropic.com/pricing, https://openai.com/api/pricing/
# Updated: January 2025

PRICING = {
    # Anthropic Claude models (per 1M tokens)
    "anthropic": {
        # Haiku models
        "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
        "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
        "haiku": {"input": 1.00, "output": 5.00},
        "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
        "claude-haiku-3": {"input": 0.25, "output": 1.25},
        # Sonnet models
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        "sonnet": {"input": 3.00, "output": 15.00},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "claude-sonnet-3-5": {"input": 3.00, "output": 15.00},
        # Opus models (Opus 4.5 = $5/$25, older Opus 4/4.1 = $15/$75)
        "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
        "claude-opus-4-5": {"input": 5.00, "output": 25.00},
        "claude-opus-4": {"input": 15.00, "output": 75.00},
        "claude-opus-4-1": {"input": 15.00, "output": 75.00},
        "opus": {"input": 5.00, "output": 25.00},  # Default to latest Opus 4.5
    },
    # OpenAI models (per 1M tokens)
    "openai": {
        # GPT-4o models
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        # O-series reasoning models
        "o1": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
        "o3": {"input": 20.00, "output": 80.00},  # Estimated
        "o3-mini": {"input": 3.00, "output": 12.00},
        "o3-pro": {"input": 30.00, "output": 120.00},  # Estimated
        "o4-mini": {"input": 3.00, "output": 12.00},
        # Embeddings (per 1M tokens)
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    },
    # Voyage AI embeddings (per 1M tokens)
    "voyage": {
        "voyage-3-large": {"input": 0.12, "output": 0.0},  # Estimated
        "voyage-3": {"input": 0.06, "output": 0.0},
        "voyage-3-lite": {"input": 0.02, "output": 0.0},
        "voyage-law-2": {"input": 0.12, "output": 0.0},  # Estimated
        "voyage-finance-2": {"input": 0.12, "output": 0.0},  # Estimated
        "voyage-multilingual-2": {"input": 0.12, "output": 0.0},  # Estimated
        "kanon-2": {"input": 0.12, "output": 0.0},  # Estimated
    },
    # Google Gemini models (per 1M tokens)
    "google": {
        # Gemini 2.5 models
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        # Aliases
        "gemini": {"input": 0.30, "output": 2.50},
        "gemini-flash": {"input": 0.30, "output": 2.50},
        "gemini-pro": {"input": 1.25, "output": 10.00},
    },
    # DeepInfra models (per 1M tokens)
    # Source: https://deepinfra.com/pricing
    "deepinfra": {
        # LLM models
        "Qwen/Qwen2.5-72B-Instruct": {"input": 0.35, "output": 0.40},
        "Qwen/Qwen2.5-7B-Instruct": {"input": 0.06, "output": 0.06},
        "qwen-72b": {"input": 0.35, "output": 0.40},
        "qwen-7b": {"input": 0.06, "output": 0.06},
        # Llama models
        "meta-llama/Meta-Llama-3.1-70B-Instruct": {"input": 0.35, "output": 0.40},
        "meta-llama/Meta-Llama-3.1-8B-Instruct": {"input": 0.06, "output": 0.06},
        "llama-70b": {"input": 0.35, "output": 0.40},
        # Embedding models
        "Qwen/Qwen3-Embedding-8B": {"input": 0.03, "output": 0.0},
        "qwen3-embedding-8b": {"input": 0.03, "output": 0.0},
        "BAAI/bge-m3": {"input": 0.01, "output": 0.0},
        "bge-m3": {"input": 0.01, "output": 0.0},
        "BAAI/bge-large-en-v1.5": {"input": 0.01, "output": 0.0},
        "intfloat/e5-large-v2": {"input": 0.01, "output": 0.0},
    },
    # Local models (free)
    "huggingface": {
        "bge-m3": {"input": 0.0, "output": 0.0},
        "BAAI/bge-m3": {"input": 0.0, "output": 0.0},
        "bge-large": {"input": 0.0, "output": 0.0},
    },
}


@dataclass
class UsageEntry:
    """Single usage entry (LLM or embedding)."""

    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    operation: str  # "summary", "context", "embedding", "agent", etc.
    cache_creation_tokens: int = 0  # Tokens written to cache
    cache_read_tokens: int = 0  # Tokens read from cache
    response_time_ms: float = 0.0  # LLM response time in milliseconds (measured via time.time()). Accumulates per agent in get_agent_breakdown(). Always 0.0 for embedding operations.


class CostTracker:
    """
    Track API costs across indexing pipeline and RAG agent.

    Features:
    - Track token usage for LLM and embeddings
    - Calculate costs based on current pricing
    - Support multiple providers (Anthropic, OpenAI, Voyage)
    - Session-based tracking (reset for each indexing/conversation)
    - Detailed breakdown by operation type
    - Immutable public interface (private fields with read-only properties)

    Usage:
        tracker = CostTracker()
        tracker.track_llm("anthropic", "haiku", 1000, 500, "summary")
        tracker.track_embedding("openai", "text-embedding-3-large", 10000, "indexing")

        print(tracker.get_summary())
        print(f"Total cost: ${tracker.total_cost:.4f}")
    """

    def __init__(self):
        """Initialize cost tracker with private fields."""
        # Private storage - prevents external mutation
        self._entries: List[UsageEntry] = []

        # Private accumulators
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost: float = 0.0

        # Cache tracking (for total tokens calculation)
        self._total_cache_read_tokens: int = 0
        self._total_cache_creation_tokens: int = 0

        # Per-message tracking (for CLI display)
        self._last_reported_cost: float = 0.0
        self._last_reported_tokens: int = 0
        self._last_reported_input_tokens: int = 0
        self._last_reported_output_tokens: int = 0
        self._last_reported_cache_read_tokens: int = 0

        # Private breakdowns
        self._cost_by_provider: Dict[str, float] = {}
        self._cost_by_operation: Dict[str, float] = {}

    # Read-only properties for public access
    @property
    def entries(self) -> List[UsageEntry]:
        """Get copy of usage entries (read-only)."""
        return self._entries.copy()

    @property
    def total_input_tokens(self) -> int:
        """Get total input tokens."""
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        """Get total output tokens."""
        return self._total_output_tokens

    @property
    def total_cost(self) -> float:
        """Get total cost."""
        return self._total_cost

    @property
    def cost_by_provider(self) -> Dict[str, float]:
        """Get cost breakdown by provider (read-only copy)."""
        return self._cost_by_provider.copy()

    @property
    def cost_by_operation(self) -> Dict[str, float]:
        """Get cost breakdown by operation (read-only copy)."""
        return self._cost_by_operation.copy()

    def track_llm(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str = "llm",
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        response_time_ms: float = 0.0,
    ) -> float:
        """
        Track LLM usage and calculate cost.

        Args:
            provider: "anthropic", "openai", "claude"
            model: Model name or alias
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation: Operation type ("summary", "context", "agent", etc.)
            cache_creation_tokens: Tokens written to cache (Anthropic only)
            cache_read_tokens: Tokens read from cache (Anthropic only)
            response_time_ms: Response time in milliseconds

        Returns:
            Cost in USD for this call
        """
        # Normalize provider name
        if provider == "claude":
            provider = "anthropic"

        # Get pricing (includes cache cost calculation)
        cost = self._calculate_llm_cost(
            provider, model, input_tokens, output_tokens, cache_read_tokens
        )

        # Store entry
        entry = UsageEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            operation=operation,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            response_time_ms=response_time_ms,
        )

        # Debug log response time tracking (always log to debug the issue)
        logger.info(
            f"💫 track_llm: operation={operation}, response_time={response_time_ms:.2f}ms, "
            f"input={input_tokens}, output={output_tokens}"
        )

        self._entries.append(entry)

        # Update accumulators
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost += cost

        # Update cache accumulators
        self._total_cache_read_tokens += cache_read_tokens
        self._total_cache_creation_tokens += cache_creation_tokens

        # Update breakdowns
        self._cost_by_provider[provider] = self._cost_by_provider.get(provider, 0.0) + cost
        self._cost_by_operation[operation] = self._cost_by_operation.get(operation, 0.0) + cost

        # Log with cache info if applicable
        if cache_creation_tokens > 0 or cache_read_tokens > 0:
            logger.debug(
                f"LLM usage tracked: {provider}/{model} - "
                f"{input_tokens} in, {output_tokens} out - ${cost:.6f} "
                f"(cache: {cache_read_tokens} read, {cache_creation_tokens} created)"
            )
        else:
            logger.debug(
                f"LLM usage tracked: {provider}/{model} - "
                f"{input_tokens} in, {output_tokens} out - ${cost:.6f}"
            )

        return cost

    def track_embedding(
        self, provider: str, model: str, tokens: int, operation: str = "embedding"
    ) -> float:
        """
        Track embedding usage and calculate cost.

        Args:
            provider: "openai", "voyage", "huggingface"
            model: Model name
            tokens: Number of tokens embedded
            operation: Operation type ("indexing", "query", etc.)

        Returns:
            Cost in USD for this call
        """
        # Get pricing (embeddings only have input cost)
        cost = self._calculate_embedding_cost(provider, model, tokens)

        # Store entry
        entry = UsageEntry(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=tokens,
            output_tokens=0,
            cost=cost,
            operation=operation,
        )
        self._entries.append(entry)

        # Update accumulators
        self._total_input_tokens += tokens
        self._total_cost += cost

        # Update breakdowns
        self._cost_by_provider[provider] = self._cost_by_provider.get(provider, 0.0) + cost
        self._cost_by_operation[operation] = self._cost_by_operation.get(operation, 0.0) + cost

        logger.debug(
            f"Embedding usage tracked: {provider}/{model} - " f"{tokens} tokens - ${cost:.6f}"
        )

        return cost

    def _calculate_llm_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
    ) -> float:
        """
        Calculate cost for LLM usage, including cache discount.

        Cache reads are billed at 10% of the regular input price (Anthropic prompt caching).
        Note: Cache calculation applies to all calls, even though only Anthropic supports it.
        Non-Anthropic providers should pass cache_read_tokens=0.

        Args:
            provider: Provider name
            model: Model name
            input_tokens: Regular input tokens (100% price)
            output_tokens: Output tokens (100% price)
            cache_read_tokens: Cache hit tokens (10% price, Anthropic only)

        Returns:
            Total cost in USD
        """
        # Get pricing for this model
        pricing = PRICING.get(provider, {}).get(model)

        if not pricing:
            logger.warning(
                f"No pricing data for {provider}/{model}. "
                f"Cost calculation skipped. Add pricing to PRICING dict."
            )
            return 0.0

        # Calculate cost (prices are per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        # Cache reads are billed at 10% of input price
        cache_cost = (cache_read_tokens / 1_000_000) * pricing["input"] * 0.1

        return input_cost + output_cost + cache_cost

    def _calculate_embedding_cost(self, provider: str, model: str, tokens: int) -> float:
        """Calculate cost for embedding usage."""
        # Get pricing for this model
        pricing = PRICING.get(provider, {}).get(model)

        if not pricing:
            logger.warning(f"No pricing data for {provider}/{model}. " f"Cost calculation skipped.")
            return 0.0

        # Calculate cost (prices are per 1M tokens)
        return (tokens / 1_000_000) * pricing["input"]

    def get_total_cost(self) -> float:
        """Get total cost in USD."""
        return self.total_cost

    def get_total_tokens(self) -> int:
        """
        Get total tokens actually used (including cache reads).

        This is the true number of tokens processed by the API,
        not the billed amount (which is discounted for cache hits).

        Use this for: Display purposes, bandwidth estimation
        Use get_billed_tokens() for: Budget tracking, cost projection

        Returns:
            Total tokens: input + output + cache_read
        """
        total = self.total_input_tokens + self.total_output_tokens + self._total_cache_read_tokens
        if total < 0:
            logger.error(f"Invalid negative token count detected: {total}")
            return 0
        return total

    def get_billed_tokens(self) -> int:
        """
        Get equivalent billed tokens (cache reads count as 10% of regular tokens).

        Use this for: Budget tracking, cost projection
        Use get_total_tokens() for: Display purposes, bandwidth estimation

        Returns:
            Billed token equivalent
        """
        # Cache reads are billed at 10% of regular price
        cache_billed_equivalent = int(self._total_cache_read_tokens * 0.1)
        total = self.total_input_tokens + self.total_output_tokens + cache_billed_equivalent
        if total < 0:
            logger.error(f"Invalid negative billed token count detected: {total}")
            return 0
        return total

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics (Anthropic prompt caching).

        Returns:
            Dictionary with cache_read_tokens and cache_creation_tokens
        """
        cache_read = 0
        cache_creation = 0

        for entry in self._entries:
            cache_read += entry.cache_read_tokens
            cache_creation += entry.cache_creation_tokens

        return {"cache_read_tokens": cache_read, "cache_creation_tokens": cache_creation}

    def get_agent_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get per-agent cost and token breakdown.

        Aggregates all entries with operation starting with "agent_" and returns
        detailed breakdown for each agent.

        Returns:
            Dictionary mapping agent name to breakdown:
            {
                "extractor": {
                    "cost": 0.000001,
                    "input_tokens": 227,
                    "output_tokens": 45,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "call_count": 1,
                    "response_time_ms": 1234.56  # Total accumulated time (sum of all calls)
                },
                "orchestrator": {...}
            }

        Note: response_time_ms is the sum of all LLM response times for that agent.
        Divide by call_count to get average response time per call.
        """
        agent_stats: Dict[str, Dict[str, Any]] = {}

        for entry in self._entries:
            # Filter for agent operations (format: "agent_<agent_name>")
            if not entry.operation.startswith("agent_"):
                continue

            # Extract agent name (remove "agent_" prefix)
            agent_name = entry.operation.replace("agent_", "", 1)

            # Initialize agent stats if not exists
            if agent_name not in agent_stats:
                agent_stats[agent_name] = {
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                    "call_count": 0,
                    "response_time_ms": 0.0,
                }

            # Accumulate stats
            stats = agent_stats[agent_name]
            stats["cost"] += entry.cost
            stats["input_tokens"] += entry.input_tokens
            stats["output_tokens"] += entry.output_tokens
            stats["cache_read_tokens"] += entry.cache_read_tokens
            stats["cache_creation_tokens"] += entry.cache_creation_tokens
            stats["call_count"] += 1
            stats["response_time_ms"] += entry.response_time_ms

        # Debug log final breakdown
        for agent_name, stats in agent_stats.items():
            logger.info(
                f"💫 Agent breakdown: {agent_name} - response_time={stats['response_time_ms']:.2f}ms, "
                f"cost=${stats['cost']:.6f}, calls={stats['call_count']}"
            )

        return agent_stats

    def get_session_cost_summary(self) -> str:
        """
        Get brief cost summary for current session with detailed per-message breakdown.

        IMPORTANT: This method has side effects - it updates internal tracking state
        (_last_reported_cost and _last_reported_tokens) to enable per-message deltas.
        Calling this multiple times between messages will show zero for "This message".

        Format:
            💰 This message: $0.0015
              Input (new): 227 tokens
              Output: 200 tokens
              Input (cached): 4,956 tokens (90% discount)

            Session total: $0.0029 (5,660 tokens)

        Returns:
            Multi-line cost summary string
        """
        # Calculate per-message token deltas
        msg_input = self._total_input_tokens - self._last_reported_input_tokens
        msg_output = self._total_output_tokens - self._last_reported_output_tokens
        msg_cache_read = self._total_cache_read_tokens - self._last_reported_cache_read_tokens

        # Calculate per-message cost (prevent negative values)
        message_cost = max(0.0, self.get_total_cost() - self._last_reported_cost)

        # Calculate session totals
        session_cost = self.get_total_cost()
        session_tokens = self.get_total_tokens()

        # Build summary
        lines = []
        lines.append(f"💰 This message: ${message_cost:.4f}")

        # Show token breakdown (only non-zero categories)
        if msg_input > 0:
            lines.append(f"  Input (new): {msg_input:,} tokens")
        if msg_output > 0:
            lines.append(f"  Output: {msg_output:,} tokens")
        if msg_cache_read > 0:
            lines.append(f"  Input (cached): {msg_cache_read:,} tokens (90% discount)")

        # Session total
        lines.append(f"\nSession total: ${session_cost:.4f} ({session_tokens:,} tokens)")

        # Update last reported state for next message
        self._last_reported_cost = session_cost
        self._last_reported_tokens = session_tokens
        self._last_reported_input_tokens = self._total_input_tokens
        self._last_reported_output_tokens = self._total_output_tokens
        self._last_reported_cache_read_tokens = self._total_cache_read_tokens

        return "\n".join(lines)

    def get_summary(self) -> str:
        """
        Get formatted cost summary.

        Returns:
            Multi-line string with cost breakdown
        """
        lines = []
        lines.append("=" * 60)
        lines.append("API COST SUMMARY")
        lines.append("=" * 60)

        # Total tokens and cost
        lines.append(f"Total tokens:  {self.get_total_tokens():,}")
        lines.append(f"  Input:       {self._total_input_tokens:,}")
        lines.append(f"  Output:      {self._total_output_tokens:,}")
        lines.append(f"Total cost:    ${self._total_cost:.4f}")
        lines.append("")

        # Cost by provider
        if self._cost_by_provider:
            lines.append("Cost by provider:")
            for provider, cost in sorted(self._cost_by_provider.items()):
                lines.append(f"  {provider:15s} ${cost:.4f}")
            lines.append("")

        # Cost by operation
        if self._cost_by_operation:
            lines.append("Cost by operation:")
            for operation, cost in sorted(self._cost_by_operation.items()):
                lines.append(f"  {operation:15s} ${cost:.4f}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def reset(self):
        """Reset tracker (for new indexing/conversation session)."""
        self._entries.clear()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._total_cache_read_tokens = 0
        self._total_cache_creation_tokens = 0
        self._last_reported_cost = 0.0
        self._last_reported_tokens = 0
        self._last_reported_input_tokens = 0
        self._last_reported_output_tokens = 0
        self._last_reported_cache_read_tokens = 0
        self._cost_by_provider.clear()
        self._cost_by_operation.clear()

        logger.info("Cost tracker reset")

    def save_to_json(self, output_path: Union[str, Path], document_id: str) -> Path:
        """
        Save cost statistics to JSON file.

        Creates a comprehensive cost report including:
        - Total costs and token counts
        - Breakdown by provider and operation
        - Cache statistics with savings calculation
        - Detailed entry log for audit trail

        Args:
            output_path: Output directory path
            document_id: Document identifier for filename

        Returns:
            Path to the saved JSON file

        Raises:
            StorageError: If directory creation or file writing fails
        """
        if not document_id or not document_id.strip():
            raise StorageError(
                "Invalid document_id for cost statistics",
                details={"document_id": document_id, "output_path": str(output_path)},
            )

        output_path = Path(output_path)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StorageError(
                f"Failed to create output directory: {output_path}",
                details={"output_path": str(output_path), "document_id": document_id},
                cause=e,
            ) from e

        # Calculate cache savings
        cache_stats = self.get_cache_stats()
        cache_savings_usd = 0.0

        # Cache savings = what would have cost at full price minus discounted price
        # Cache reads are billed at 10%, so savings = 90% of what full price would be
        for entry in self._entries:
            if entry.cache_read_tokens > 0 and entry.provider in PRICING:
                model_pricing = PRICING[entry.provider].get(entry.model, {})
                if model_pricing:
                    # Full price - discounted price = savings
                    full_price = (entry.cache_read_tokens / 1_000_000) * model_pricing.get("input", 0)
                    discounted = full_price * 0.1
                    cache_savings_usd += full_price - discounted

        # Build comprehensive stats
        stats: Dict[str, Any] = {
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_cost_usd": round(self._total_cost, 6),
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_api_calls": len(self._entries),
            },
            "cost_by_provider": {k: round(v, 6) for k, v in self._cost_by_provider.items()},
            "cost_by_operation": {k: round(v, 6) for k, v in self._cost_by_operation.items()},
            "cache_stats": {
                "cache_read_tokens": cache_stats["cache_read_tokens"],
                "cache_creation_tokens": cache_stats["cache_creation_tokens"],
                "cache_savings_usd": round(cache_savings_usd, 6),
            },
            "entries": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "provider": e.provider,
                    "model": e.model,
                    "operation": e.operation,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cache_read_tokens": e.cache_read_tokens,
                    "cache_creation_tokens": e.cache_creation_tokens,
                    "cost_usd": round(e.cost, 8),
                    "response_time_ms": round(e.response_time_ms, 2),
                }
                for e in self._entries
            ],
        }

        # Save to file
        output_file = output_path / f"{document_id}_costs.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        except (OSError, IOError) as e:
            raise StorageError(
                f"Failed to write cost statistics to {output_file}",
                details={
                    "output_file": str(output_file),
                    "document_id": document_id,
                    "total_cost": self._total_cost,
                },
                cause=e,
            ) from e

        logger.info(
            f"Cost statistics saved to: {output_file} "
            f"(${self._total_cost:.4f}, {len(self._entries)} API calls)"
        )

        return output_file


# Global instance for easy access
_global_tracker: Optional[CostTracker] = None


def get_global_tracker() -> CostTracker:
    """Get or create global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_global_tracker():
    """Reset global cost tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()


# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = CostTracker()

    # Simulate indexing pipeline
    print("Simulating indexing pipeline...")

    # PHASE 2: Summaries (Claude Haiku)
    tracker.track_llm("anthropic", "haiku", 5000, 750, "summary")
    tracker.track_llm("anthropic", "haiku", 3000, 500, "summary")

    # PHASE 3: Contextual retrieval (Claude Haiku)
    tracker.track_llm("anthropic", "haiku", 10000, 1500, "context")

    # PHASE 4: Embeddings (BGE-M3 local - free)
    tracker.track_embedding("huggingface", "bge-m3", 50000, "indexing")

    # Print summary
    print(tracker.get_summary())

    # Simulate RAG agent conversation
    print("\nSimulating RAG agent conversation...")
    tracker.reset()

    # Agent queries (Claude Sonnet)
    tracker.track_llm("anthropic", "sonnet", 2000, 500, "agent")
    tracker.track_llm("anthropic", "sonnet", 1500, 300, "agent")
    tracker.track_llm("anthropic", "sonnet", 3000, 800, "agent")

    # Query embeddings (text-embedding-3-large)
    tracker.track_embedding("openai", "text-embedding-3-large", 500, "query")
    tracker.track_embedding("openai", "text-embedding-3-large", 300, "query")

    # Print summary
    print(tracker.get_summary())
