"""
Query Expansion Module

Expands user queries into multiple variations for better recall.
Based on IMPROVEMENTS.md Priority 1.1 - Query Understanding & Expansion.

Research shows +15-25% recall improvement with multi-query generation.
"""

import logging
import re
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """Result from query expansion."""

    original_query: str
    expanded_queries: List[str]  # Includes original
    num_expansions: int
    expansion_method: str  # "llm" or "none"
    model_used: Optional[str] = None
    cost_estimate: float = 0.0


class QueryExpander:
    """
    Expand user queries using LLM for better retrieval recall.

    Uses multi-question generation strategy:
    - Generate N related questions with different phrasings
    - Keep same semantic intent but vary vocabulary
    - Enable finding relevant docs with different terminology

    Features:
    - Configurable LLM provider (OpenAI/Anthropic)
    - Graceful fallback on errors (returns original query)
    - Warning at num_expansions > 5 (performance impact)
    - Cost tracking via global tracker
    - Skip expansion when num_expansions == 1 (optimization)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialize query expander.

        Args:
            provider: "openai" or "anthropic"
            model: LLM model name (default: gpt-4o-mini for stability/cost per CLAUDE.md)
            anthropic_api_key: Anthropic API key (if provider="anthropic")
            openai_api_key: OpenAI API key (if provider="openai")
        """
        self.provider = provider.lower()
        self.model = model

        # Initialize LLM client
        if self.provider == "openai":
            if not openai_api_key:
                raise ValueError("openai_api_key required when provider='openai'")
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=openai_api_key)
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )

        elif self.provider == "anthropic":
            if not anthropic_api_key:
                raise ValueError("anthropic_api_key required when provider='anthropic'")
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=anthropic_api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )

        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'")

        logger.info(f"QueryExpander initialized: provider={self.provider}, model={self.model}")

    def expand(
        self, query: str, num_expansions: int = 3, warn_threshold: int = 5
    ) -> ExpansionResult:
        """
        Expand query into multiple variations.

        Args:
            query: Original user query
            num_expansions: Number of ADDITIONAL expansions to generate (default: 3)
                           Final queries = [original] + num_expansions variations
            warn_threshold: Warn if num_expansions exceeds this (default: 5)

        Returns:
            ExpansionResult with original + expanded queries (num_expansions + 1 total)

        Note:
            - num_expansions=0: Returns [original_query] without LLM call
            - num_expansions=1: Generates 1 variation + original = 2 queries total
            - num_expansions=3: Generates 3 variations + original = 4 queries total
        """
        # Optimization: Skip expansion if num_expansions == 0
        if num_expansions == 0:
            logger.debug(f"Skipping expansion (num_expansions=0): {query}")
            return ExpansionResult(
                original_query=query,
                expanded_queries=[query],
                num_expansions=1,  # 1 query total (original only)
                expansion_method="none",
            )

        # Warning for performance impact
        if num_expansions > warn_threshold:
            logger.warning(
                f"High expansion count ({num_expansions}) may impact latency. "
                f"Consider using num_expansions <= {warn_threshold} for better performance."
            )

        # Generate expansions via LLM
        try:
            expanded = self._generate_expansions_llm(query, num_expansions)

            logger.info(
                f"Generated {len(expanded)} expansions for query '{query}': {expanded}"
            )

            return ExpansionResult(
                original_query=query,
                expanded_queries=[query] + expanded,
                num_expansions=len(expanded) + 1,  # Include original
                expansion_method="llm",
                model_used=self.model,
            )

        except Exception as e:
            logger.warning(
                f"Query expansion failed ({type(e).__name__}: {e}), "
                f"using original query only"
            )

            # Graceful fallback
            return ExpansionResult(
                original_query=query,
                expanded_queries=[query],
                num_expansions=1,
                expansion_method="fallback",
            )

    def _generate_expansions_llm(self, query: str, num_expansions: int) -> List[str]:
        """
        Generate query variations using LLM.

        Strategy: Multi-question generation
        - Generate N related questions
        - Use synonyms and related terms
        - Rephrase with different structures
        - Keep same semantic intent

        Returns:
            List of expanded queries (without original)
        """
        # Construct prompt
        prompt = self._build_expansion_prompt(query, num_expansions)

        # Call LLM based on provider
        if self.provider == "openai":
            # GPT-5 models (gpt-5-*, o-series) have special requirements:
            # - Use max_completion_tokens instead of max_tokens
            # - Don't support custom temperature (only default 1.0)
            params = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}

            if self.model.startswith(("gpt-5", "o1-", "o3-")):
                # GPT-5/o-series models
                params["max_completion_tokens"] = 300
                # Don't set temperature (only default 1.0 is supported)
            else:
                # GPT-4 and earlier models
                params["max_tokens"] = 300
                params["temperature"] = 0.7  # Higher temperature for diversity

            response = self.client.chat.completions.create(**params)

            expanded_text = response.choices[0].message.content

            # Debug: Log full response if content is empty or None
            if not expanded_text:
                logger.error(
                    f"GPT-5 returned empty content. Full response: {response.model_dump()}"
                )
                expanded_text = ""  # Ensure it's a string

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        else:  # anthropic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )

            expanded_text = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        # Track cost
        self._track_cost(input_tokens, output_tokens)

        # Parse response (one query per line)
        expansions = [line.strip() for line in expanded_text.split("\n") if line.strip()]

        # Filter out numbering (e.g., "1. query" → "query")
        expansions = [self._strip_numbering(exp) for exp in expansions]

        # Filter out empty strings and original query (if LLM repeated it)
        expansions = [
            exp for exp in expansions
            if exp and exp.lower() != query.lower()
        ]

        # Log warning if we got fewer expansions than requested
        if len(expansions) < num_expansions:
            logger.warning(
                f"Generated {len(expansions)}/{num_expansions} expansions. "
                f"LLM response may have been incomplete. Raw response: {expanded_text[:200]}"
            )

        # Return up to num_expansions (may be fewer if LLM didn't generate enough)
        return expansions[:num_expansions]

    def _build_expansion_prompt(self, query: str, num_expansions: int) -> str:
        """Build LLM prompt for query expansion."""
        return f"""Given this search query: "{query}"

Generate {num_expansions} related questions that capture the same intent using different wording:
- Use synonyms and related terminology
- Rephrase with different sentence structures
- Keep the same semantic meaning
- Vary specificity levels (general/specific)

Return ONLY the {num_expansions} variations, one per line, without numbering or explanation."""

    def _strip_numbering(self, text: str) -> str:
        """Remove numbering from LLM output (e.g., '1. query' → 'query')."""
        # Remove leading numbering: "1.", "2)", "3 -", etc.
        text = re.sub(r"^\s*\d+[\.):\-\s]+", "", text)
        return text.strip()

    def _track_cost(self, input_tokens: int, output_tokens: int):
        """Track expansion cost via global cost tracker."""
        try:
            from src.cost_tracker import get_global_tracker

            tracker = get_global_tracker()
            tracker.track_llm(self.provider, self.model, input_tokens, output_tokens)

        except Exception as e:
            logger.debug(f"Cost tracking failed: {e}")
            # Non-critical, continue

    def get_stats(self) -> dict:
        """Get expander statistics (for debugging)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "initialized": hasattr(self, "client"),
        }
