"""
Context Enricher - Merge user clarification into query context.

Takes original query + user response and creates enriched query
for better retrieval on re-run.
"""

import logging
from typing import Dict, Any

from .config import HITLConfig

logger = logging.getLogger(__name__)


class ContextEnricher:
    """
    Enrich query context with user clarification.

    Strategies:
    - append_with_context: Add user response as additional context
    - replace: Replace original query with user response
    - merge: Intelligent merge using LLM (future)
    """

    def __init__(self, config: HITLConfig):
        """
        Initialize context enricher.

        Args:
            config: HITL configuration with enrichment strategy
        """
        self.config = config
        logger.info(f"ContextEnricher initialized with strategy={config.enrichment_strategy}")

    def enrich(
        self,
        original_query: str,
        user_response: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich query context with user clarification.

        Args:
            original_query: Original user query
            user_response: User's clarification answer
            state: Current workflow state

        Returns:
            Updated state with enriched query
        """
        # Validate inputs
        if not user_response or not user_response.strip():
            logger.warning("Empty user response, using original query unchanged")
            return state

        # Apply enrichment strategy
        strategy = self.config.enrichment_strategy

        if strategy == "append_with_context":
            enriched_query = self._append_with_context(original_query, user_response)
        elif strategy == "replace":
            enriched_query = self._replace(original_query, user_response)
        elif strategy == "merge":
            # Future: LLM-based intelligent merge
            logger.warning("Merge strategy not yet implemented, falling back to append_with_context")
            enriched_query = self._append_with_context(original_query, user_response)
        else:
            logger.error(f"Unknown enrichment strategy: {strategy}, using append_with_context")
            enriched_query = self._append_with_context(original_query, user_response)

        # Enforce max length
        if len(enriched_query) > self.config.max_enriched_length:
            logger.warning(
                f"Enriched query exceeds max length ({len(enriched_query)} > {self.config.max_enriched_length}), truncating"
            )
            enriched_query = enriched_query[: self.config.max_enriched_length]

        # Update state
        state["original_query"] = original_query
        state["user_clarification"] = user_response
        state["enriched_query"] = enriched_query
        state["query"] = enriched_query  # Use enriched query for re-run

        # Store in shared_context for downstream agents
        if "shared_context" not in state:
            state["shared_context"] = {}

        state["shared_context"]["hitl_clarification"] = {
            "original_query": original_query,
            "user_response": user_response,
            "enriched_query": enriched_query,
            "enrichment_strategy": strategy
        }

        logger.info(
            f"Query enriched: original_len={len(original_query)}, "
            f"enriched_len={len(enriched_query)}, strategy={strategy}"
        )

        return state

    def _append_with_context(self, original_query: str, user_response: str) -> str:
        """
        Append user response as additional context.

        Uses template from config.
        """
        template = self.config.enrichment_template
        enriched = template.format(
            original_query=original_query,
            user_response=user_response
        )
        return enriched

    def _replace(self, original_query: str, user_response: str) -> str:
        """
        Replace original query with user response.

        Use when user provides a complete reformulation.
        """
        return user_response

    def clear_clarification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clear clarification fields from state.

        Use after successful enrichment or when skipping clarification.
        """
        fields_to_clear = [
            "quality_check_required",
            "quality_issues",
            "clarifying_questions",
            "user_clarification",
            "enriched_query"
        ]

        for field in fields_to_clear:
            if field in state:
                del state[field]

        logger.debug("Cleared clarification fields from state")
        return state
