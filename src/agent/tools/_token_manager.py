"""
Smart Token Management for Tool Outputs

Dynamically manages token budgets to prevent overflow while maximizing information.
Uses actual token counting (tiktoken) instead of character limits.

Key features:
- Real token counting (not char-based)
- Smart truncation (sentence boundaries, not mid-word)
- Dynamic budgets (adapts to available context)
- Progressive disclosure (summary/medium/full detail levels)
- Auto-summarization fallback (when data exceeds budget)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Token counting
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.warning("tiktoken not installed, falling back to char-based estimation")
    TIKTOKEN_AVAILABLE = False


class DetailLevel(Enum):
    """Progressive detail levels for outputs."""

    SUMMARY = "summary"  # ~100 tokens per item
    MEDIUM = "medium"  # ~300 tokens per item
    FULL = "full"  # ~600 tokens per item


@dataclass
class TokenBudget:
    """
    Token budget configuration for tool outputs.

    Default values are calibrated for Claude's context window:
    - max_total_tokens=8000: ~4% of 200K context, leaves room for conversation history
    - max_tokens_per_chunk=600: Allows 2-3 paragraphs of detailed content
    - max_tokens_per_section=400: Sufficient for section metadata + summary
    - reserved_tokens=1000: Safety buffer for JSON structure, citations, metadata
    """

    max_total_tokens: int = 8000  # Max tokens for entire tool output
    max_tokens_per_chunk: int = 600  # Max per chunk (FULL detail)
    max_tokens_per_section: int = 400  # Max per section metadata
    reserved_tokens: int = 1000  # Reserved for metadata, citations, etc.

    def __post_init__(self):
        """Validate configuration consistency."""
        if self.max_total_tokens <= 0:
            raise ValueError(f"max_total_tokens must be positive, got {self.max_total_tokens}")
        if self.max_tokens_per_chunk <= 0:
            raise ValueError(
                f"max_tokens_per_chunk must be positive, got {self.max_tokens_per_chunk}"
            )
        if self.max_tokens_per_section <= 0:
            raise ValueError(
                f"max_tokens_per_section must be positive, got {self.max_tokens_per_section}"
            )
        if self.reserved_tokens < 0:
            raise ValueError(f"reserved_tokens must be non-negative, got {self.reserved_tokens}")
        if self.reserved_tokens >= self.max_total_tokens:
            raise ValueError(
                f"reserved_tokens ({self.reserved_tokens}) must be less than "
                f"max_total_tokens ({self.max_total_tokens})"
            )
        if self.max_tokens_per_chunk > self.get_content_budget():
            raise ValueError(
                f"max_tokens_per_chunk ({self.max_tokens_per_chunk}) exceeds available "
                f"content budget ({self.get_content_budget()})"
            )

    def get_content_budget(self) -> int:
        """Get available tokens for actual content (guaranteed non-negative after validation)."""
        return max(0, self.max_total_tokens - self.reserved_tokens)

    def tokens_per_item(self, detail_level: DetailLevel) -> int:
        """Get token limit per item based on detail level."""
        if detail_level == DetailLevel.SUMMARY:
            return 100
        elif detail_level == DetailLevel.MEDIUM:
            return 300
        else:  # FULL
            return self.max_tokens_per_chunk


class TokenCounter:
    """
    Accurate token counting using tiktoken.

    Falls back to character-based estimation if tiktoken unavailable.
    """

    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        """
        Initialize token counter.

        Args:
            model: Model name for tokenizer. Uses cl100k_base encoding (GPT-4 tokenizer)
                   which provides approximate token counts for Claude models.
                   For exact Claude tokens, use Anthropic's tokenizer API.
        """
        self.model = model

        if TIKTOKEN_AVAILABLE:
            # Use cl100k_base (GPT-4 tokenizer) for approximate Claude token counts
            self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if not text:
            return 0

        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: estimate 4 chars per token (rough average for English)
            return len(text) // 4

    def estimate_tokens(self, obj: Any) -> int:
        """
        Estimate tokens for any object (dict, list, str).

        Args:
            obj: Object to estimate

        Returns:
            Estimated token count
        """
        if isinstance(obj, str):
            return self.count_tokens(obj)
        elif isinstance(obj, dict):
            # Convert to JSON string and count
            import json

            return self.count_tokens(json.dumps(obj))
        elif isinstance(obj, list):
            return sum(self.estimate_tokens(item) for item in obj)
        else:
            return self.count_tokens(str(obj))


class SmartTruncator:
    """
    Intelligently truncates text at sentence boundaries.

    Avoids mid-sentence cuts and preserves readability.
    """

    # Sentence boundary patterns (Czech + English)
    SENTENCE_ENDINGS = re.compile(r"([.!?…][\s\n]+|[.!?…]$)")

    @staticmethod
    def truncate_at_sentence(
        text: str, max_tokens: int, token_counter: TokenCounter
    ) -> Tuple[str, bool]:
        """
        Truncate text at sentence boundary within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            token_counter: TokenCounter instance

        Returns:
            (truncated_text, was_truncated)
        """
        if not text:
            return "", False

        current_tokens = token_counter.count_tokens(text)

        if current_tokens <= max_tokens:
            return text, False

        # Find sentence boundaries
        sentences = SmartTruncator.SENTENCE_ENDINGS.split(text)

        # Reconstruct text sentence by sentence until we hit limit
        result = ""
        for i in range(0, len(sentences), 2):  # Step by 2 (sentence + separator)
            sentence = sentences[i]
            separator = sentences[i + 1] if i + 1 < len(sentences) else ""

            candidate = result + sentence + separator
            candidate_tokens = token_counter.count_tokens(candidate)

            if candidate_tokens > max_tokens:
                # Current sentence would exceed limit
                if not result:
                    # First sentence is too long, truncate mid-sentence as fallback
                    return (
                        SmartTruncator._truncate_to_token_limit(text, max_tokens, token_counter),
                        True,
                    )
                else:
                    # Return accumulated sentences
                    return result.strip(), True

            result = candidate

        # All sentences fit
        return result.strip(), False

    @staticmethod
    def _truncate_to_token_limit(text: str, max_tokens: int, token_counter: TokenCounter) -> str:
        """
        Fallback: Binary search to find truncation point.

        Used when even first sentence exceeds limit.
        """
        if not TIKTOKEN_AVAILABLE:
            # Char-based estimate: 4 chars per token
            char_limit = max_tokens * 4
            return text[:char_limit] + "..."

        # Binary search for exact token cutoff
        left, right = 0, len(text)
        result = text

        while left < right:
            mid = (left + right + 1) // 2
            candidate = text[:mid]
            tokens = token_counter.count_tokens(candidate)

            if tokens <= max_tokens:
                result = candidate
                left = mid
            else:
                right = mid - 1

        return result + "..."

    @staticmethod
    def truncate_at_word(
        text: str, max_tokens: int, token_counter: TokenCounter
    ) -> Tuple[str, bool]:
        """
        Truncate at word boundary (fallback if sentence truncation too aggressive).

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens
            token_counter: TokenCounter instance

        Returns:
            (truncated_text, was_truncated)
        """
        if not text:
            return "", False

        current_tokens = token_counter.count_tokens(text)

        if current_tokens <= max_tokens:
            return text, False

        words = text.split()
        result = ""

        for word in words:
            candidate = result + " " + word if result else word
            tokens = token_counter.count_tokens(candidate)

            if tokens > max_tokens:
                return result.strip() + "...", True

            result = candidate

        return result.strip(), False


class AdaptiveFormatter:
    """
    Adaptively formats tool outputs based on available token budget.

    Automatically switches between detail levels or applies summarization.
    """

    def __init__(self, budget: TokenBudget = None, token_counter: TokenCounter = None):
        """
        Initialize adaptive formatter.

        Args:
            budget: Token budget configuration
            token_counter: Token counter instance
        """
        self.budget = budget or TokenBudget()
        self.token_counter = token_counter or TokenCounter()

    def format_chunks(
        self,
        chunks: List[Dict[str, Any]],
        detail_level: DetailLevel = DetailLevel.MEDIUM,
        include_score: bool = True,
        auto_adjust: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Format chunks with adaptive detail level.

        Args:
            chunks: List of chunk dicts
            detail_level: Desired detail level
            include_score: Include scores in output
            auto_adjust: Automatically reduce detail if over budget

        Returns:
            (formatted_chunks, metadata)
            metadata includes: actual_detail_level, total_tokens, truncated_count
        """
        if not chunks:
            return [], {"actual_detail_level": detail_level.value, "total_tokens": 0}

        tokens_per_chunk = self.budget.tokens_per_item(detail_level)
        content_budget = self.budget.get_content_budget()

        # Check if we need to reduce detail level
        estimated_tokens = len(chunks) * tokens_per_chunk
        actual_detail_level = detail_level

        chunks_capped = False
        original_chunk_count = len(chunks)

        if auto_adjust and estimated_tokens > content_budget:
            # Try reducing detail level
            if detail_level == DetailLevel.FULL:
                actual_detail_level = DetailLevel.MEDIUM
                tokens_per_chunk = self.budget.tokens_per_item(DetailLevel.MEDIUM)
            elif detail_level == DetailLevel.MEDIUM:
                actual_detail_level = DetailLevel.SUMMARY
                tokens_per_chunk = self.budget.tokens_per_item(DetailLevel.SUMMARY)

            logger.info(
                f"Auto-adjusting detail level: {detail_level.value} → {actual_detail_level.value} "
                f"(estimated {estimated_tokens} tokens exceeds budget {content_budget})"
            )

            # After detail reduction, check if we still exceed budget
            estimated_tokens_after_adjustment = len(chunks) * tokens_per_chunk
            if estimated_tokens_after_adjustment > content_budget:
                # Cap chunk count to fit within budget
                max_chunks = max(3, content_budget // tokens_per_chunk)  # At least 3 chunks
                if len(chunks) > max_chunks:
                    logger.info(
                        f"Capping chunk count: {len(chunks)} → {max_chunks} "
                        f"(budget {content_budget} tokens / {tokens_per_chunk} per chunk)"
                    )
                    chunks = chunks[:max_chunks]
                    chunks_capped = True

        # Format each chunk
        formatted = []
        total_tokens = 0
        truncated_count = 0

        for chunk in chunks:
            content = chunk.get("content", chunk.get("raw_content", ""))

            # Smart truncation
            truncated_content, was_truncated = SmartTruncator.truncate_at_sentence(
                content, tokens_per_chunk, self.token_counter
            )

            if was_truncated:
                truncated_count += 1

            result = {
                "content": truncated_content,
                "document_id": chunk.get("document_id", "unknown"),
                "section_title": chunk.get("section_title", ""),
                "chunk_id": chunk.get("chunk_id", ""),
            }

            if include_score:
                score = (
                    chunk.get("rerank_score")
                    or chunk.get("boosted_score")
                    or chunk.get("rrf_score")
                    or chunk.get("score")
                    or 0.0
                )
                result["score"] = round(float(score), 4)

            if "page_number" in chunk:
                result["page"] = chunk["page_number"]

            # Add truncation indicator if needed
            if was_truncated:
                result["truncated"] = True

            formatted.append(result)
            total_tokens += self.token_counter.estimate_tokens(result)

        metadata = {
            "requested_detail_level": detail_level.value,
            "actual_detail_level": actual_detail_level.value,
            "total_tokens": total_tokens,
            "chunks_count": len(formatted),
            "original_chunk_count": original_chunk_count,
            "truncated_count": truncated_count,
            "auto_adjusted": actual_detail_level != detail_level,
            "chunks_capped": chunks_capped,
        }

        return formatted, metadata

    def adaptive_k(
        self, requested_k: int, tokens_per_item: int = None, min_k: int = 3, max_k: int = 50
    ) -> Tuple[int, str]:
        """
        Adaptively determine k based on token budget.

        Args:
            requested_k: User-requested k
            tokens_per_item: Estimated tokens per result (default: MEDIUM level)
            min_k: Minimum results to return
            max_k: Maximum results (safety limit)

        Returns:
            (actual_k, reason)
        """
        if tokens_per_item is None:
            tokens_per_item = self.budget.tokens_per_item(DetailLevel.MEDIUM)

        content_budget = self.budget.get_content_budget()
        budget_based_k = content_budget // tokens_per_item

        # Clamp to reasonable range
        budget_based_k = max(min_k, min(budget_based_k, max_k))

        if requested_k <= budget_based_k:
            return requested_k, "within_budget"
        else:
            logger.info(
                f"Reducing k from {requested_k} to {budget_based_k} "
                f"(token budget: {content_budget}, est. tokens/item: {tokens_per_item})"
            )
            return budget_based_k, "budget_limited"

    def format_sections_with_budget(
        self, sections: List[Dict[str, Any]], include_summary: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Format sections list with dynamic truncation.

        Args:
            sections: List of section dicts
            include_summary: Include section summaries

        Returns:
            (formatted_sections, metadata)
        """
        if not sections:
            return [], {"total_tokens": 0, "truncated": False}

        # Calculate budget per section
        content_budget = self.budget.get_content_budget()
        tokens_per_section = self.budget.max_tokens_per_section

        max_sections = content_budget // tokens_per_section
        max_sections = max(10, min(max_sections, 100))  # Clamp 10-100

        truncated = len(sections) > max_sections
        actual_sections = sections[:max_sections]

        formatted = []
        total_tokens = 0

        for section in actual_sections:
            result = {
                "section_id": section.get("section_id", ""),
                "section_title": section.get("section_title", ""),
                "chunk_count": section.get("chunk_count", 0),
            }

            if include_summary and "section_summary" in section:
                summary = section["section_summary"]
                # Truncate summary if needed (max 200 tokens)
                truncated_summary, _ = SmartTruncator.truncate_at_sentence(
                    summary, 200, self.token_counter
                )
                result["summary"] = truncated_summary

            formatted.append(result)
            total_tokens += self.token_counter.estimate_tokens(result)

        metadata = {
            "total_sections": len(sections),
            "returned_sections": len(formatted),
            "truncated": truncated,
            "total_tokens": total_tokens,
            "max_sections_allowed": max_sections,
        }

        return formatted, metadata


# Global instances (lazy-initialized)
_default_token_counter: Optional[TokenCounter] = None
_default_formatter: Optional[AdaptiveFormatter] = None


def get_token_counter() -> TokenCounter:
    """Get or create default token counter."""
    global _default_token_counter
    if _default_token_counter is None:
        _default_token_counter = TokenCounter()
    return _default_token_counter


def get_adaptive_formatter(budget: TokenBudget = None) -> AdaptiveFormatter:
    """Get or create default adaptive formatter."""
    global _default_formatter
    if _default_formatter is None or budget is not None:
        _default_formatter = AdaptiveFormatter(budget=budget)
    return _default_formatter
