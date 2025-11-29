"""
Shared Tool Utilities

Helper functions used across multiple tools.

IMPORTANT: This module now uses smart token management (token_manager.py) instead of
hardcoded character limits. Old functions preserved for backward compatibility.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import smart token management (lazy import to avoid circular deps)
_token_manager_import_attempted = False
_token_manager_imported = False
_DetailLevel = None
_get_adaptive_formatter = None
_get_token_counter = None
_SmartTruncator = None


def _ensure_token_manager_imported():
    """Lazy import token management to avoid circular dependencies."""
    global _token_manager_import_attempted, _token_manager_imported
    global _DetailLevel, _get_adaptive_formatter, _get_token_counter, _SmartTruncator

    # Only attempt import once
    if _token_manager_import_attempted:
        return

    _token_manager_import_attempted = True

    try:
        from .token_manager import (
            DetailLevel,
            get_adaptive_formatter,
            get_token_counter,
            SmartTruncator,
        )

        _DetailLevel = DetailLevel
        _get_adaptive_formatter = get_adaptive_formatter
        _get_token_counter = get_token_counter
        _SmartTruncator = SmartTruncator
        _token_manager_imported = True
    except ImportError:
        # token_manager is optional - fall back to legacy character-based limits
        # Log only once at debug level (not warning) to avoid log spam
        logger.debug("Token manager not available - using legacy character-based truncation")
        _token_manager_imported = False


def format_chunk_result(
    chunk: Dict[str, Any],
    include_score: bool = True,
    max_content_length: int = None,
    smart_truncate: bool = True,
    detail_level: str = "medium",
) -> Dict[str, Any]:
    """
    Format a chunk result for tool output.

    NEW: Now uses smart token-aware truncation by default (truncates at sentence boundaries).
    Falls back to character-based truncation if token_manager unavailable.

    Args:
        chunk: Chunk dict from retrieval
        include_score: Whether to include score
        max_content_length: Maximum content length (DEPRECATED: use detail_level instead).
                          If None, uses smart token-based limits.
                          If specified, overrides smart truncation.
        smart_truncate: Use token-aware sentence-boundary truncation (default: True)
        detail_level: Detail level ("summary", "medium", "full"). Ignored if max_content_length set.

    Returns:
        Formatted dict with 'content', 'document_id', 'section_title', 'chunk_id',
        optionally 'score', 'page', 'truncated'
    """
    content = chunk.get("content") or chunk.get("raw_content") or ""

    # Ensure content is always a string (handles None values)
    if not isinstance(content, str):
        if content is not None:
            # Log warning when type coercion occurs to detect data quality issues
            logger.warning(
                f"Content field has unexpected type {type(content).__name__} for chunk "
                f"{chunk.get('chunk_id', 'unknown')}. Converting to string. "
                f"This may indicate a data structure issue in the indexing pipeline. "
                f"Content preview: {str(content)[:100]}"
            )
            content = str(content)
        else:
            content = ""

    # Determine truncation strategy
    if max_content_length is not None:
        # Legacy mode: character-based truncation (backward compatibility)
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"
            was_truncated = True
        else:
            was_truncated = False

    elif smart_truncate:
        # NEW: Smart token-aware truncation
        _ensure_token_manager_imported()

        if _token_manager_imported:
            # Map detail_level string to enum
            detail_map = {
                "summary": _DetailLevel.SUMMARY,  # ~100 tokens
                "medium": _DetailLevel.MEDIUM,  # ~300 tokens
                "full": _DetailLevel.FULL,  # ~600 tokens
            }
            level = detail_map.get(detail_level.lower(), _DetailLevel.MEDIUM)

            formatter = _get_adaptive_formatter()
            token_counter = _get_token_counter()
            max_tokens = formatter.budget.tokens_per_item(level)

            content, was_truncated = _SmartTruncator.truncate_at_sentence(
                content, max_tokens, token_counter
            )
        else:
            # Fallback: character-based with estimate
            char_limits = {"summary": 400, "medium": 1200, "full": 2400}
            char_limit = char_limits.get(detail_level.lower(), 1200)

            if len(content) > char_limit:
                content = content[:char_limit] + "... [truncated]"
                was_truncated = True
            else:
                was_truncated = False
    else:
        # No truncation
        was_truncated = False

    # Build result
    section_title = chunk.get("section_title", "")

    result = {
        "content": content,
        "document_id": chunk.get("document_id", "unknown"),
        "section_title": section_title,
        "chunk_id": chunk.get("chunk_id", ""),
    }

    # Add hierarchical breadcrumb path for better navigation
    hierarchical_path = chunk.get("hierarchical_path")
    if hierarchical_path and isinstance(hierarchical_path, str):
        # Clean up "Untitled" from breadcrumb
        if " > Untitled" in hierarchical_path:
            # Replace "doc_id > Untitled" with just "doc_id" or add page info
            page_num = chunk.get("page_number")
            if page_num:
                result["breadcrumb"] = hierarchical_path.replace(" > Untitled", f" (page {page_num})")
            else:
                result["breadcrumb"] = hierarchical_path.replace(" > Untitled", "")
        else:
            result["breadcrumb"] = hierarchical_path

        # Extract section_title from breadcrumb if empty
        # Add type safety to prevent AttributeError on non-string section_title
        if not section_title or (not isinstance(section_title, str)) or (not section_title.strip()):
            # Format: "doc_id > section1 > section2 > ..."
            # Extract last part as section_title
            parts = hierarchical_path.split(" > ")
            if len(parts) > 1:
                last_part = parts[-1].strip()
                # Don't use "Untitled" as section_title
                if last_part and last_part != "Untitled":
                    result["section_title"] = last_part
                else:
                    # Use page number as fallback
                    page_num = chunk.get("page_number")
                    if page_num:
                        result["section_title"] = f"Page {page_num}"
    else:
        # Fallback: construct from document_id + section_path or section_title
        doc_id = chunk.get("document_id", "unknown")
        section_path = chunk.get("section_path")

        if section_path and isinstance(section_path, str) and section_path != "Untitled":
            result["breadcrumb"] = f"{doc_id} > {section_path}"
            # Extract section_title from section_path if empty
            if not section_title or not section_title.strip():
                result["section_title"] = section_path.strip()
        elif section_title and isinstance(section_title, str):
            result["breadcrumb"] = f"{doc_id} > {section_title}"
        else:
            # Ultimate fallback: use page number
            page_num = chunk.get("page_number")
            if page_num:
                result["breadcrumb"] = f"{doc_id} (page {page_num})"
                if not section_title or not section_title.strip():
                    result["section_title"] = f"Page {page_num}"

    if include_score:
        # Prefer rerank_score, fall back to boosted_score, then rrf_score, then score
        score = (
            chunk.get("rerank_score")
            or chunk.get("boosted_score")
            or chunk.get("rrf_score")
            or chunk.get("score")
            or 0.0
        )
        result["score"] = round(float(score), 4)

    # Add page number if available
    if "page_number" in chunk:
        result["page"] = chunk["page_number"]

    # NOTE: Removed debug fields (truncated, dense_score, bm25_score) to reduce token usage
    # These were only used for training data collection, not for agent reasoning
    # Token savings: ~5% per chunk (approx 15-20 tokens per chunk)

    return result


def generate_citation(chunk: Dict[str, Any], chunk_number: int, format: str = "inline") -> str:
    """
    Generate citation string for a chunk with hierarchical breadcrumb path.

    Args:
        chunk: Chunk dict
        chunk_number: Citation number (1-indexed)
        format: "inline", "detailed", or "footnote"

    Returns:
        Citation string with breadcrumb path for better navigation
    """
    # Prefer hierarchical_path (full breadcrumb), fallback to document_id
    breadcrumb = chunk.get("hierarchical_path") or chunk.get("breadcrumb")

    if not breadcrumb:
        # Fallback: construct breadcrumb from available metadata
        doc_name = chunk.get("document_name") or chunk.get("document_id", "unknown")
        section_path = chunk.get("section_path")
        section_title = chunk.get("section_title", "")

        if section_path:
            breadcrumb = f"{doc_name} > {section_path}"
        elif section_title:
            breadcrumb = f"{doc_name} > {section_title}"
        else:
            breadcrumb = doc_name

    # Clean up "Untitled" from breadcrumb (same logic as format_chunk_result)
    if breadcrumb and " > Untitled" in breadcrumb:
        page_num = chunk.get("page_number")
        if page_num:
            breadcrumb = breadcrumb.replace(" > Untitled", f" (page {page_num})")
        else:
            breadcrumb = breadcrumb.replace(" > Untitled", "")

    page = chunk.get("page_number")

    if format == "inline":
        return f"[{chunk_number}] {breadcrumb}"

    elif format == "detailed":
        citation = f"[{chunk_number}] {breadcrumb}"
        if page:
            citation += f" (Page {page})"
        return citation

    elif format == "footnote":
        citation = f"[{chunk_number}] {breadcrumb}"
        if page:
            citation += f", p. {page}"
        return citation

    else:
        return f"[{chunk_number}] {breadcrumb}"


def create_error_result(error_message: str, tool_name: str = None) -> Dict[str, Any]:
    """
    Create standardized error result.

    Args:
        error_message: Error description
        tool_name: Name of tool that failed

    Returns:
        Error dict
    """
    return {
        "error": error_message,
        "tool": tool_name,
        "success": False,
    }


def validate_k_parameter(
    k: int, max_k: int = None, adaptive: bool = True, detail_level: str = "medium"
) -> Tuple[int, Optional[str]]:
    """
    Validate and adaptively adjust k parameter based on token budget.

    NEW: Now adaptively calculates max_k based on available token budget.
    Falls back to hardcoded limits if token_manager unavailable.

    Args:
        k: Number of results requested
        max_k: Maximum allowed value (optional). If None, calculates from token budget.
        adaptive: Use adaptive k based on token budget (default: True)
        detail_level: Detail level for results ("summary", "medium", "full")

    Returns:
        Tuple of (validated_k, adjustment_reason)
        - validated_k: Adjusted k value
        - adjustment_reason: Why k was adjusted (None if not adjusted)
    """
    if k < 1:
        logger.warning(f"k={k} is too small, using k=1")
        return 1, "below_minimum"

    if adaptive and max_k is None:
        # NEW: Calculate max_k from token budget
        _ensure_token_manager_imported()

        if _token_manager_imported:
            formatter = _get_adaptive_formatter()
            detail_map = {
                "summary": _DetailLevel.SUMMARY,
                "medium": _DetailLevel.MEDIUM,
                "full": _DetailLevel.FULL,
            }
            level = detail_map.get(detail_level.lower(), _DetailLevel.MEDIUM)
            tokens_per_item = formatter.budget.tokens_per_item(level)

            calculated_k, reason = formatter.adaptive_k(
                requested_k=k, tokens_per_item=tokens_per_item, min_k=3, max_k=200  # Safety limit (increased for benchmarks)
            )

            if calculated_k != k:
                logger.info(
                    f"Adaptive k: {k} → {calculated_k} "
                    f"(detail_level={detail_level}, reason={reason})"
                )
                return calculated_k, reason
            else:
                return k, None
        else:
            # Fallback: use hardcoded max_k
            max_k = 200  # Increased for benchmarks/evaluation

    # Use provided or fallback max_k
    if max_k is None:
        max_k = 200  # Increased for benchmarks/evaluation (legacy default was 10)

    if k > max_k:
        logger.warning(f"k={k} exceeds maximum {max_k}, clamping to {max_k}")
        return max_k, "exceeded_maximum"

    return k, None


def deduplicate_chunks(chunks: List[Dict], key: str = "chunk_id") -> List[Dict]:
    """
    Remove duplicate chunks based on key.

    Args:
        chunks: List of chunk dicts
        key: Key to use for deduplication

    Returns:
        Deduplicated list (preserves order)
    """
    seen = set()
    result = []

    for chunk in chunks:
        chunk_key = chunk.get(key)
        if chunk_key and chunk_key not in seen:
            seen.add(chunk_key)
            result.append(chunk)

    if len(result) < len(chunks):
        logger.debug(f"Deduplicated {len(chunks)} chunks to {len(result)}")

    return result


# =============================================================================
# FusionRetriever Factory (SSOT)
# =============================================================================

# Lazy-loaded FusionRetriever dependencies (avoid circular imports)
_fusion_retriever_import_attempted = False
_fusion_retriever_imported = False
_DeepInfraClient = None
_FusionRetriever = None
_FusionConfig = None


def _ensure_fusion_retriever_imported():
    """Lazy import FusionRetriever dependencies to avoid circular imports."""
    global _fusion_retriever_import_attempted, _fusion_retriever_imported
    global _DeepInfraClient, _FusionRetriever, _FusionConfig

    if _fusion_retriever_import_attempted:
        return _fusion_retriever_imported

    _fusion_retriever_import_attempted = True

    try:
        from src.retrieval import DeepInfraClient, FusionRetriever, FusionConfig

        _DeepInfraClient = DeepInfraClient
        _FusionRetriever = FusionRetriever
        _FusionConfig = FusionConfig
        _fusion_retriever_imported = True
        return True
    except ImportError as e:
        logger.error(f"Failed to import retrieval module: {e}")
        _fusion_retriever_imported = False
        return False


def create_fusion_retriever(
    vector_store,
    config,
    layer: int = 3,
    hyde_weight: float = 0.6,
    expansion_weight: float = 0.4,
):
    """
    SSOT factory for FusionRetriever initialization.

    This is the single source of truth for creating FusionRetriever instances.
    Both SearchTool (Layer 3) and SectionSearchTool (Layer 2) use this factory.

    Args:
        vector_store: PostgreSQL vector store instance
        config: Tool config object (may have hyde_weight, expansion_weight, default_k)
        layer: Layer number (2=sections, 3=chunks). Affects default_k.
        hyde_weight: Weight for HyDE component (default: 0.6)
        expansion_weight: Weight for expansion component (default: 0.4)

    Returns:
        Initialized FusionRetriever instance

    Raises:
        ImportError: If retrieval module cannot be imported
        ValueError: If configuration is invalid

    Example:
        retriever = create_fusion_retriever(
            vector_store=self.vector_store,
            config=self.config,
            layer=2,  # For section search
        )
    """
    if not _ensure_fusion_retriever_imported():
        raise ImportError("Failed to import retrieval module (DeepInfraClient, FusionRetriever)")

    # Determine default_k based on layer
    # Layer 3 (chunks): higher k (10) because chunks are smaller
    # Layer 2 (sections): lower k (5) because sections are larger
    layer_default_k = {2: 5, 3: 10}
    default_k = layer_default_k.get(layer, 10)

    # Get config values with fallbacks
    actual_hyde_weight = getattr(config, "hyde_weight", hyde_weight)
    actual_expansion_weight = getattr(config, "expansion_weight", expansion_weight)
    actual_default_k = getattr(config, "default_k", default_k)

    # Initialize DeepInfra client
    client = _DeepInfraClient()

    # Create fusion config
    fusion_config = _FusionConfig(
        hyde_weight=actual_hyde_weight,
        expansion_weight=actual_expansion_weight,
        default_k=actual_default_k,
    )

    # Create and return FusionRetriever
    retriever = _FusionRetriever(
        client=client,
        vector_store=vector_store,
        config=fusion_config,
    )

    logger.info(
        f"FusionRetriever initialized (layer={layer}, default_k={actual_default_k}, "
        f"hyde_weight={actual_hyde_weight}, expansion_weight={actual_expansion_weight})"
    )

    return retriever


def merge_chunk_lists(
    *chunk_lists: List[Dict], max_total: int = None, sort_by: str = "score"
) -> List[Dict]:
    """
    Merge multiple chunk lists, deduplicate, and optionally sort.

    Args:
        *chunk_lists: Variable number of chunk lists
        max_total: Maximum total chunks to return
        sort_by: Field to sort by (usually "score")

    Returns:
        Merged and sorted chunk list
    """
    # Flatten all lists
    all_chunks = []
    for chunk_list in chunk_lists:
        all_chunks.extend(chunk_list)

    # Deduplicate
    deduplicated = deduplicate_chunks(all_chunks)

    # Sort by score (ascending: lowest confidence first, highest last)
    if sort_by in deduplicated[0] if deduplicated else False:
        deduplicated.sort(key=lambda x: x.get(sort_by, 0), reverse=False)

    # Limit to max_total
    if max_total and len(deduplicated) > max_total:
        deduplicated = deduplicated[:max_total]

    return deduplicated
