"""
RAG Tools

17 specialized tools for retrieval and analysis.
All tools are registered automatically via @register_tool decorator.
"""

import logging

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)

# Import all tools to trigger registration
# This uses explicit imports for clarity (no auto-discovery magic)

_failed_imports = []


def _safe_import(module_name: str):
    """Import a tool module with error handling."""
    try:
        return __import__(f".{module_name}", globals(), locals(), ["*"], 1)
    except Exception as e:
        logger.error(f"Failed to import tool module '{module_name}': {e}", exc_info=True)
        _failed_imports.append((module_name, str(e)))
        return None


# Basic retrieval tools (5)
_safe_import("get_tool_help")
_safe_import("search")
_safe_import("get_document_list")
_safe_import("list_available_tools")
_safe_import("get_document_info")

# Advanced retrieval tools (10)
_safe_import("graph_search")
_safe_import("multi_doc_synthesizer")
_safe_import("contextual_chunk_enricher")
_safe_import("explain_search_results")
_safe_import("assess_retrieval_confidence")
_safe_import("filtered_search")
_safe_import("similarity_search")
_safe_import("expand_context")
_safe_import("browse_entities")
_safe_import("cluster_search")

# Analysis tools (2)
_safe_import("get_stats")
_safe_import("definition_aligner")

# Report import failures
if _failed_imports:
    logger.warning(
        f"Tool import failures ({len(_failed_imports)}): {', '.join(name for name, _ in _failed_imports)}"
    )

__all__ = ["BaseTool", "ToolInput", "ToolResult", "ToolRegistry", "get_registry"]
