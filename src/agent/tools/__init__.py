"""
RAG Tools

12 specialized tools for retrieval and analysis.
All tools are registered automatically via @register_tool decorator.
"""

import logging
import importlib

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import ToolRegistry, get_registry

logger = logging.getLogger(__name__)

# Import all tools to trigger registration
# This uses explicit imports for clarity (no auto-discovery magic)

_failed_imports = []


def _safe_import(module_name: str):
    """Import a tool module with error handling."""
    try:
        return importlib.import_module(f".{module_name}", package=__name__)
    except ImportError as e:
        # Expected: missing optional dependency - log at warning level
        logger.warning(f"Tool module '{module_name}' unavailable (missing dependency): {e}")
        _failed_imports.append((module_name, str(e)))
        return None
    except (SyntaxError, NameError, AttributeError) as e:
        # Code bugs should be fixed, not silently ignored
        logger.error(f"Code error in tool module '{module_name}': {e}", exc_info=True)
        raise
    except Exception as e:
        # Unexpected errors - log with traceback but don't crash
        logger.error(f"Unexpected error importing '{module_name}': {e}", exc_info=True)
        _failed_imports.append((module_name, str(e)))
        return None


# Basic retrieval tools (5)
_safe_import("get_tool_help")
_safe_import("search")
_safe_import("list_available_tools")
_safe_import("get_document_info")  # Document metadata and summaries
_safe_import("get_document_list")  # List all documents for orchestrator routing

# Advanced retrieval tools (5)
_safe_import("graphiti_search")  # Temporal knowledge graph search (replaces graph_search)
_safe_import("multi_doc_synthesizer")
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
