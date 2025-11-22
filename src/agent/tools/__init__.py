"""
RAG Tools

17 specialized tools for retrieval and analysis.
All tools are registered automatically via @register_tool decorator.
"""

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import ToolRegistry, get_registry

# Import all tools to trigger registration
# This uses explicit imports for clarity (no auto-discovery magic)

# Basic retrieval tools (5)
from . import get_tool_help
from . import search
from . import get_document_list
from . import list_available_tools
from . import get_document_info

# Advanced retrieval tools (10)
from . import graph_search
from . import multi_doc_synthesizer
from . import contextual_chunk_enricher
from . import explain_search_results
from . import assess_retrieval_confidence
from . import filtered_search
from . import similarity_search
from . import expand_context
from . import browse_entities
from . import cluster_search

# Analysis tools (2)
from . import get_stats
from . import definition_aligner

__all__ = ["BaseTool", "ToolInput", "ToolResult", "ToolRegistry", "get_registry"]
