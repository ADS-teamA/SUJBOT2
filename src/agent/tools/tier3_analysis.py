"""
TIER 3: Analysis & Insights Tools

Deep analysis tools (1-3s) for specialized tasks.
Use for entity analysis, timeline extraction, statistics.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import BaseTool, ToolInput, ToolResult
from .registry import register_tool
from .utils import format_chunk_result

logger = logging.getLogger(__name__)


# ============================================================================
# TIER 3 TOOLS: Analysis & Insights
# ============================================================================


# timeline_view tool removed - primitive regex-based implementation
# Use filtered_search with filter_type='temporal' or knowledge graph entity extraction instead

# summarize_section tool removed - false advertising (returns chunks, not LLM summary)
# Use get_document_info(info_type='section_details') to get section chunks
# Use filtered_search with section filter for more control


# ============================================================================
# UNIFIED TOOLS (Consolidated from multiple similar tools)
# ============================================================================


class GetStatsInput(ToolInput):
    """Input for unified get_stats tool."""

    stat_scope: str = Field(
        ...,
        description="Statistics scope: 'corpus' (overall stats), 'index' (comprehensive index stats with embedding/cache info), 'document' (per-document stats), 'entity' (knowledge graph stats)",
    )
    include_cache_stats: bool = Field(
        False, description="Include embedding cache statistics (for 'index' scope)"
    )


@register_tool
class GetStatsTool(BaseTool):
    """Get corpus/index statistics."""

    name = "get_stats"
    description = "Get corpus/index stats"
    detailed_help = """
    Get statistics about corpus, index, documents, or entities.

    **Stat scopes:**
    - 'corpus': Overall document counts, sizes
    - 'index': Comprehensive FAISS index stats (layers, dimensions, cache)
    - 'document': Per-document statistics
    - 'entity': Knowledge graph entity statistics

    **When to use:**
    - "How many documents?"
    - "Corpus statistics"
    - "Index information"
    - System/debugging queries

    **Best practices:**
    - Use 'corpus' for quick document counts
    - Use 'index' for detailed FAISS information
    - Set include_cache_stats=true for embedding cache info
    - Fast metadata aggregation (no search required)

    **Method:** Metadata aggregation
    **Speed:** <100ms (metadata lookup only)
    """
    tier = 3
    input_schema = GetStatsInput

    def execute_impl(self, stat_scope: str, include_cache_stats: bool = False) -> ToolResult:
        try:
            stats = {}

            if stat_scope in ["corpus", "document", "index"]:
                # Get vector store statistics
                vs_stats = self.vector_store.get_stats()
                stats["vector_store"] = vs_stats

                # Count unique documents
                sample_results = self.vector_store.hierarchical_search(
                    query_text="",
                    query_embedding=None,
                    k_layer1=100,
                )

                unique_docs = set()
                for doc in sample_results.get("layer1", []):
                    doc_id = doc.get("document_id") or doc.get("doc_id", "Unknown")
                    unique_docs.add(doc_id)

                stats["unique_documents"] = len(unique_docs)
                stats["document_list"] = sorted(list(unique_docs))

            if stat_scope in ["corpus", "entity", "index"]:
                # Get knowledge graph statistics
                if self.knowledge_graph:
                    from collections import Counter

                    entity_types = Counter()
                    relationship_types = Counter()

                    for entity in self.knowledge_graph.entities.values():
                        entity_types[entity.type] += 1

                    for rel in self.knowledge_graph.relationships.values():
                        relationship_types[rel.type] += 1

                    stats["knowledge_graph"] = {
                        "total_entities": len(self.knowledge_graph.entities),
                        "total_relationships": len(self.knowledge_graph.relationships),
                        "entity_types": dict(entity_types),
                        "relationship_types": dict(relationship_types),
                    }

                    if stat_scope == "index":
                        # Add top entities for index scope
                        stats["knowledge_graph"]["top_entities"] = [
                            {"id": e.id, "name": e.name, "type": e.type}
                            for e in sorted(
                                self.knowledge_graph.entities.values(),
                                key=lambda x: x.confidence,
                                reverse=True,
                            )[:10]
                        ]
                else:
                    stats["knowledge_graph"] = None

            if stat_scope == "index":
                # Additional index-specific information
                stats["hybrid_search_enabled"] = stats.get("vector_store", {}).get(
                    "hybrid_enabled", False
                )

                # Embedding model information
                if self.embedder:
                    stats["embedding_model"] = {
                        "model_name": self.embedder.model_name,
                        "dimensions": self.embedder.dimensions,
                        "model_type": self.embedder.model_type,
                    }

                    # Cache statistics if requested
                    if include_cache_stats and hasattr(self.embedder, "get_cache_stats"):
                        stats["embedding_cache"] = self.embedder.get_cache_stats()

                # Document info
                stats["documents"] = {
                    "count": stats["unique_documents"],
                    "document_ids": stats["document_list"],
                }

                # System configuration
                stats["configuration"] = {
                    "reranking_enabled": hasattr(self, "reranker") and self.reranker is not None,
                    "context_assembler_enabled": hasattr(self, "context_assembler")
                    and self.context_assembler is not None,
                }

            if stat_scope not in ["corpus", "index", "document", "entity"]:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid stat_scope: {stat_scope}. Must be 'corpus', 'index', 'document', or 'entity'",
                )

            return ToolResult(
                success=True,
                data=stats,
                metadata={
                    "stat_scope": stat_scope,
                    "stat_categories": list(stats.keys()),
                },
            )

        except Exception as e:
            logger.error(f"Get stats failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
