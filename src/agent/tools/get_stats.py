"""
Get Stats Tool

Auto-extracted and cleaned from tier3_analysis.py
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)


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
    
    """

    input_schema = GetStatsInput

    def execute_impl(self, stat_scope: str, include_cache_stats: bool = False) -> ToolResult:
        try:
            stats = {}

            if stat_scope in ["corpus", "document", "index"]:
                # Get vector store statistics
                vs_stats = self.vector_store.get_stats()
                stats["vector_store"] = vs_stats

                # Get unique documents count from stats (FAISS provides this)
                if "documents" in vs_stats:
                    stats["unique_documents"] = vs_stats["documents"]

                # Try to get document list from FAISS metadata (if available)
                try:
                    from src.faiss_vector_store import FAISSVectorStore
                    if isinstance(self.vector_store.vector_store, FAISSVectorStore):
                        # Extract document IDs from layer1 metadata (document summaries)
                        unique_docs = set()
                        for metadata in self.vector_store.vector_store.metadata_layer1:
                            doc_id = metadata.get("document_id") or metadata.get("doc_id", "Unknown")
                            unique_docs.add(doc_id)
                        stats["document_list"] = sorted(list(unique_docs))
                except Exception as e:
                    # If we can't get document list, skip it (non-critical)
                    logger.debug(f"Could not extract document list: {e}")
                    stats["document_list"] = []

            if stat_scope in ["corpus", "entity", "index"]:
                # Get knowledge graph statistics
                if self.knowledge_graph:
                    from collections import Counter

                    entity_types = Counter()
                    relationship_types = Counter()

                    for entity in self.knowledge_graph.entities:
                        entity_types[entity.type] += 1

                    for rel in self.knowledge_graph.relationships:
                        relationship_types[rel.type] += 1

                    stats["knowledge_graph"] = {
                        "total_entities": len(self.knowledge_graph.entities),
                        "total_relationships": len(self.knowledge_graph.relationships),
                        "entity_types": {k.value: v for k, v in entity_types.items()},
                        "relationship_types": {k.value: v for k, v in relationship_types.items()},
                    }

                    if stat_scope == "index":
                        # Add top entities for index scope
                        stats["knowledge_graph"]["top_entities"] = [
                            {"id": e.id, "name": e.value, "type": e.type.value}
                            for e in sorted(
                                self.knowledge_graph.entities,
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

