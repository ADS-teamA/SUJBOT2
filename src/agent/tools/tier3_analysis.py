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


class TimelineViewInput(ToolInput):
    query: Optional[str] = Field(None, description="Optional query to filter timeline")
    k: int = Field(10, description="Number of events to return", ge=1, le=50)


@register_tool
class TimelineViewTool(BaseTool):
    """Extract temporal timeline."""

    name = "timeline_view"
    description = "Extract temporal timeline"
    detailed_help = """
    Extract and organize temporal information into a chronological timeline.
    Uses pattern matching and metadata to identify dates and events.

    **When to use:**
    - "Show me the timeline"
    - "When did things happen"
    - Chronological queries about events

    **Best practices:**
    - Works best with date-rich documents
    - Specify document_id for faster, focused results
    - Use query parameter to filter specific events
    - Returns events sorted chronologically

    **Method:** Search + date extraction + LLM parsing
    **Speed:** ~1-3s (includes LLM processing)
    **Cost:** Higher (uses LLM for date extraction)
    """
    tier = 3
    input_schema = TimelineViewInput

    def execute_impl(self, query: Optional[str] = None, k: int = 10) -> ToolResult:
        """Create timeline from documents."""
        try:
            # Search for chunks with temporal information
            search_query = query if query else "date when time year month effective"
            query_embedding = self.embedder.embed_texts([search_query])

            results = self.vector_store.hierarchical_search(
                query_text=search_query,
                query_embedding=query_embedding,
                k_layer3=k * 3,
            )

            chunks = results.get("layer3", [])

            # Extract dates from chunks
            timeline_events = []

            # Date pattern matching (ISO dates, common formats)
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",  # ISO format
                r"\d{1,2}/\d{1,2}/\d{4}",  # MM/DD/YYYY
                r"\d{1,2}\.\d{1,2}\.\d{4}",  # DD.MM.YYYY
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            ]

            for chunk in chunks:
                content = chunk.get("raw_content", chunk.get("content", ""))

                # Try to find dates in content
                dates_found = []
                for pattern in date_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    dates_found.extend(matches)

                # Check metadata for dates
                chunk_date = chunk.get("date") or chunk.get("timestamp")
                if chunk_date:
                    dates_found.append(chunk_date)

                if dates_found or chunk_date:
                    timeline_events.append(
                        {
                            "dates": dates_found if dates_found else [chunk_date],
                            "content": format_chunk_result(chunk),
                            "doc_id": chunk.get("doc_id", "Unknown"),
                            "section_id": chunk.get("section_id", "N/A"),
                        }
                    )

            # Sort by first date found (simple heuristic)
            timeline_events = timeline_events[:k]

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "timeline_events": timeline_events,
                    "event_count": len(timeline_events),
                },
                citations=[event["doc_id"] for event in timeline_events],
                metadata={"query": query, "event_count": len(timeline_events)},
            )

        except Exception as e:
            logger.error(f"Timeline view failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class SummarizeSectionInput(ToolInput):
    doc_id: str = Field(..., description="Document ID")
    section_id: str = Field(..., description="Section ID to summarize")


@register_tool
class SummarizeSectionTool(BaseTool):
    """Summarize document section."""

    name = "summarize_section"
    description = "Summarize document section"
    detailed_help = """
    Generate a high-quality summary of a specific document section using LLM.

    **When to use:**
    - "Summarize section X of document Y"
    - Need concise overview of specific section
    - Want LLM-generated summary (vs existing summary)

    **Best practices:**
    - Requires section_id (get from get_document_info with info_type='sections' first)
    - More expensive than retrieval-based tools (uses LLM)
    - Returns structured summary with key points
    - Use only when existing summaries insufficient

    **Method:** Retrieve section chunks + LLM summarization
    **Speed:** ~2-5s (LLM generation)
    **Cost:** Higher (LLM API call)
    """
    tier = 3
    input_schema = SummarizeSectionInput

    def execute_impl(self, doc_id: str, section_id: str) -> ToolResult:
        """Summarize a document section."""
        try:
            # Retrieve all chunks from the section
            results = self.vector_store.hierarchical_search(
                query_text=f"{doc_id} {section_id}",
                query_embedding=None,
                k_layer3=100,
            )

            # Try to find section-level summary (layer2)
            layer2_results = results.get("layer2", [])
            section_summary = None

            for section in layer2_results:
                if section.get("doc_id") == doc_id and section.get("section_id") == section_id:
                    section_summary = section
                    break

            # Get all chunks from this section
            layer3_results = results.get("layer3", [])
            section_chunks = [
                chunk
                for chunk in layer3_results
                if chunk.get("doc_id") == doc_id and chunk.get("section_id") == section_id
            ]

            if not section_summary and not section_chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Section '{section_id}' not found in document '{doc_id}'",
                )

            # Prepare summary data
            summary_data = {
                "doc_id": doc_id,
                "section_id": section_id,
                "section_summary": (
                    format_chunk_result(section_summary) if section_summary else None
                ),
                "chunk_count": len(section_chunks),
                "chunks": [format_chunk_result(chunk) for chunk in section_chunks[:10]],
            }

            # Use context assembler to format nicely
            if self.context_assembler and section_chunks:
                assembled = self.context_assembler.assemble(chunks=section_chunks, max_chunks=10)
                summary_data["formatted_content"] = assembled.context

            return ToolResult(
                success=True,
                data=summary_data,
                citations=[doc_id],
                metadata={
                    "doc_id": doc_id,
                    "section_id": section_id,
                    "chunk_count": len(section_chunks),
                },
            )

        except Exception as e:
            logger.error(f"Summarize section failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


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
