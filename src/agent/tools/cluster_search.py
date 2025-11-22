"""
Cluster Search Tool

Auto-extracted and cleaned from tier2_advanced.py
"""

import logging
from typing import Any, Dict, List, Optional, Literal
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)



class ClusterSearchInput(ToolInput):
    """Input for cluster_search tool."""

    query: str = Field(
        ...,
        description="Search query to find relevant chunks",
    )

    cluster_ids: Optional[List[int]] = Field(
        None,
        description=(
            "Filter to specific cluster IDs (optional). "
            "If None, searches all clusters. "
            "Use get_stats tool first to see available clusters."
        ),
    )

    diversity_mode: bool = Field(
        False,
        description=(
            "Enable diversity mode: return max 1 result per cluster. "
            "Useful to avoid redundant results from the same topic."
        ),
    )

    k: int = Field(
        10,
        description="Maximum number of results to return",
        ge=1,
        le=50,
    )




@register_tool
class ClusterSearchTool(BaseTool):
    """
    Search with cluster awareness for topic-focused retrieval.

    Use cases:
    - Find all chunks about a specific topic (cluster)
    - Diversity-aware search (one result per cluster)
    - Cluster-filtered search (only specific clusters)

    Requires semantic clustering to be enabled during indexing.
    """

    name = "cluster_search"
    description = (
        "Search chunks with cluster awareness. "
        "Enables topic-focused retrieval and diversity-aware results. "
        "Requires semantic clustering enabled during indexing."
    )


    input_schema = ClusterSearchInput

    def execute_impl(
        self,
        query: str,
        cluster_ids: Optional[List[int]] = None,
        diversity_mode: bool = False,
        k: int = 10,
    ) -> ToolResult:
        """
        Execute cluster-aware search.

        Args:
            query: Search query
            cluster_ids: Filter to specific clusters (None = all)
            diversity_mode: Return max 1 result per cluster
            k: Number of results

        Returns:
            ToolResult with cluster-aware search results
        """
        import time

        start_time = time.time()

        try:
            # Perform standard hybrid search
            query_embedding = self.embedder.embed_texts([query])[0]

            results = self.vector_store.hierarchical_search(
                query_text=query,
                query_embedding=query_embedding,
                k_layer3=k * 3 if diversity_mode else k,  # Get more for diversity filtering
            )

            layer3_results = results.get("layer3", [])

            # Filter by cluster_ids if specified
            if cluster_ids is not None:
                layer3_results = [
                    r for r in layer3_results if r.get("cluster_id") in cluster_ids
                ]

            # Apply diversity mode if enabled
            if diversity_mode:
                seen_clusters = set()
                diverse_results = []

                for result in layer3_results:
                    cluster_id = result.get("cluster_id")
                    if cluster_id is None or cluster_id == -1:
                        # Include noise points
                        diverse_results.append(result)
                    elif cluster_id not in seen_clusters:
                        # First result from this cluster
                        seen_clusters.add(cluster_id)
                        diverse_results.append(result)

                    if len(diverse_results) >= k:
                        break

                layer3_results = diverse_results

            # Limit to k results
            layer3_results = layer3_results[:k]

            # Format results
            formatted_results = []
            for result in layer3_results:
                formatted = format_chunk_result(result)
                # Add cluster info
                formatted["cluster_id"] = result.get("cluster_id")
                formatted["cluster_label"] = result.get("cluster_label")
                formatted["cluster_confidence"] = result.get("cluster_confidence")
                formatted_results.append(formatted)

            # Extract citations
            citations = [
                {
                    "document_id": r.get("document_id"),
                    "section_title": r.get("section_title"),
                    "page_number": r.get("page_number"),
                }
                for r in layer3_results
            ]

            # Collect cluster statistics
            cluster_distribution = {}
            for result in layer3_results:
                cluster_id = result.get("cluster_id")
                if cluster_id is not None:
                    cluster_label = result.get("cluster_label", f"Cluster {cluster_id}")
                    cluster_distribution[cluster_label] = (
                        cluster_distribution.get(cluster_label, 0) + 1
                    )

            execution_time = int((time.time() - start_time) * 1000)

            return ToolResult(
                success=True,
                data=formatted_results,
                citations=citations,
                metadata={
                    "query": query,
                    "total_results": len(formatted_results),
                    "cluster_filter": cluster_ids,
                    "diversity_mode": diversity_mode,
                    "cluster_distribution": cluster_distribution,
                },
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Cluster search failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=[],
                error=f"Cluster search failed: {str(e)}",
                metadata={
                    "query": query,
                    "cluster_filter": cluster_ids,
                    "diversity_mode": diversity_mode,
                },
            )
