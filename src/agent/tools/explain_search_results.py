"""
Explain Search Results Tool

Auto-extracted and cleaned from tier2_advanced.py
"""

import logging
from typing import Any, Dict, List, Optional, Literal
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)


@register_tool
class ExplainSearchResultsTool(BaseTool):
    """Debug search results."""

    name = "explain_search_results"
    description = "Explain search result scores"
    detailed_help = """
    Debug tool to understand why specific chunks were retrieved.
    Shows score breakdowns: BM25, Dense, RRF, and Rerank scores.

    **When to use:**
    - Debugging unexpected search results
    - Understanding why certain chunks appeared
    - Investigating retrieval quality

    **Best practices:**
    - Use AFTER a search (requires chunk IDs from search results)
    - Most useful for debugging, not regular queries
    - Helps identify which retrieval method (BM25 vs Dense) found the chunk

    **Method:** Score analysis from hybrid search metadata
    
    """

    input_schema = ExplainSearchResultsInput

    def execute_impl(self, chunk_ids: List[str]) -> ToolResult:
        """Explain search results with score breakdowns."""
        try:
            # Get Layer 3 metadata
            layer3_chunks = []
            if hasattr(self.vector_store, "metadata_layer3"):
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            explanations = []

            for chunk_id in chunk_ids:
                # Find chunk metadata
                chunk = None
                for meta in layer3_chunks:
                    if meta.get("chunk_id") == chunk_id:
                        chunk = meta
                        break

                if not chunk:
                    explanations.append(
                        {"chunk_id": chunk_id, "found": False, "error": "Chunk not found"}
                    )
                    continue

                # Extract scores (from HybridVectorStore modification)
                explanation = {
                    "chunk_id": chunk_id,
                    "found": True,
                    "document_id": chunk.get("document_id"),
                    "section_id": chunk.get("section_id"),
                    "scores": {
                        "bm25_score": chunk.get("bm25_score", None),
                        "dense_score": chunk.get("dense_score", None),
                        "rrf_score": chunk.get("rrf_score", None),
                        "rerank_score": chunk.get("rerank_score", None),
                    },
                    "fusion_method": chunk.get("fusion_method", "unknown"),
                    "content_preview": chunk.get("content", "")[:200] + "...",
                }

                # Determine which retrieval method contributed most
                scores = explanation["scores"]
                if scores["bm25_score"] and scores["dense_score"]:
                    if scores["bm25_score"] > scores["dense_score"]:
                        explanation["primary_retrieval_method"] = "sparse (BM25 keyword match)"
                    else:
                        explanation["primary_retrieval_method"] = "dense (semantic similarity)"
                elif scores["bm25_score"]:
                    explanation["primary_retrieval_method"] = "sparse (BM25 only)"
                elif scores["dense_score"]:
                    explanation["primary_retrieval_method"] = "dense (semantic only)"
                else:
                    explanation["primary_retrieval_method"] = "unknown"

                explanations.append(explanation)

            # Check if hybrid search is enabled
            hybrid_enabled = hasattr(self.vector_store, "bm25_store") or (
                hasattr(self.vector_store, "faiss_store")
                and hasattr(self.vector_store.faiss_store, "bm25_store")
            )

            return ToolResult(
                success=True,
                data={
                    "explanations": explanations,
                    "hybrid_search_enabled": hybrid_enabled,
                    "note": "BM25/Dense scores only available if retrieved via hybrid search with score preservation",
                },
                metadata={"chunk_count": len(chunk_ids), "hybrid_enabled": hybrid_enabled},
            )

        except Exception as e:
            logger.error(f"Explain search results failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class AssessRetrievalConfidenceInput(ToolInput):
    """Input for assess_retrieval_confidence tool."""

    chunk_ids: List[str] = Field(..., description="Chunk IDs from search results to assess")


