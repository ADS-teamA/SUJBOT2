"""
Assess Retrieval Confidence Tool

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
class AssessRetrievalConfidenceTool(BaseTool):
    """Assess confidence of retrieval results."""

    name = "assess_retrieval_confidence"
    description = "Evaluate confidence in search results"
    detailed_help = """
    Assess the confidence/quality of retrieval results using multiple signals.

    Analyzes:
    - Retrieval scores (top score, gap, spread, consensus)
    - Method agreement (BM25 vs Dense correlation)
    - Context quality (document diversity, section diversity)
    - Knowledge graph support

    **When to use:**
    - After search, to determine if results are reliable
    - Before answering critical questions (legal, compliance, financial)
    - When results seem uncertain or contradictory
    - To decide if query expansion or alternative retrieval is needed

    **Best practices:**
    - Use AFTER a search (requires chunk IDs from search results)
    - High confidence (≥0.85): Safe to use for automated response
    - Medium confidence (0.70-0.84): Review recommended
    - Low confidence (<0.70): Mandatory review or alternative retrieval

    **Returns:** Confidence score (0-1) with detailed breakdown

    **Method:** Score analysis from hybrid search metadata
    
    """

    input_schema = AssessRetrievalConfidenceInput

    def execute_impl(self, chunk_ids: List[str]) -> ToolResult:
        """Assess confidence of specific chunks."""
        try:
            from src.agent.rag_confidence import RAGConfidenceScorer

            # Get Layer 3 metadata
            layer3_chunks = []
            if hasattr(self.vector_store, "metadata_layer3"):
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            # Find chunks by ID
            chunks = []
            for chunk_id in chunk_ids:
                for meta in layer3_chunks:
                    if meta.get("chunk_id") == chunk_id:
                        chunks.append(meta)
                        break

            if not chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    metadata={"chunk_ids": chunk_ids, "found": False},
                    error=f"No chunks found with provided IDs: {chunk_ids[:5]}{'...' if len(chunk_ids) > 5 else ''}",
                )

            # Score confidence
            scorer = RAGConfidenceScorer()
            confidence = scorer.score_retrieval(chunks)

            # Build response
            response_data = confidence.to_dict()

            # Add actionable recommendations
            recommendations = []
            if confidence.overall_confidence < 0.50:
                recommendations.append(
                    "CRITICAL: Very low confidence. Consider query expansion or alternative retrieval methods."
                )
                recommendations.append("Try: Use 'search' with num_expands=3-5 for better recall.")
            elif confidence.overall_confidence < 0.70:
                recommendations.append(
                    "WARNING: Low confidence. Recommend manual review before using results."
                )
                if confidence.bm25_dense_agreement < 0.5:
                    recommendations.append(
                        "BM25 and Dense retrieval disagree. Try exact_match_search for keyword-based retrieval."
                    )
            elif confidence.overall_confidence < 0.85:
                recommendations.append(
                    "MODERATE: Medium confidence. Review recommended for critical use cases."
                )

            if confidence.document_diversity > 0.7:
                recommendations.append(
                    "High document diversity detected. Results may be scattered across multiple sources."
                )

            if not confidence.graph_support and hasattr(self, "knowledge_graph"):
                recommendations.append(
                    "No knowledge graph support. Consider using multi_hop_search for graph-based retrieval."
                )

            response_data["recommendations"] = recommendations

            # Safely extract confidence level
            interpretation = confidence.interpretation
            if " - " in interpretation:
                confidence_level = interpretation.split(" - ")[0]
            else:
                # Fallback: extract from interpretation directly
                if "HIGH" in interpretation:
                    confidence_level = "HIGH"
                elif "MEDIUM" in interpretation:
                    confidence_level = "MEDIUM"
                elif "LOW" in interpretation:
                    confidence_level = "LOW"
                elif "VERY LOW" in interpretation:
                    confidence_level = "VERY LOW"
                else:
                    logger.warning(f"Unknown confidence interpretation format: {interpretation}")
                    confidence_level = "UNKNOWN"

            return ToolResult(
                success=True,
                data=response_data,
                metadata={
                    "chunk_count": len(chunks),
                    "confidence_level": confidence_level,
                    "should_flag": confidence.should_flag,
                },
            )

        except Exception as e:
            logger.error(
                f"Assess retrieval confidence failed for chunks {chunk_ids}: {e}",
                exc_info=True,
                extra={
                    "chunk_ids": chunk_ids,
                    "error_type": type(e).__name__,
                },
            )
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to assess confidence for {len(chunk_ids)} chunks: {type(e).__name__}: {str(e)[:200]}. "
                f"This may indicate a data integrity issue. Try refreshing the search or using a different tool.",
            )


# ============================================================================
# UNIFIED TOOLS (Consolidated from multiple similar tools)
# ============================================================================


class FilteredSearchInput(ToolInput):
    """Input for unified filtered_search tool with search method control."""

    query: str = Field(..., description="Search query")
    search_method: str = Field(
        "hybrid",
        description="Search method: 'hybrid' (BM25+Dense+RRF, default, ), 'bm25_only' (keyword only, ~50-100ms), 'dense_only' (semantic only, ~100-200ms)"
    )
    filter_type: Optional[str] = Field(
        None, description="Type of filter to apply: 'document', 'section', 'metadata', 'temporal'. If None, searches entire database"
    )
    filter_value: Optional[str] = Field(
        None, description="Filter value (document_id, section_title, or date range). Required if filter_type is set"
    )
    document_type: Optional[str] = Field(None, description="For metadata filter: document type")
    section_type: Optional[str] = Field(None, description="For metadata filter: section type")
    start_date: Optional[str] = Field(
        None, description="For temporal filter: start date (ISO: YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        None, description="For temporal filter: end date (ISO: YYYY-MM-DD)"
    )
    k: int = Field(6, description="Number of results", ge=1, le=10)

    # Legacy compatibility for exact_match_search
    search_type: Optional[str] = Field(
        None, description="DEPRECATED: Use search_method='bm25_only' instead. Maps 'keywords'/'cross_references' to bm25_only"
    )
    document_id: Optional[str] = Field(
        None, description="DEPRECATED: Use filter_type='document', filter_value=<doc_id> instead"
    )
    section_id: Optional[str] = Field(
        None, description="DEPRECATED: Use filter_type='section', filter_value=<section_title> instead"
    )
    use_hyde: bool = Field(
        False, description="Enable HyDE (Hypothetical Document Embeddings) for better zero-shot retrieval. Slower () but higher quality."
    )


