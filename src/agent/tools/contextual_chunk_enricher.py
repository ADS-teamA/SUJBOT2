"""
Contextual Chunk Enricher Tool

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
class ContextualChunkEnricherTool(BaseTool):
    """Enrich chunks with contextual information."""

    name = "contextual_chunk_enricher"
    description = "Add context to chunks for better embeddings"
    detailed_help = """
    Enrich chunks with document/section context using Anthropic Contextual Retrieval technique.
    Prepends contextual information to improve embedding quality and reduce context drift.

    **Research basis:** Anthropic (2024) - Contextual Retrieval reduces context drift by 58%

    **Enrichment modes:**
    - 'auto': Automatically detect best mode (section summary if available, else document)
    - 'document_summary': Prepend document-level summary
    - 'section_summary': Prepend section-level summary
    - 'both': Prepend both document and section summaries

    **When to use:**
    - Before embedding new chunks (indexing pipeline)
    - When chunks lack surrounding context
    - To improve semantic search accuracy
    - For ambiguous or short chunks

    **Best practices:**
    - Use 'auto' mode for most cases (intelligent selection)
    - Use 'both' for maximum context (but increases token count)
    - Enable include_metadata for richer context
    - Apply before embedding, not after retrieval

    **Method:** Retrieve summaries + prepend to chunk content
    
    **Cost:** Low (no LLM calls, just retrieval)
    """

    input_schema = ContextualChunkEnricherInput

    def execute_impl(
        self,
        chunk_ids: List[str],
        enrichment_mode: str = "auto",
        include_metadata: bool = True,
    ) -> ToolResult:
        """Enrich chunks with contextual information."""
        try:
            # Validate inputs
            if enrichment_mode not in ["auto", "document_summary", "section_summary", "both"]:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid enrichment_mode: {enrichment_mode}. Must be 'auto', 'document_summary', 'section_summary', or 'both'",
                )

            # Retrieve chunks from vector store
            enriched_chunks = []
            failed_chunks = []

            for chunk_id in chunk_ids:
                try:
                    # Get chunk from vector store
                    chunk = self._get_chunk_by_id(chunk_id)

                    if not chunk:
                        failed_chunks.append(chunk_id)
                        logger.warning(f"Chunk {chunk_id} not found in vector store")
                        continue

                    # Determine enrichment mode
                    actual_mode = self._determine_enrichment_mode(chunk, enrichment_mode)

                    # Enrich chunk with context
                    enriched_chunk = self._enrich_chunk(chunk, actual_mode, include_metadata)

                    enriched_chunks.append(enriched_chunk)

                except Exception as e:
                    logger.error(f"Failed to enrich chunk {chunk_id}: {e}")
                    failed_chunks.append(chunk_id)

            if not enriched_chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Failed to enrich any chunks. Failed: {failed_chunks}",
                )

            # Prepare result data
            result_data = {
                "enriched_count": len(enriched_chunks),
                "failed_count": len(failed_chunks),
                "enrichment_mode": enrichment_mode,
                "enriched_chunks": enriched_chunks,
                "failed_chunk_ids": failed_chunks,
            }

            return ToolResult(
                success=True,
                data=result_data,
                metadata={
                    "enrichment_mode": enrichment_mode,
                    "success_rate": len(enriched_chunks) / len(chunk_ids),
                },
            )

        except Exception as e:
            logger.error(f"Contextual chunk enrichment failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve chunk by ID from vector store."""
        # Search for chunk by exact chunk_id match
        # Use hierarchical_search to access layer3
        results = self.vector_store.hierarchical_search(
            query_text=chunk_id,
            query_embedding=None,
            k_layer3=100,
        )

        layer3_chunks = results.get("layer3", [])

        # Find exact match
        for chunk in layer3_chunks:
            if chunk.get("chunk_id") == chunk_id:
                return chunk

        return None

    def _determine_enrichment_mode(self, chunk: Dict, requested_mode: str) -> str:
        """Determine actual enrichment mode based on available data."""
        if requested_mode != "auto":
            return requested_mode

        # Auto mode: Intelligently select based on available summaries
        has_section_summary = bool(chunk.get("section_summary"))
        has_document_summary = bool(chunk.get("document_summary"))

        if has_section_summary:
            return "section_summary"
        elif has_document_summary:
            return "document_summary"
        else:
            return "document_summary"  # Fallback

    def _enrich_chunk(self, chunk: Dict, mode: str, include_metadata: bool) -> Dict:
        """Enrich chunk with contextual information."""
        enriched = {
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "original_content": chunk.get("content", chunk.get("text", "")),
            "enrichment_mode": mode,
        }

        # Build contextual prefix
        context_parts = []

        # Add document context
        if mode in ["document_summary", "both"]:
            doc_summary = chunk.get("document_summary", "(Document summary unavailable)")
            doc_id = chunk.get("doc_id") or chunk.get("document_id", "Unknown")
            context_parts.append(f"Document: {doc_id}")
            context_parts.append(f"Document Summary: {doc_summary}")

        # Add section context
        if mode in ["section_summary", "both"]:
            section_summary = chunk.get("section_summary")
            section_title = chunk.get("section_title", "Unknown Section")

            if section_summary:
                context_parts.append(f"Section: {section_title}")
                context_parts.append(f"Section Summary: {section_summary}")
            elif mode == "section_summary":
                # Fallback to section title only
                context_parts.append(f"Section: {section_title}")

        # Add metadata
        if include_metadata:
            metadata_parts = []
            if "page_number" in chunk:
                metadata_parts.append(f"Page {chunk['page_number']}")
            if "section_title" in chunk:
                metadata_parts.append(f"Section: {chunk['section_title']}")

            if metadata_parts:
                context_parts.append(" | ".join(metadata_parts))

        # Combine context + original content
        context_prefix = "\n".join(context_parts)
        enriched_content = f"{context_prefix}\n\n{enriched['original_content']}"

        enriched["enriched_content"] = enriched_content
        enriched["context_prefix"] = context_prefix
        enriched["content_length_increase"] = len(enriched_content) - len(enriched["original_content"])

        return enriched


class ExplainSearchResultsInput(ToolInput):
    """Input for explain_search_results tool."""

    chunk_ids: List[str] = Field(..., description="Chunk IDs from search results to explain")


