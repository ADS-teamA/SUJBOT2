"""
Multi-Doc Synthesizer Tool

Auto-extracted and cleaned from tier2_advanced.py
"""

import logging
from typing import Any, Dict, List, Optional, Literal
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)



class MultiDocSynthesizerInput(ToolInput):
    """Input for multi_doc_synthesizer tool."""

    document_ids: List[str] = Field(
        ...,
        description="List of document IDs to synthesize (2-5 documents recommended)",
        min_items=2,
        max_items=10,
    )
    synthesis_query: str = Field(
        ...,
        description="Query describing what to synthesize from the documents (e.g., 'Compare privacy policies', 'Summarize requirements')",
    )
    k_per_document: int = Field(
        5, description="Number of chunks to retrieve per document", ge=1, le=20
    )
    synthesis_mode: str = Field(
        "compare",
        description="Synthesis mode: 'compare' (find differences/similarities), 'summarize' (unified summary), 'analyze' (deep analysis)",
    )




@register_tool
class MultiDocSynthesizerTool(BaseTool):
    """Synthesize information from multiple documents."""

    name = "multi_doc_synthesizer"
    description = "Synthesize info from multiple docs"
    detailed_help = """
    Synthesize information from multiple documents using LLM.
    Retrieves relevant chunks from each document and generates unified synthesis.

    **Synthesis modes:**
    - 'compare': Find similarities and differences between documents
    - 'summarize': Create unified summary across all documents
    - 'analyze': Deep analysis of common themes and patterns

    **When to use:**
    - "Compare privacy policies of documents A, B, C"
    - "Summarize requirements across multiple standards"
    - "Analyze how different regulations address data retention"
    - Multi-document question answering

    **Best practices:**
    - Use 2-5 documents for best results (10 max)
    - Provide specific synthesis_query (not generic)
    - Use compare mode for differences, summarize for unified overview
    - Results cite all source documents

    **Method:** search (with filter_type='document') per document + LLM synthesis
    
    **Cost:** Higher (multiple retrievals + LLM synthesis)
    """

    input_schema = MultiDocSynthesizerInput

    def execute_impl(
        self,
        document_ids: List[str],
        synthesis_query: str,
        k_per_document: int = 5,
        synthesis_mode: str = "compare",
    ) -> ToolResult:
        """Synthesize information from multiple documents."""
        try:
            # Validate inputs
            if len(document_ids) < 2:
                return ToolResult(
                    success=False,
                    data=None,
                    error="At least 2 documents required for synthesis",
                )

            if len(document_ids) > 10:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Maximum 10 documents allowed (performance constraint)",
                )

            if synthesis_mode not in ["compare", "summarize", "analyze"]:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid synthesis_mode: {synthesis_mode}. Must be 'compare', 'summarize', or 'analyze'",
                )

            # Retrieve relevant chunks from each document using public API
            document_chunks = {}
            total_chunks = 0

            # Generate query embedding once for efficiency
            query_embedding = self.embedder.embed_texts([synthesis_query])

            for doc_id in document_ids:
                # Use hierarchical_search with document filter (public API)
                try:
                    results = self.vector_store.hierarchical_search(
                        query_text=synthesis_query,
                        query_embedding=query_embedding,
                        k_layer3=k_per_document,
                        document_filter=doc_id,
                    )

                    # Get layer3 chunks for this document
                    chunks = results.get("layer3", [])

                    # Format chunks for consistency
                    from ._utils import format_chunk_result

                    formatted_chunks = [format_chunk_result(c) for c in chunks]

                    document_chunks[doc_id] = formatted_chunks
                    total_chunks += len(formatted_chunks)

                    logger.info(f"Retrieved {len(formatted_chunks)} chunks from document {doc_id}")

                except Exception as e:
                    logger.warning(f"Failed to retrieve chunks from document {doc_id}: {e}")
                    document_chunks[doc_id] = []
                    continue

            # Check if we got any chunks
            if total_chunks == 0:
                return ToolResult(
                    success=False,
                    data=None,
                    error="No relevant chunks found in any of the specified documents",
                )

            # Prepare synthesis prompt
            synthesis_prompt = self._build_synthesis_prompt(
                document_chunks, synthesis_query, synthesis_mode
            )

            # Generate synthesis using LLM (if available)
            if hasattr(self, "_generate_synthesis"):
                synthesis_text = self._generate_synthesis(synthesis_prompt, synthesis_mode)
            else:
                # Fallback: Return structured chunks without LLM
                synthesis_text = "LLM synthesis not available. Returning structured chunks."

            # Prepare result data
            result_data = {
                "synthesis_query": synthesis_query,
                "synthesis_mode": synthesis_mode,
                "document_count": len(document_ids),
                "total_chunks_retrieved": total_chunks,
                "synthesis": synthesis_text,
                "document_chunks": {
                    doc_id: [
                        {
                            "chunk_id": chunk.get("chunk_id", "unknown"),
                            "content": chunk.get("content", chunk.get("text", ""))[:500],
                            "section_title": chunk.get("section_title", "N/A"),
                        }
                        for chunk in chunks[:3]  # Show first 3 chunks per doc
                    ]
                    for doc_id, chunks in document_chunks.items()
                },
            }

            # Generate citations
            citations = list(document_ids)

            return ToolResult(
                success=True,
                data=result_data,
                citations=citations,
                metadata={
                    "synthesis_mode": synthesis_mode,
                    "document_count": len(document_ids),
                    "total_chunks": total_chunks,
                },
            )

        except Exception as e:
            logger.error(f"Multi-document synthesis failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _build_synthesis_prompt(
        self, document_chunks: Dict[str, List[Dict]], query: str, mode: str
    ) -> str:
        """Build synthesis prompt for LLM."""
        prompt_parts = []

        if mode == "compare":
            prompt_parts.append(
                f"Compare the following documents regarding: {query}\n"
                f"Identify similarities, differences, and conflicts.\n\n"
            )
        elif mode == "summarize":
            prompt_parts.append(
                f"Summarize the following documents regarding: {query}\n"
                f"Create a unified summary synthesizing information from all sources.\n\n"
            )
        elif mode == "analyze":
            prompt_parts.append(
                f"Analyze the following documents regarding: {query}\n"
                f"Identify patterns, themes, and key insights across all documents.\n\n"
            )

        # Add chunks from each document
        for doc_id, chunks in document_chunks.items():
            if not chunks:
                continue

            prompt_parts.append(f"=== Document: {doc_id} ===\n")
            for i, chunk in enumerate(chunks[:5], 1):  # Max 5 chunks per doc
                content = chunk.get("content", chunk.get("text", ""))
                section = chunk.get("section_title", "N/A")
                prompt_parts.append(f"[Chunk {i} - {section}]\n{content}\n\n")

        return "".join(prompt_parts)

    def _generate_synthesis(self, prompt: str, mode: str) -> str:
        """Generate synthesis using LLM."""
        if not self.llm_provider:
            return "LLM synthesis not available (no provider configured). Returning structured chunks."

        try:
            # Adjust system prompt based on mode
            system_prompt = "You are a helpful assistant synthesizing information from multiple documents."
            if mode == "compare":
                system_prompt += " Focus on identifying similarities, differences, and conflicts."
            elif mode == "summarize":
                system_prompt += " Create a comprehensive unified summary."
            elif mode == "analyze":
                system_prompt += " Analyze patterns, themes, and deeper insights."

            response = self.llm_provider.create_message(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,  # Allow long synthesis
                temperature=0.3,  # Lower temperature for factual synthesis
                system=system_prompt,
                tools=[]
            )
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return f"Synthesis failed due to error: {str(e)}. Please review the retrieved chunks directly."


# ============================================================================
# Contextual Chunk Enricher Tool (NEW 2025-01) - Anthropic Contextual Retrieval
# ============================================================================


class ContextualChunkEnricherInput(ToolInput):
    """Input for contextual_chunk_enricher tool."""

    chunk_ids: List[str] = Field(
        ...,
        description="Chunk IDs to enrich with contextual information",
        min_items=1,
        max_items=50,
    )
    enrichment_mode: str = Field(
        "auto",
        description="Enrichment mode: 'auto' (detect best mode), 'document_summary', 'section_summary', 'both'",
    )
    include_metadata: bool = Field(
        True, description="Include metadata (page numbers, section titles) in enriched output"
    )


