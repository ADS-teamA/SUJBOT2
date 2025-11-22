"""
Expand Context Tool

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
class ExpandContextTool(BaseTool):
    """Expand chunk context."""

    name = "expand_context"
    description = "Expand chunk context"
    detailed_help = """
    Expand chunks with additional surrounding or related context.

    **Expansion modes:**
    - 'adjacent': Chunks immediately before/after (linear context)
    - 'section': All chunks from same section
    - 'similarity': Semantically similar chunks
    - 'hybrid': Combination of section + similarity

    **When to use:**
    - Need more context around a specific chunk
    - Answer requires surrounding text
    - Chunk alone is insufficient

    **Best practices:**
    - Use 'adjacent' for simple before/after context
    - Use 'section' to see full section around chunk
    - Use 'similarity' to find related content elsewhere
    - Use 'hybrid' for comprehensive context
    - Start with k=3, increase if needed

    **Method:** Metadata lookup + embeddings (similarity mode)
    
    """

    input_schema = ExpandContextInput

    def execute_impl(self, chunk_ids: List[str], expansion_mode: str, k: int = 3) -> ToolResult:
        try:
            # Get Layer 3 metadata
            layer3_chunks = []
            if hasattr(self.vector_store, "metadata_layer3"):
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            # Find target chunks
            target_chunks = []
            for chunk_id in chunk_ids:
                for meta in layer3_chunks:
                    if meta.get("chunk_id") == chunk_id:
                        target_chunks.append(meta)
                        break

            if not target_chunks:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"No chunks found for IDs: {chunk_ids}",
                    metadata={"chunk_ids": chunk_ids},
                )

            expanded_results = []

            for target_chunk in target_chunks:
                expansion = {
                    "target_chunk": format_chunk_result(target_chunk),
                    "expanded_chunks": [],
                }

                if expansion_mode == "adjacent":
                    # Get adjacent chunks (before/after)
                    context_chunks = self._expand_adjacent(target_chunk, layer3_chunks, k=k)
                    expansion["expanded_chunks"] = context_chunks
                    expansion["context_before"] = [
                        c for c in context_chunks if c.get("position") == "before"
                    ]
                    expansion["context_after"] = [
                        c for c in context_chunks if c.get("position") == "after"
                    ]

                elif expansion_mode == "section":
                    # Get chunks from same section
                    section_chunks = self._expand_by_section(target_chunk, layer3_chunks, k=k)
                    expansion["expanded_chunks"] = section_chunks

                elif expansion_mode == "similarity":
                    # Get semantically similar chunks
                    try:
                        similarity_chunks = self._expand_by_similarity(target_chunk, k=k)
                        expansion["expanded_chunks"] = similarity_chunks
                    except Exception as e:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Similarity expansion failed: {e}",
                        )

                elif expansion_mode == "hybrid":
                    # Combine section + similarity
                    section_chunks = self._expand_by_section(target_chunk, layer3_chunks, k=k // 2)
                    expansion["expanded_chunks"].extend(section_chunks)

                    try:
                        similarity_chunks = self._expand_by_similarity(target_chunk, k=k // 2)
                        expansion["expanded_chunks"].extend(similarity_chunks)
                    except Exception as e:
                        expansion["expansion_warning"] = f"Similarity expansion failed: {str(e)}"

                    # Remove duplicates
                    seen_ids = {target_chunk.get("chunk_id")}
                    unique_expanded = []
                    for chunk in expansion["expanded_chunks"]:
                        chunk_id = chunk.get("chunk_id")
                        if chunk_id not in seen_ids:
                            seen_ids.add(chunk_id)
                            unique_expanded.append(chunk)
                    expansion["expanded_chunks"] = unique_expanded

                else:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Invalid expansion_mode: {expansion_mode}. Must be 'adjacent', 'section', 'similarity', or 'hybrid'",
                    )

                expansion["expansion_count"] = len(expansion["expanded_chunks"])
                expanded_results.append(expansion)

            # Collect citations
            citations = []
            for result in expanded_results:
                doc_id = result["target_chunk"].get("document_id", "Unknown")
                if doc_id not in citations:
                    citations.append(doc_id)

            return ToolResult(
                success=True,
                data={"expansions": expanded_results, "expansion_mode": expansion_mode},
                citations=citations,
                metadata={
                    "chunk_count": len(target_chunks),
                    "expansion_mode": expansion_mode,
                },
            )

        except Exception as e:
            logger.error(f"Expand context failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))

    def _expand_adjacent(self, target_chunk: Dict, all_chunks: List[Dict], k: int) -> List[Dict]:
        """Get adjacent chunks (before/after)."""
        document_id = target_chunk.get("document_id")
        section_id = target_chunk.get("section_id")
        chunk_id = target_chunk.get("chunk_id")

        # Find all chunks from same section
        same_section = [
            (i, chunk)
            for i, chunk in enumerate(all_chunks)
            if chunk.get("document_id") == document_id and chunk.get("section_id") == section_id
        ]

        # Sort by chunk_id
        same_section.sort(key=lambda x: x[1].get("chunk_id", ""))

        # Find target position
        target_position = None
        for pos, (_, chunk) in enumerate(same_section):
            if chunk.get("chunk_id") == chunk_id:
                target_position = pos
                break

        if target_position is None:
            return []

        # Get context_window from config (default: 2)
        context_window = k

        # Extract context chunks
        start_pos = max(0, target_position - context_window)
        end_pos = min(len(same_section), target_position + context_window + 1)

        context_chunks = []

        # Before chunks
        for i in range(start_pos, target_position):
            chunk = format_chunk_result(same_section[i][1])
            chunk["position"] = "before"
            context_chunks.append(chunk)

        # After chunks
        for i in range(target_position + 1, end_pos):
            chunk = format_chunk_result(same_section[i][1])
            chunk["position"] = "after"
            context_chunks.append(chunk)

        return context_chunks

    def _expand_by_section(self, target_chunk: Dict, all_chunks: List[Dict], k: int) -> List[Dict]:
        """Expand by finding neighboring chunks from same section."""
        try:
            document_id = target_chunk.get("document_id")
            section_id = target_chunk.get("section_id")
            chunk_id = target_chunk.get("chunk_id")

            if not document_id or not section_id:
                return []

            # Find chunks from same section
            same_section = [
                chunk
                for chunk in all_chunks
                if isinstance(chunk, dict)
                and chunk.get("document_id") == document_id
                and chunk.get("section_id") == section_id
                and chunk.get("chunk_id") != chunk_id
            ]

            # Sort by chunk_id
            same_section.sort(key=lambda x: x.get("chunk_id", ""))

            return [format_chunk_result(chunk) for chunk in same_section[:k]]

        except Exception as e:
            logger.error(f"Section-based expansion failed: {e}", exc_info=True)
            return []

    def _expand_by_similarity(self, target_chunk: Dict, k: int) -> List[Dict]:
        """Expand by finding semantically similar chunks."""
        # Validate embedder
        if (
            not self.embedder
            or not hasattr(self.embedder, "dimensions")
            or not self.embedder.dimensions
        ):
            raise ValueError("Embedder not available for similarity-based expansion")

        content = target_chunk.get("content", "")
        if not content:
            return []

        try:
            # Embed and search
            query_embedding = self.embedder.embed_texts([content])

            results = self.vector_store.hierarchical_search(
                query_text=content,
                query_embedding=query_embedding,
                k_layer3=k + 1,
            )

            chunks = results.get("layer3", [])

            # Filter out target chunk
            target_id = target_chunk.get("chunk_id")
            similar_chunks = [chunk for chunk in chunks if chunk.get("chunk_id") != target_id]

            return [format_chunk_result(chunk) for chunk in similar_chunks[:k]]

        except Exception as e:
            logger.error(f"Similarity expansion failed: {e}", exc_info=True)
            raise


# ----------------------------------------------------------------------------
# Browse Entities Tool
# ----------------------------------------------------------------------------


class BrowseEntitiesInput(ToolInput):
    """Input for browse_entities tool."""

    entity_type: Optional[str] = Field(
        None,
        description=(
            "Filter by entity type (e.g., 'regulation', 'standard', 'organization', "
            "'clause', 'topic', 'date'). Leave empty to see all types."
        ),
    )

    search_term: Optional[str] = Field(
        None,
        description=(
            "Filter entities by value substring (case-insensitive). "
            "Searches both entity.value and entity.normalized_value fields. "
            "Example: 'waste' matches 'Waste Management', 'waste disposal', etc."
        ),
    )

    min_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score (0.0-1.0). Default 0.0 shows all entities.",
    )

    limit: int = Field(
        20,
        ge=1,
        le=50,
        description="Maximum number of entities to return (max 50). Default 20.",
    )


