"""
Similarity Search Tool

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
class SimilaritySearchTool(BaseTool):
    """Find similar chunks."""

    name = "similarity_search"
    description = "Find similar chunks"
    detailed_help = """
    Find semantically similar or related chunks based on embedding similarity.

    **Search modes:**
    - 'related': Semantically related content
    - 'similar': More content like this

    **When to use:**
    - Find content similar to a specific chunk
    - Explore related information
    - "Show me more like this"

    **Best practices:**
    - Requires chunk_id from previous search result
    - Use k=5-10 for best results
    - Set cross_document=false to search within same document only
    - Good for discovering related content

    **Method:** Dense embedding similarity (cosine)
    
    """

    input_schema = SimilaritySearchInput

    def execute_impl(
        self, chunk_id: str, search_mode: str, cross_document: bool = True, k: int = 6
    ) -> ToolResult:
        logger.warning(
            "SimilaritySearchTool is DEPRECATED. Use ExpandContextTool with mode='similarity' instead."
        )
        try:
            # Get Layer 3 metadata
            layer3_chunks = []
            if hasattr(self.vector_store, "metadata_layer3"):
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            # Find target chunk
            target_chunk = None
            for meta in layer3_chunks:
                if meta.get("chunk_id") == chunk_id:
                    target_chunk = meta
                    break

            if not target_chunk:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Chunk '{chunk_id}' not found",
                    metadata={"chunk_id": chunk_id},
                )

            # Get chunk content
            content = target_chunk.get("raw_content", target_chunk.get("content", ""))
            if not content:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Chunk '{chunk_id}' has no content",
                    metadata={"chunk_id": chunk_id},
                )

            # Validate embedder
            if (
                not self.embedder
                or not hasattr(self.embedder, "dimensions")
                or not self.embedder.dimensions
            ):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Embedder not properly initialized",
                    metadata={"chunk_id": chunk_id},
                )

            # Embed and search
            query_embedding = self.embedder.embed_texts([content])

            document_filter = None
            if not cross_document:
                document_filter = target_chunk.get("document_id")

            results = self.vector_store.hierarchical_search(
                query_text=content,
                query_embedding=query_embedding,
                k_layer3=k * 2 + 1,
                document_filter=document_filter,
                use_doc_filtering=not cross_document,
            )

            chunks = results.get("layer3", [])

            # Filter out target chunk
            similar_chunks = [chunk for chunk in chunks if chunk.get("chunk_id") != chunk_id]

            # Apply document filter if needed (redundant but safe)
            if not cross_document:
                target_doc_id = target_chunk.get("document_id")
                similar_chunks = [
                    chunk for chunk in similar_chunks if chunk.get("document_id") == target_doc_id
                ]

            # Rerank if available and mode is 'related'
            if search_mode == "related" and self.reranker and len(similar_chunks) > k:
                similar_chunks = self.reranker.rerank(content, similar_chunks, top_k=k)
            else:
                similar_chunks = similar_chunks[:k]

            if not similar_chunks:
                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "chunk_id": chunk_id,
                        "search_mode": search_mode,
                        "cross_document": cross_document,
                        "results_count": 0,
                    },
                )

            formatted = [format_chunk_result(chunk) for chunk in similar_chunks]
            citations = list(set(chunk.get("document_id", "Unknown") for chunk in similar_chunks))

            return ToolResult(
                success=True,
                data={
                    "target_chunk": format_chunk_result(target_chunk),
                    "similar_chunks": formatted,
                    "similarity_count": len(formatted),
                },
                citations=citations,
                metadata={
                    "source_chunk_id": chunk_id,
                    "search_mode": search_mode,
                    "cross_document": cross_document,
                    "results_count": len(formatted),
                    "deprecated": True,
                    "deprecation_message": "Use ExpandContextTool with mode='similarity' instead",
                },
            )

        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))


class ExpandContextInput(ToolInput):
    """Input for unified expand_context tool."""

    chunk_ids: List[str] = Field(..., description="List of chunk IDs to expand")
    expansion_mode: str = Field(
        ...,
        description="Expansion mode: 'adjacent' (before/after chunks), 'section' (same section), 'similarity' (semantically similar), 'hybrid' (section + similarity)",
    )
    k: int = Field(3, description="Number of additional chunks per input chunk", ge=1, le=10)


