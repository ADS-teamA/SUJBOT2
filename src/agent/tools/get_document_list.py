"""
Get Document List Tool

Lists all indexed documents with their summaries.
"""

import logging

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GetDocumentListInput(ToolInput):
    """Input for get_document_list tool."""

    pass  # No parameters needed


@register_tool
class GetDocumentListTool(BaseTool):
    """List all indexed documents."""

    name = "get_document_list"
    description = "List all indexed documents"
    detailed_help = """
    Returns a list of all indexed documents with their summaries.

    **When to use:**
    - User asks "what documents are available?"
    - Need to discover corpus contents
    - Before document-specific queries

    **Data source:** Vector store metadata (Layer 1 - document level)
    """
    input_schema = GetDocumentListInput

    def execute_impl(self) -> ToolResult:
        # Extract document IDs and summaries from Layer 1 metadata
        documents_map = {}  # {doc_id: summary}

        # Try different approaches depending on store type
        if hasattr(self.vector_store, "metadata_layer1"):
            for meta in self.vector_store.metadata_layer1:
                doc_id = meta.get("document_id")
                summary = meta.get("content", "")  # Layer 1 content is the document summary
                if doc_id and doc_id not in documents_map:
                    # Only store first occurrence (all Layer 1 entries for same doc have same summary)
                    documents_map[doc_id] = summary
        elif hasattr(self.vector_store, "faiss_store"):
            for meta in self.vector_store.faiss_store.metadata_layer1:
                doc_id = meta.get("document_id")
                summary = meta.get("content", "")
                if doc_id and doc_id not in documents_map:
                    documents_map[doc_id] = summary

        # Fallback: If Layer 1 is empty (single-layer optimization), extract from Layer 3
        if not documents_map:
            # Try metadata_layer3 (direct FAISSVectorStore)
            if hasattr(self.vector_store, "metadata_layer3"):
                for meta in self.vector_store.metadata_layer3:
                    doc_id = meta.get("document_id")
                    if doc_id and doc_id not in documents_map:
                        # Layer 3 doesn't have document summaries, use document title or placeholder
                        doc_title = meta.get("document_title", doc_id)
                        documents_map[doc_id] = f"Privacy policy for {doc_title}"
            # Try HybridVectorStore wrapper
            elif hasattr(self.vector_store, "faiss_store") and hasattr(
                self.vector_store.faiss_store, "metadata_layer3"
            ):
                for meta in self.vector_store.faiss_store.metadata_layer3:
                    doc_id = meta.get("document_id")
                    if doc_id and doc_id not in documents_map:
                        doc_title = meta.get("document_title", doc_id)
                        documents_map[doc_id] = f"Privacy policy for {doc_title}"

        # Build list of document objects with id and summary
        document_list = [
            {"id": doc_id, "summary": summary} for doc_id, summary in sorted(documents_map.items())
        ]

        return ToolResult(
            success=True,
            data={"documents": document_list, "count": len(document_list)},
            metadata={"total_documents": len(document_list)},
        )
