"""
Get Document List Tool

Lists all documents in the vector store for orchestrator routing decisions.
"""

import logging
from typing import List

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GetDocumentListInput(ToolInput):
    """Input for get_document_list tool (no parameters needed)."""
    pass


@register_tool
class GetDocumentListTool(BaseTool):
    """List all available documents in the corpus."""

    name = "get_document_list"
    description = "Get list of all documents in the vector store"
    detailed_help = """
    Returns a list of all document IDs and their summaries from the vector store.

    **Use cases:**
    - Orchestrator routing: understand what documents exist
    - Query scoping: check if relevant documents are indexed
    - Corpus overview: see available document coverage

    **Returns:**
    - document_ids: List of unique document identifiers
    - document_count: Total number of documents
    - summaries: Brief description of each document (if available)
    """

    input_schema = GetDocumentListInput
    requires_reranker = False

    def execute_impl(self) -> ToolResult:
        """
        Get list of all documents in vector store.

        Returns:
            ToolResult with document list and summaries
        """
        try:
            # Get document list from vector store
            document_ids: List[str] = []
            summaries: dict = {}

            # Try PostgreSQL adapter method first
            if hasattr(self.vector_store, 'get_document_list'):
                document_ids = self.vector_store.get_document_list() or []

            # Try to get summaries if available
            if hasattr(self.vector_store, 'get_document_summaries'):
                summaries = self.vector_store.get_document_summaries() or {}

            # Sort for consistent output
            document_ids = sorted(document_ids)

            # Build response with summaries
            documents = []
            for doc_id in document_ids:
                doc_info = {
                    "id": doc_id,
                    "summary": summaries.get(doc_id, "No summary available")
                }
                documents.append(doc_info)

            return ToolResult(
                success=True,
                data={
                    "document_count": len(document_ids),
                    "document_ids": document_ids,
                    "documents": documents,
                },
                metadata={
                    "source": "vector_store",
                    "has_summaries": bool(summaries),
                },
            )

        except AttributeError as e:
            logger.error(f"Vector store missing required method: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error="Vector store not properly initialized. Check DATABASE_URL configuration.",
            )
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Database connection failed: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Database connection failed: {e}. Verify DATABASE_URL in .env file.",
            )
        except Exception as e:
            logger.error(f"Unexpected error getting document list: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Internal error: {type(e).__name__}: {e}",
            )
