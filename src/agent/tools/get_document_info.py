"""
Get Document Info Tool

Unified tool for retrieving document information (summary, metadata, sections).
"""

import logging
from typing import Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class GetDocumentInfoInput(ToolInput):
    """Input for unified get_document_info tool."""

    document_id: str = Field(..., description="Document ID")
    info_type: str = Field(
        ...,
        description="Type of information: 'summary' (high-level overview), 'metadata' (comprehensive stats), 'sections' (list all sections), 'section_details' (specific section info)",
    )
    section_id: Optional[str] = Field(
        None, description="Section ID (required for info_type='section_details')"
    )


@register_tool
class GetDocumentInfoTool(BaseTool):
    """Get document information."""

    name = "get_document_info"
    description = "Get document info/metadata"
    detailed_help = """
    Unified tool for retrieving document information with multiple info types:
    - 'summary': High-level document overview
    - 'metadata': Comprehensive stats (sections, chunks, source info)
    - 'sections': List all sections with titles and hierarchy
    - 'section_details': Detailed info about a specific section

    **When to use:**
    - Need document overview before detailed search
    - Want to understand document structure
    - Looking for specific section to search within

    **Best practices:**
    - Use 'summary' for quick overview
    - Use 'sections' to understand structure
    - Use 'metadata' for comprehensive stats
    - Combine with search (filter_type='section') to search within sections

    **Data source:** Vector store metadata (Layer 1/2/3)
    """
    input_schema = GetDocumentInfoInput

    def execute_impl(
        self, document_id: str, info_type: str, section_id: Optional[str] = None
    ) -> ToolResult:
        try:
            # Get layer metadata
            layer1_chunks = []
            layer2_chunks = []
            layer3_chunks = []

            if hasattr(self.vector_store, "metadata_layer1"):
                layer1_chunks = self.vector_store.metadata_layer1
                layer2_chunks = self.vector_store.metadata_layer2
                layer3_chunks = self.vector_store.metadata_layer3
            elif hasattr(self.vector_store, "faiss_store"):
                layer1_chunks = self.vector_store.faiss_store.metadata_layer1
                layer2_chunks = self.vector_store.faiss_store.metadata_layer2
                layer3_chunks = self.vector_store.faiss_store.metadata_layer3

            if info_type == "summary":
                # Get document summary (Layer 1)
                doc_summary = None
                for meta in layer1_chunks:
                    if meta.get("document_id") == document_id:
                        doc_summary = meta.get("content")
                        break

                if not doc_summary:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                return ToolResult(
                    success=True,
                    data={"document_id": document_id, "summary": doc_summary},
                    metadata={"document_id": document_id, "summary_length": len(doc_summary)},
                )

            elif info_type == "metadata":
                # Get comprehensive metadata (all layers)
                metadata = {"document_id": document_id}

                # Layer 1: Summary
                for meta in layer1_chunks:
                    if meta.get("document_id") == document_id:
                        metadata["summary"] = meta.get("content")
                        break

                # Layer 2: Sections
                sections = [
                    meta.get("section_title")
                    for meta in layer2_chunks
                    if meta.get("document_id") == document_id
                ]
                metadata["section_count"] = len(sections)
                metadata["sections"] = sections

                # Layer 3: Chunks
                chunk_count = sum(
                    1 for meta in layer3_chunks if meta.get("document_id") == document_id
                )
                metadata["chunk_count"] = chunk_count

                # Estimate document length
                total_chars = sum(
                    len(meta.get("content", ""))
                    for meta in layer3_chunks
                    if meta.get("document_id") == document_id
                )
                metadata["estimated_chars"] = total_chars
                metadata["estimated_words"] = total_chars // 5

                if not metadata.get("summary") and metadata["section_count"] == 0:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                return ToolResult(
                    success=True,
                    data=metadata,
                    metadata={"document_id": document_id, "total_sections": len(sections)},
                )

            elif info_type == "sections":
                # Get list of sections (Layer 2)
                sections = []
                for meta in layer2_chunks:
                    if meta.get("document_id") == document_id:
                        section_info = {
                            "section_id": meta.get("section_id"),
                            "section_title": meta.get("section_title"),
                        }
                        sections.append(section_info)

                if not sections:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={"document_id": document_id, "found": False},
                    )

                # Sort sections by section_id
                sections.sort(key=lambda x: x.get("section_id", ""))

                # Use adaptive formatter for dynamic section limits
                try:
                    from ._token_manager import get_adaptive_formatter

                    formatter = get_adaptive_formatter()
                    formatted_sections, format_metadata = formatter.format_sections_with_budget(
                        sections, include_summary=False
                    )

                    return ToolResult(
                        success=True,
                        data={
                            "document_id": document_id,
                            "sections": formatted_sections,
                            "count": format_metadata["returned_sections"],
                            "total_sections": format_metadata["total_sections"],
                            "truncated": format_metadata["truncated"],
                            "max_sections_allowed": format_metadata["max_sections_allowed"],
                        },
                        metadata={
                            "document_id": document_id,
                            "section_count": format_metadata["returned_sections"],
                            "total_sections": format_metadata["total_sections"],
                            "truncated": format_metadata["truncated"],
                        },
                    )

                except ImportError:
                    # Fallback
                    total_count = len(sections)
                    max_sections = 50
                    truncated = total_count > max_sections
                    sections = sections[:max_sections]

                    return ToolResult(
                        success=True,
                        data={
                            "document_id": document_id,
                            "sections": sections,
                            "count": len(sections),
                            "total_sections": total_count,
                            "truncated": truncated,
                        },
                        metadata={
                            "document_id": document_id,
                            "section_count": len(sections),
                            "total_sections": total_count,
                            "truncated": truncated,
                        },
                    )

            elif info_type == "section_details":
                # Get specific section details (Layer 2 + Layer 3)
                if not section_id:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="section_id is required for info_type='section_details'",
                    )

                # Find section in Layer 2
                section_data = None
                for meta in layer2_chunks:
                    if (
                        meta.get("document_id") == document_id
                        and meta.get("section_id") == section_id
                    ):
                        section_data = {
                            "document_id": document_id,
                            "section_id": section_id,
                            "section_title": meta.get("section_title"),
                            "section_path": meta.get("section_path"),
                            "summary": meta.get("content"),
                            "page_number": meta.get("page_number"),
                        }
                        break

                if not section_data:
                    return ToolResult(
                        success=True,
                        data=None,
                        metadata={
                            "document_id": document_id,
                            "section_id": section_id,
                            "found": False,
                        },
                    )

                # Get chunk count (Layer 3)
                chunk_count = sum(
                    1
                    for meta in layer3_chunks
                    if meta.get("document_id") == document_id
                    and meta.get("section_id") == section_id
                )
                section_data["chunk_count"] = chunk_count

                return ToolResult(
                    success=True,
                    data=section_data,
                    metadata={
                        "document_id": document_id,
                        "section_id": section_id,
                        "chunk_count": chunk_count,
                    },
                )

            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid info_type: {info_type}. Must be 'summary', 'metadata', 'sections', or 'section_details'",
                )

        except Exception as e:
            logger.error(f"Get document info failed: {e}", exc_info=True)
            return ToolResult(success=False, data=None, error=str(e))
