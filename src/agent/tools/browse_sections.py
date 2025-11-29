"""
Browse Sections Tool - Hierarchical Section Navigation

Browse document section structure without search query.
Returns section hierarchy with optional summaries and chunk counts.

Use cases:
- Exploring document structure ("Jaká je struktura dokumentu?")
- Table of contents style navigation
- Finding sections to search within
"""

import logging
from typing import List, Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool

logger = logging.getLogger(__name__)


class BrowseSectionsInput(ToolInput):
    """Input for browse_sections tool."""

    document_id: str = Field(
        ...,
        description="Document ID to browse sections of",
    )
    parent_section_path: Optional[str] = Field(
        None,
        description=(
            "Parent section path to list children of. "
            "Use None for top-level sections, or e.g., 'Chapter 3' to drill down."
        ),
    )
    max_depth: int = Field(
        2,
        description="Maximum depth of section hierarchy to return (1=direct children only)",
        ge=1,
        le=5,
    )
    include_summaries: bool = Field(
        False,
        description="Include section content summaries (increases response size)",
    )
    sort_by: str = Field(
        "path",
        description="Sort sections by: 'path' (hierarchical), 'page' (document order), 'size' (chunk count)",
    )


@register_tool
class BrowseSectionsTool(BaseTool):
    """
    Browse document section hierarchy without search.

    This tool enables hierarchical navigation of document structure:
    - List top-level sections (chapters)
    - Drill down into subsections
    - Get section metadata (page numbers, chunk counts)

    Use for:
    - Understanding document structure
    - Building table of contents
    - Finding sections before searching within them
    """

    name = "browse_sections"
    description = "Browse document section hierarchy/TOC without search."
    detailed_help = """
    Section Navigation Tool

    **When to use:**
    - "Jaká je struktura tohoto dokumentu?"
    - "Co obsahuje dokument X?" (structure overview)
    - Finding which section to search in

    **Workflow:**
    1. browse_sections(document_id, max_depth=1) - see top-level chapters
    2. browse_sections(document_id, parent_section_path="Chapter 3") - drill down
    3. section_search(query, section_path_prefix="Chapter 3 > ") - search within

    **Output includes:**
    - section_id, section_title, section_path
    - section_level (1=top, 2=sub, etc.)
    - page_number (if available)
    - chunk_count (number of chunks in section)
    - has_children, children_count
    """

    input_schema = BrowseSectionsInput

    def execute_impl(
        self,
        document_id: str,
        parent_section_path: Optional[str] = None,
        max_depth: int = 2,
        include_summaries: bool = False,
        sort_by: str = "path",
    ) -> ToolResult:
        """
        Execute section browsing.

        Args:
            document_id: Document to browse
            parent_section_path: Parent path (None = top-level)
            max_depth: How deep to traverse
            include_summaries: Include section text
            sort_by: Sort order

        Returns:
            ToolResult with section hierarchy
        """
        logger.info(f"Browse sections: doc={document_id}, parent={parent_section_path}")

        try:
            # Get all Layer 2 sections for this document
            all_sections = self._get_document_sections(document_id)

            if not all_sections:
                return ToolResult(
                    success=True,
                    data={
                        "document_id": document_id,
                        "sections": [],
                        "total_sections": 0,
                        "message": f"No sections found for document '{document_id}'",
                    },
                    metadata={"found": False, "document_id": document_id},
                )

            # Filter to requested parent level
            if parent_section_path:
                # Get direct children of the parent
                target_sections = self._get_children_of_path(
                    all_sections, parent_section_path
                )
            else:
                # Get top-level sections only (level 1)
                target_sections = [
                    s for s in all_sections
                    if s.get("section_level", 0) == 1
                ]

            # Build tree structure with max_depth
            tree = self._build_tree(
                target_sections,
                all_sections,
                parent_section_path,
                max_depth,
                current_depth=1,
            )

            # Add chunk counts from Layer 3
            self._add_chunk_counts(tree)

            # Sort
            tree = self._sort_sections(tree, sort_by)

            # Add summaries if requested
            if include_summaries:
                self._add_summaries(tree, all_sections)

            return ToolResult(
                success=True,
                data={
                    "document_id": document_id,
                    "parent_path": parent_section_path,
                    "sections": tree,
                    "total_sections": len(all_sections),
                    "returned_sections": self._count_nodes(tree),
                },
                metadata={
                    "document_id": document_id,
                    "parent_path": parent_section_path,
                    "max_depth": max_depth,
                },
            )

        except Exception as e:
            logger.error(f"Browse sections error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to browse sections: {type(e).__name__}: {e}",
            )

    def _get_document_sections(self, document_id: str) -> List[dict]:
        """Get all Layer 2 sections for a document."""
        try:
            layer2_meta = getattr(self.vector_store, "metadata_layer2", [])
            return [
                s for s in layer2_meta
                if s.get("document_id") == document_id
            ]
        except Exception as e:
            logger.error(f"Failed to get Layer 2 metadata: {e}")
            return []

    def _get_children_of_path(
        self, all_sections: List[dict], parent_path: str
    ) -> List[dict]:
        """Get direct children of a parent section path."""
        # Find parent's level
        parent_level = None
        for s in all_sections:
            if s.get("section_path") == parent_path:
                parent_level = s.get("section_level", 0)
                break

        if parent_level is None:
            # Parent not found, try to infer from path
            parent_level = parent_path.count(" > ")

        # Find direct children (path starts with parent + " > " and level is parent + 1)
        children = []
        for s in all_sections:
            section_path = s.get("section_path") or ""
            section_level = s.get("section_level") or 0

            # Check if this is a direct child
            if section_path.startswith(parent_path + " > "):
                # Check it's exactly one level deeper
                if section_level == parent_level + 1:
                    children.append(s)

        return children

    def _build_tree(
        self,
        sections: List[dict],
        all_sections: List[dict],
        parent_path: Optional[str],
        max_depth: int,
        current_depth: int,
    ) -> List[dict]:
        """Build hierarchical tree from flat section list."""
        result = []

        for s in sections:
            section_path = s.get("section_path") or ""
            section_level = s.get("section_level") or 0

            node = {
                "section_id": s.get("section_id"),
                "section_title": s.get("section_title"),
                "section_path": section_path,
                "section_level": section_level,
                "page_number": s.get("page_number"),
            }

            # Find children
            children = [
                c for c in all_sections
                if (c.get("section_path") or "").startswith(section_path + " > ")
                and c.get("section_level") == section_level + 1
            ]

            node["has_children"] = len(children) > 0
            node["children_count"] = len(children)

            # Recurse if within depth limit
            if children and current_depth < max_depth:
                node["children"] = self._build_tree(
                    children,
                    all_sections,
                    section_path,
                    max_depth,
                    current_depth + 1,
                )

            result.append(node)

        return result

    def _add_chunk_counts(self, tree: List[dict]) -> None:
        """Add chunk counts from Layer 3 metadata to tree nodes."""
        try:
            layer3_meta = getattr(self.vector_store, "metadata_layer3", [])

            def add_counts(nodes: List[dict]) -> None:
                for node in nodes:
                    section_id = node.get("section_id")
                    if section_id:
                        count = sum(
                            1 for c in layer3_meta
                            if c.get("section_id") == section_id
                        )
                        node["chunk_count"] = count
                    else:
                        node["chunk_count"] = 0

                    if "children" in node:
                        add_counts(node["children"])

            add_counts(tree)

        except Exception as e:
            logger.warning(f"Failed to add chunk counts: {e}")

    def _sort_sections(self, tree: List[dict], sort_by: str) -> List[dict]:
        """Sort sections by specified field."""
        if sort_by == "page":
            tree.sort(key=lambda x: x.get("page_number") or 0)
        elif sort_by == "size":
            tree.sort(key=lambda x: x.get("chunk_count") or 0, reverse=True)
        else:  # path (default)
            tree.sort(key=lambda x: x.get("section_path") or "")

        # Sort children recursively
        for node in tree:
            if "children" in node:
                node["children"] = self._sort_sections(node["children"], sort_by)

        return tree

    def _add_summaries(self, tree: List[dict], all_sections: List[dict]) -> None:
        """Add content summaries to tree nodes."""
        id_to_content = {
            s.get("section_id"): (s.get("content") or "")[:500]
            for s in all_sections
        }

        def add_content(nodes: List[dict]) -> None:
            for node in nodes:
                section_id = node.get("section_id")
                if section_id:
                    node["summary"] = id_to_content.get(section_id, "")
                if "children" in node:
                    add_content(node["children"])

        add_content(tree)

    def _count_nodes(self, tree: List[dict]) -> int:
        """Count total nodes in tree."""
        count = len(tree)
        for node in tree:
            if "children" in node:
                count += self._count_nodes(node["children"])
        return count
