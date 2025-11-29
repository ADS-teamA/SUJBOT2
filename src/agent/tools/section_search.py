"""
Section Search Tool - Layer 2 Section-Level Search

Searches section summaries (Layer 2) with path and level filtering.
Uses HyDE + Expansion fusion for high-quality section retrieval.

Based on:
- Mix-of-Granularity (MoG) paper (2024)
- HiRAG: Hierarchical RAG (2024)
"""

import logging
from typing import List, Optional

from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation

logger = logging.getLogger(__name__)


class SectionSearchInput(ToolInput):
    """Input for section_search tool."""

    query: str = Field(
        ...,
        description="Natural language search query for finding relevant sections",
        min_length=1,
    )
    k: int = Field(
        5,
        description="Number of sections to return (default: 5, sections are larger than chunks)",
        ge=1,
        le=50,
    )
    document_filter: Optional[str] = Field(
        None,
        description="Filter to specific document ID",
    )
    section_path_prefix: Optional[str] = Field(
        None,
        description=(
            "Filter to sections starting with this path "
            "(e.g., 'Chapter 3 > ' to search only within Chapter 3)"
        ),
    )
    min_section_level: Optional[int] = Field(
        None,
        description="Minimum section hierarchy level (1=top-level chapters, 2=subsections)",
        ge=1,
        le=10,
    )
    max_section_level: Optional[int] = Field(
        None,
        description="Maximum section hierarchy level",
        ge=1,
        le=10,
    )


@register_tool
class SectionSearchTool(BaseTool):
    """
    Section-level search using Layer 2 embeddings.

    This tool searches document sections (not fine-grained chunks) and is optimal for:
    - Finding which sections discuss a topic
    - Getting section-level overviews
    - Understanding document structure
    - Answering "what sections cover X?" queries

    Uses HyDE + Expansion fusion algorithm (same as chunk search) but operates
    on section summaries stored in Layer 2.
    """

    name = "section_search"
    description = "Search sections (Layer 2). For overviews, chapters, section-level content."
    detailed_help = """
    Section-Level Search Tool (Layer 2)

    **When to use:**
    - Query asks about chapters/sections: "Co obsahuje kapitola 3?"
    - Requesting summaries: "Shrň sekci o bezpečnosti"
    - Understanding document structure: "Které sekce pojednávají o X?"
    - Overview questions with vague scope

    **When NOT to use:**
    - Need specific text passages (use 'search')
    - Need exact quotes or numbers (use 'search')
    - Looking for detailed technical data (use 'search')

    **Filtering options:**
    - document_filter: Limit to one document
    - section_path_prefix: Limit to sections under a path (e.g., "Chapter 3 > ")
    - min/max_section_level: Filter by hierarchy depth (1=top, 2=sub, etc.)

    **Algorithm:**
    Uses HyDE + Expansion fusion on Layer 2 section embeddings.
    """

    input_schema = SectionSearchInput

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fusion_retriever = None

    def _get_fusion_retriever(self):
        """Lazy initialization of FusionRetriever for Layer 2."""
        if self._fusion_retriever is None:
            try:
                from src.retrieval import DeepInfraClient, FusionRetriever, FusionConfig

                client = DeepInfraClient()

                fusion_config = FusionConfig(
                    hyde_weight=getattr(self.config, "hyde_weight", 0.6),
                    expansion_weight=getattr(self.config, "expansion_weight", 0.4),
                    default_k=getattr(self.config, "default_k", 5),
                )

                self._fusion_retriever = FusionRetriever(
                    client=client,
                    vector_store=self.vector_store,
                    config=fusion_config,
                )

                logger.info("FusionRetriever initialized for section_search")

            except ImportError as e:
                logger.error(f"Failed to import retrieval module: {e}")
                raise
            except ValueError as e:
                logger.error(f"FusionRetriever configuration error: {e}")
                raise

        return self._fusion_retriever

    def execute_impl(
        self,
        query: str,
        k: int = 5,
        document_filter: Optional[str] = None,
        section_path_prefix: Optional[str] = None,
        min_section_level: Optional[int] = None,
        max_section_level: Optional[int] = None,
    ) -> ToolResult:
        """
        Execute section-level search on Layer 2.

        Args:
            query: Natural language query
            k: Number of sections to return
            document_filter: Optional document ID filter
            section_path_prefix: Filter sections by path prefix
            min_section_level: Minimum hierarchy level
            max_section_level: Maximum hierarchy level

        Returns:
            ToolResult with formatted sections and citations
        """
        logger.info(f"Section search: '{query[:50]}...' (k={k})")

        try:
            retriever = self._get_fusion_retriever()

            # Over-fetch to allow for filtering
            fetch_k = k * 3 if (section_path_prefix or min_section_level or max_section_level) else k

            # Execute Layer 2 search using fusion retriever
            sections = retriever.search_layer2(
                query=query,
                k=fetch_k,
                document_filter=document_filter,
            )

            # Apply section path and level filters
            filtered_sections = self._apply_section_filters(
                sections,
                section_path_prefix=section_path_prefix,
                min_level=min_section_level,
                max_level=max_section_level,
            )

            # Limit to requested k
            filtered_sections = filtered_sections[:k]

            # Enrich with chunk counts from Layer 3 metadata
            enriched_sections = self._enrich_with_chunk_counts(filtered_sections)

            # Format results
            formatted = []
            for section in enriched_sections:
                result = format_chunk_result(section, detail_level="medium")
                # Add section-specific metadata
                result["section_level"] = section.get("section_level")
                result["section_path"] = section.get("section_path")
                result["chunk_count"] = section.get("chunk_count", 0)
                formatted.append(result)

            # Generate citations
            citations = [
                generate_citation(s, i + 1, format="inline")
                for i, s in enumerate(formatted)
            ]

            # Build metadata
            result_metadata = {
                "query": query,
                "k": k,
                "document_filter": document_filter,
                "section_path_prefix": section_path_prefix,
                "min_section_level": min_section_level,
                "max_section_level": max_section_level,
                "search_method": "layer2_hyde_expansion_fusion",
                "final_count": len(formatted),
                "granularity": "section",
            }

            return ToolResult(
                success=True,
                data=formatted,
                citations=citations,
                metadata=result_metadata,
            )

        except ValueError as e:
            logger.error(f"Section search configuration error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Configuration error: {e}. Check DEEPINFRA_API_KEY in .env",
            )

        except ConnectionError as e:
            logger.error(f"Database connection error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Database connection error: {e}",
            )

        except Exception as e:
            logger.error(f"Unexpected section search error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Section search failed: {type(e).__name__}: {e}",
            )

    def _apply_section_filters(
        self,
        sections: List[dict],
        section_path_prefix: Optional[str] = None,
        min_level: Optional[int] = None,
        max_level: Optional[int] = None,
    ) -> List[dict]:
        """
        Apply section path and level filters.

        Args:
            sections: List of section dicts from Layer 2 search
            section_path_prefix: Filter by path prefix
            min_level: Minimum section level
            max_level: Maximum section level

        Returns:
            Filtered list of sections
        """
        filtered = sections

        # Filter by path prefix
        if section_path_prefix:
            filtered = [
                s for s in filtered
                if (s.get("section_path") or "").startswith(section_path_prefix)
            ]

        # Filter by minimum level
        if min_level is not None:
            filtered = [
                s for s in filtered
                if (s.get("section_level") or 0) >= min_level
            ]

        # Filter by maximum level
        if max_level is not None:
            filtered = [
                s for s in filtered
                if (s.get("section_level") or 0) <= max_level
            ]

        return filtered

    def _enrich_with_chunk_counts(self, sections: List[dict]) -> List[dict]:
        """
        Add chunk_count from Layer 3 metadata to each section.

        This helps the LLM understand how much content each section contains.

        Args:
            sections: List of section dicts

        Returns:
            Sections with chunk_count field added
        """
        try:
            # Get Layer 3 metadata (cached property on vector store)
            layer3_meta = getattr(self.vector_store, "metadata_layer3", [])

            if not layer3_meta:
                # No Layer 3 metadata available, skip enrichment
                for section in sections:
                    section["chunk_count"] = 0
                return sections

            for section in sections:
                section_id = section.get("section_id")
                doc_id = section.get("document_id")

                if section_id and doc_id:
                    count = sum(
                        1 for c in layer3_meta
                        if c.get("section_id") == section_id
                        and c.get("document_id") == doc_id
                    )
                    section["chunk_count"] = count
                else:
                    section["chunk_count"] = 0

        except Exception as e:
            logger.warning(f"Failed to enrich sections with chunk counts: {e}")
            for section in sections:
                section["chunk_count"] = 0

        return sections
