"""
Browse Entities Tool

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
class BrowseEntitiesTool(BaseTool):
    """
    Browse and list entities from the knowledge graph without knowing specific names.

    **Purpose**: Discover what entities exist in the knowledge graph by type, confidence,
    or search term. Useful for exploratory queries like "list all regulations" or
    "show me high-confidence standards about waste".

    **Tier 2** (Advanced, ): Direct Neo4j queries using indexed fields.

    **Use Cases**:
    - "List all regulations in the knowledge graph"
    - "Show me organizations with high confidence"
    - "Find entities related to 'waste management'"
    - "Browse standards about carbon emissions"

    **Complements graph_search**: Use browse_entities to discover entities, then
    graph_search to explore specific entities and their relationships.

    **Returns**: List of entities with:
    - Entity ID, type, value
    - Confidence score
    - Number of mentions (source chunks)
    - Document origin

    **Example**:
    ```python
    # List all regulations
    browse_entities(entity_type="regulation")

    # Find high-confidence standards
    browse_entities(entity_type="standard", min_confidence=0.85)

    # Search for waste-related entities
    browse_entities(search_term="waste", min_confidence=0.7)

    # Browse all entity types (top 20 by confidence)
    browse_entities(limit=20)
    ```
    """

    name = "browse_entities"
    description = (
        "Browse and list entities from the knowledge graph by type, confidence, or search term. "
        "Useful for discovering what entities exist without knowing specific names. "
        "Complements graph_search which requires a specific entity_value."
    )

    input_schema = BrowseEntitiesInput
    requires_kg = True

    def execute_impl(
        self,
        entity_type: Optional[str] = None,
        search_term: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> ToolResult:
        """
        Browse entities using GraphAdapter.find_entities() for efficient Neo4j queries.

        Args:
            entity_type: Filter by entity type (e.g., 'regulation', 'standard')
            search_term: Filter by value substring (case-insensitive)
            min_confidence: Minimum confidence score (0.0-1.0)
            limit: Maximum results to return (max 50)

        Returns:
            ToolResult with list of entities and metadata
        """
        start_time = datetime.now()

        # Validate knowledge graph availability
        if not self.knowledge_graph:
            return ToolResult(
                success=False,
                data=[],
                error="Knowledge graph not available. Cannot browse entities.",
                metadata={"requires": "knowledge_graph"},
            )

        # Validate that knowledge graph has find_entities method (GraphAdapter)
        if not hasattr(self.knowledge_graph, "find_entities"):
            return ToolResult(
                success=False,
                data=[],
                error=(
                    "Knowledge graph does not support find_entities(). "
                    "This tool requires GraphAdapter with Neo4j backend."
                ),
                metadata={"kg_type": type(self.knowledge_graph).__name__},
            )

        try:
            # Query Neo4j via GraphAdapter.find_entities()
            logger.info(
                f"Browsing entities: type={entity_type}, search={search_term}, "
                f"min_conf={min_confidence}, limit={limit}"
            )

            entities = self.knowledge_graph.find_entities(
                entity_type=entity_type,
                min_confidence=min_confidence,
                value_contains=search_term,
            )

            # Sort by confidence (descending) and limit results
            entities.sort(key=lambda e: e.confidence, reverse=True)
            entities = entities[:limit]

            if not entities:
                # Build helpful message about what was searched
                filters = []
                if entity_type:
                    filters.append(f"type='{entity_type}'")
                if search_term:
                    filters.append(f"search='{search_term}'")
                if min_confidence > 0.0:
                    filters.append(f"confidence≥{min_confidence}")

                filter_desc = " with " + ", ".join(filters) if filters else ""

                return ToolResult(
                    success=True,
                    data=[],
                    metadata={
                        "count": 0,
                        "filters": {
                            "entity_type": entity_type,
                            "search_term": search_term,
                            "min_confidence": min_confidence,
                        },
                        "message": f"No entities found{filter_desc}",
                    },
                )

            # Format results
            formatted_entities = []
            for entity in entities:
                entity_data = {
                    "id": entity.id,
                    "type": entity.type if hasattr(entity.type, "value") else str(entity.type),
                    "value": entity.value,
                    "normalized_value": entity.normalized_value,
                    "confidence": round(entity.confidence, 3),
                    "mentions": len(entity.source_chunk_ids),
                    "document_id": entity.document_id,
                }
                formatted_entities.append(entity_data)

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Build filter description for metadata
            filters_applied = []
            if entity_type:
                filters_applied.append(f"type={entity_type}")
            if search_term:
                filters_applied.append(f"search='{search_term}'")
            if min_confidence > 0.0:
                filters_applied.append(f"confidence≥{min_confidence}")

            return ToolResult(
                success=True,
                data=formatted_entities,
                metadata={
                    "count": len(formatted_entities),
                    "filters": {
                        "entity_type": entity_type,
                        "search_term": search_term,
                        "min_confidence": min_confidence,
                        "limit": limit,
                    },
                    "filters_description": ", ".join(filters_applied) if filters_applied else "none",
                    "sorted_by": "confidence (descending)",
                    "execution_time_ms": round(execution_time, 2),
                },
                citations=[],  # No document citations for entity browsing
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Browse entities failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=[],
                error=f"Failed to browse entities: {str(e)}",
                metadata={
                    "entity_type": entity_type,
                    "search_term": search_term,
                    "min_confidence": min_confidence,
                },
            )


# ----------------------------------------------------------------------------
# Cluster Search Tool
# ----------------------------------------------------------------------------


class ClusterSearchInput(ToolInput):
    """Input for cluster_search tool."""

    query: str = Field(
        ...,
        description="Search query to find relevant chunks",
    )

    cluster_ids: Optional[List[int]] = Field(
        None,
        description=(
            "Filter to specific cluster IDs (optional). "
            "If None, searches all clusters. "
            "Use get_stats tool first to see available clusters."
        ),
    )

    diversity_mode: bool = Field(
        False,
        description=(
            "Enable diversity mode: return max 1 result per cluster. "
            "Useful to avoid redundant results from the same topic."
        ),
    )

    k: int = Field(
        10,
        description="Maximum number of results to return",
        ge=1,
        le=50,
    )


