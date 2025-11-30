"""
Graphiti Search Tool for Agentic Knowledge Graph Queries.

Provides temporal knowledge graph search capabilities for the multi-agent system.
The LLM agent autonomously decides when to use this tool vs. vector search.

When to use graphiti_search (agent decides):
- Entity-focused queries: "Who regulates facility X?"
- Relationship queries: "What connects regulation A to requirement B?"
- Temporal queries: "What was the dose limit in 2020?"
- Historical queries: "How has this regulation evolved?"

When to use vector search instead:
- Semantic similarity: "Find documents about waste management"
- Content extraction: "What does documentation say about cooling?"
- Unknown entities: "What safety systems are mentioned?"

Search Modes:
- semantic: Hybrid semantic + BM25 search (default)
- entity_lookup: Find entity and all relationships
- temporal_query: Point-in-time fact retrieval
- entity_evolution: Track entity changes over time
- fact_timeline: Timeline of facts about topic
- path_finder: Find paths between two entities
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from src.agent.tools._base import BaseTool, ToolInput, ToolResult
from src.agent.tools._registry import register_tool

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT SCHEMA
# =============================================================================


class GraphitiSearchInput(ToolInput):
    """
    Input schema for graphiti_search tool.

    The agent uses this schema to understand available search parameters.
    """

    query: str = Field(
        ...,
        description="Search query - entity name, relationship, or fact to find",
        min_length=1,
        max_length=1000,
    )
    mode: Literal[
        "semantic",
        "entity_lookup",
        "temporal_query",
        "entity_evolution",
        "fact_timeline",
        "path_finder",
    ] = Field(
        default="semantic",
        description=(
            "Search mode: "
            "'semantic' for hybrid search, "
            "'entity_lookup' to find entity and relationships, "
            "'temporal_query' for point-in-time facts, "
            "'entity_evolution' to track changes, "
            "'fact_timeline' for chronological facts, "
            "'path_finder' to find connection paths"
        ),
    )
    valid_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp for point-in-time query (e.g., '2020-01-01T00:00:00Z')",
    )
    valid_after: Optional[str] = Field(
        default=None,
        description="Filter facts valid after this ISO timestamp",
    )
    valid_before: Optional[str] = Field(
        default=None,
        description="Filter facts valid before this ISO timestamp",
    )
    entity_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by entity types (e.g., ['Regulation', 'Organization'])",
    )
    num_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return",
    )
    group_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter by document IDs (group_ids in Graphiti)",
    )
    target_entity: Optional[str] = Field(
        default=None,
        description="Target entity name for path_finder mode",
    )


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class GraphitiSearchResult:
    """Result from Graphiti search operation."""

    facts: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    query: str = ""
    mode: str = "semantic"
    processing_time_ms: float = 0.0
    temporal_filter: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "facts": self.facts,
            "entities": self.entities,
            "total_results": self.total_results,
            "query": self.query,
            "mode": self.mode,
            "processing_time_ms": self.processing_time_ms,
            "temporal_filter": self.temporal_filter,
        }


# =============================================================================
# GRAPHITI SEARCH TOOL
# =============================================================================


@register_tool
class GraphitiSearchTool(BaseTool):
    """
    Temporal knowledge graph search using Graphiti.

    **When to use this tool (agent decides):**
    - Entity-focused queries: "Who regulates facility X?"
    - Relationship queries: "What connects regulation A to requirement B?"
    - Temporal queries: "What was the dose limit in 2020?"
    - Historical queries: "How has this regulation evolved?"

    **When to use vector search instead:**
    - Semantic similarity: "Find documents about waste management"
    - Content extraction: "What does documentation say about cooling?"
    - Unknown entities: "What safety systems are mentioned?"

    **Modes:**
    - semantic: Hybrid semantic + BM25 search (default)
    - entity_lookup: Find entity and all relationships
    - temporal_query: Point-in-time fact retrieval
    - entity_evolution: Track entity changes over time
    - fact_timeline: Timeline of facts about topic
    - path_finder: Find paths between two entities
    """

    name: str = "graphiti_search"
    description: str = (
        "Knowledge graph search for entities, relationships, and temporal facts. "
        "Use for: who/what queries, entity connections, historical data."
    )
    input_schema = GraphitiSearchInput
    requires_graphiti: bool = True

    def __init__(self, *args, **kwargs):
        """Initialize GraphitiSearchTool."""
        super().__init__(*args, **kwargs)
        self._graphiti = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure Graphiti client is initialized."""
        if self._initialized:
            return

        try:
            from graphiti_core import Graphiti
        except ImportError as e:
            logger.error(f"graphiti-core not installed: {e}")
            raise RuntimeError(
                "graphiti-core required. Install with: uv add graphiti-core"
            ) from e

        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not neo4j_password:
            raise RuntimeError(
                "NEO4J_PASSWORD environment variable not set. "
                "Set it in .env file for Neo4j authentication."
            )

        try:
            self._graphiti = Graphiti(
                uri=neo4j_uri,
                user="neo4j",
                password=neo4j_password,
            )
            self._initialized = True
            logger.info(f"GraphitiSearchTool connected to {neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {neo4j_uri}: {e}", exc_info=True)
            raise RuntimeError(
                f"Cannot connect to Neo4j at {neo4j_uri}. "
                "Check NEO4J_URI, NEO4J_PASSWORD env vars and that Neo4j is running."
            ) from e

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string. Raises ValueError for invalid formats."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(
                f"Invalid datetime format '{dt_str}'. Use ISO format like '2020-01-01T00:00:00Z'"
            ) from e

    def execute_impl(
        self,
        query: str,
        mode: str = "semantic",
        valid_at: Optional[str] = None,
        valid_after: Optional[str] = None,
        valid_before: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None,
        target_entity: Optional[str] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute Graphiti search (sync wrapper for async implementation).

        Args:
            query: Search query
            mode: Search mode (semantic, entity_lookup, temporal_query, etc.)
            valid_at: Point-in-time timestamp (ISO format)
            valid_after: Filter facts valid after this timestamp
            valid_before: Filter facts valid before this timestamp
            entity_types: Filter by entity types
            num_results: Maximum results
            group_ids: Filter by document IDs
            target_entity: Target entity for path_finder mode

        Returns:
            ToolResult with facts and entities
        """
        import asyncio
        import concurrent.futures

        # Run async implementation in a way that works regardless of event loop state
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # We're in an async context - use ThreadPoolExecutor to run in separate thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._execute_async(
                        query=query,
                        mode=mode,
                        valid_at=valid_at,
                        valid_after=valid_after,
                        valid_before=valid_before,
                        entity_types=entity_types,
                        num_results=num_results,
                        group_ids=group_ids,
                        target_entity=target_entity,
                    ),
                )
                return future.result(timeout=60)  # 60s timeout for graph queries
        except RuntimeError:
            # No running event loop - we can use asyncio.run directly
            return asyncio.run(
                self._execute_async(
                    query=query,
                    mode=mode,
                    valid_at=valid_at,
                    valid_after=valid_after,
                    valid_before=valid_before,
                    entity_types=entity_types,
                    num_results=num_results,
                    group_ids=group_ids,
                    target_entity=target_entity,
                )
            )

    async def _execute_async(
        self,
        query: str,
        mode: str = "semantic",
        valid_at: Optional[str] = None,
        valid_after: Optional[str] = None,
        valid_before: Optional[str] = None,
        entity_types: Optional[List[str]] = None,
        num_results: int = 10,
        group_ids: Optional[List[str]] = None,
        target_entity: Optional[str] = None,
    ) -> ToolResult:
        """
        Async implementation of Graphiti search.

        This is the actual search logic, called by the sync wrapper.
        """
        start_time = datetime.now(timezone.utc)

        try:
            await self._ensure_initialized()

            # Parse temporal filters
            try:
                valid_at_dt = self._parse_datetime(valid_at)
                valid_after_dt = self._parse_datetime(valid_after)
                valid_before_dt = self._parse_datetime(valid_before)
            except ValueError as e:
                return ToolResult(success=False, data=None, error=str(e))

            # Execute search based on mode
            if mode == "semantic":
                result = await self._semantic_search(
                    query=query,
                    num_results=num_results,
                    group_ids=group_ids,
                    entity_types=entity_types,
                    valid_after=valid_after_dt,
                    valid_before=valid_before_dt,
                )
            elif mode == "entity_lookup":
                result = await self._entity_lookup(
                    query=query,
                    num_results=num_results,
                    group_ids=group_ids,
                )
            elif mode == "temporal_query":
                result = await self._temporal_query(
                    query=query,
                    valid_at=valid_at_dt or datetime.now(timezone.utc),
                    num_results=num_results,
                    group_ids=group_ids,
                )
            elif mode == "entity_evolution":
                result = await self._entity_evolution(
                    query=query,
                    num_results=num_results,
                    group_ids=group_ids,
                )
            elif mode == "fact_timeline":
                result = await self._fact_timeline(
                    query=query,
                    num_results=num_results,
                    group_ids=group_ids,
                )
            elif mode == "path_finder":
                if not target_entity:
                    return ToolResult(
                        success=False,
                        data=None,
                        error="path_finder mode requires target_entity parameter",
                    )
                result = await self._path_finder(
                    source_entity=query,
                    target_entity=target_entity,
                    num_results=num_results,
                    group_ids=group_ids,
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown search mode: {mode}",
                )

            # Check if search returned an error
            if result.get("error"):
                return ToolResult(
                    success=False,
                    data=None,
                    error=result["error"],
                )

            processing_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            # Build temporal filter info
            temporal_filter = None
            if valid_at or valid_after or valid_before:
                temporal_filter = {
                    "valid_at": valid_at,
                    "valid_after": valid_after,
                    "valid_before": valid_before,
                }

            search_result = GraphitiSearchResult(
                facts=result.get("facts", []),
                entities=result.get("entities", []),
                total_results=result.get("total", 0),
                query=query,
                mode=mode,
                processing_time_ms=processing_time_ms,
                temporal_filter=temporal_filter,
            )

            # Format output for agent
            output_lines = [
                f"**Graphiti Search Results** (mode: {mode})",
                f"Query: {query}",
                f"Found: {search_result.total_results} results",
                "",
            ]

            if search_result.facts:
                output_lines.append("**Facts:**")
                for i, fact in enumerate(search_result.facts[:10], 1):
                    fact_text = fact.get("fact", fact.get("name", "Unknown"))
                    valid_at_str = fact.get("valid_at", "")
                    output_lines.append(f"{i}. {fact_text}")
                    if valid_at_str:
                        output_lines.append(f"   Valid: {valid_at_str}")
                    if fact.get("source"):
                        output_lines.append(f"   Source: {fact['source']}")
                output_lines.append("")

            if search_result.entities:
                output_lines.append("**Entities:**")
                for entity in search_result.entities[:5]:
                    name = entity.get("name", "Unknown")
                    labels = entity.get("labels", [])
                    summary = entity.get("summary", "")[:100]
                    output_lines.append(f"- {name} ({', '.join(labels)})")
                    if summary:
                        output_lines.append(f"  {summary}")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                data=search_result.to_dict(),
            )

        except (ConnectionError, TimeoutError, RuntimeError) as e:
            # Expected connection/configuration errors
            logger.warning(f"Graphiti search connection error for query '{query}': {e}")
            return ToolResult(
                success=False,
                data=None,
                error=(
                    f"Knowledge graph search unavailable: {str(e)}. "
                    "Verify Neo4j is running (docker compose up neo4j) and "
                    "NEO4J_URI/NEO4J_PASSWORD are set in .env file."
                ),
            )
        except Exception as e:
            # Unexpected errors - log with traceback for debugging
            logger.error(f"Unexpected Graphiti search error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Internal error in Graphiti search: {str(e)}",
            )

    async def _semantic_search(
        self,
        query: str,
        num_results: int,
        group_ids: Optional[List[str]],
        entity_types: Optional[List[str]],
        valid_after: Optional[datetime],
        valid_before: Optional[datetime],
    ) -> Dict[str, Any]:
        """
        Hybrid semantic + BM25 search.

        Returns facts (edges) matching the query.
        """
        from graphiti_core.search.search_filters import SearchFilters

        # Build search filter
        search_filter = None
        if entity_types or valid_after or valid_before:
            search_filter = SearchFilters(
                entity_labels=entity_types,
                valid_after=valid_after,
                valid_before=valid_before,
            )

        # Execute search
        results = await self._graphiti.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results,
            search_filter=search_filter,
        )

        # Convert to dict format
        facts = []
        for edge in results:
            facts.append({
                "uuid": edge.uuid,
                "fact": edge.fact,
                "name": edge.name,
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                "source_uuid": edge.source_node_uuid,
                "target_uuid": edge.target_node_uuid,
                "episodes": len(edge.episodes) if hasattr(edge, "episodes") else 0,
            })

        return {
            "facts": facts,
            "entities": [],
            "total": len(facts),
        }

    async def _entity_lookup(
        self,
        query: str,
        num_results: int,
        group_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Find entity and all its relationships.

        Uses node search to find entity, then retrieves connected edges.
        """
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

        # Search for nodes matching the entity name
        node_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_config.limit = num_results

        results = await self._graphiti.search_(
            query=query,
            config=node_config,
            group_ids=group_ids,
        )

        entities = []
        for node in results.nodes:
            entities.append({
                "uuid": node.uuid,
                "name": node.name,
                "labels": node.labels,
                "summary": getattr(node, "summary", ""),
                "attributes": getattr(node, "attributes", {}),
            })

        # Also get related facts
        facts = []
        for edge in results.edges:
            facts.append({
                "uuid": edge.uuid,
                "fact": edge.fact,
                "name": edge.name,
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
            })

        return {
            "facts": facts,
            "entities": entities,
            "total": len(entities) + len(facts),
        }

    async def _temporal_query(
        self,
        query: str,
        valid_at: datetime,
        num_results: int,
        group_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Point-in-time fact retrieval.

        Returns facts that were valid at the specified timestamp.
        """
        from graphiti_core.search.search_filters import SearchFilters

        # Filter for facts valid at the specified time
        search_filter = SearchFilters(
            valid_after=None,  # No lower bound
            valid_before=valid_at,  # Must be valid before query time
        )

        results = await self._graphiti.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results * 2,  # Get more to filter
            search_filter=search_filter,
        )

        # Filter for facts that were valid at the specified time
        # (valid_at <= query_time AND (invalid_at is NULL OR invalid_at > query_time))
        valid_facts = []
        for edge in results:
            if edge.valid_at and edge.valid_at <= valid_at:
                if edge.invalid_at is None or edge.invalid_at > valid_at:
                    valid_facts.append({
                        "uuid": edge.uuid,
                        "fact": edge.fact,
                        "name": edge.name,
                        "valid_at": edge.valid_at.isoformat(),
                        "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                    })

        return {
            "facts": valid_facts[:num_results],
            "entities": [],
            "total": len(valid_facts),
        }

    async def _entity_evolution(
        self,
        query: str,
        num_results: int,
        group_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Track how an entity has changed over time.

        Returns chronological list of facts about the entity.
        """
        # Search for all facts mentioning the entity
        results = await self._graphiti.search(
            query=query,
            group_ids=group_ids,
            num_results=num_results * 3,  # Get more to sort
        )

        # Sort by valid_at to show evolution
        facts_with_time = []
        for edge in results:
            if edge.valid_at:
                facts_with_time.append({
                    "uuid": edge.uuid,
                    "fact": edge.fact,
                    "name": edge.name,
                    "valid_at": edge.valid_at.isoformat(),
                    "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                    "is_current": edge.invalid_at is None,
                })

        # Sort chronologically
        facts_with_time.sort(key=lambda x: x["valid_at"])

        return {
            "facts": facts_with_time[:num_results],
            "entities": [],
            "total": len(facts_with_time),
        }

    async def _fact_timeline(
        self,
        query: str,
        num_results: int,
        group_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Get timeline of facts about a topic.

        Similar to entity_evolution but for broader topics.
        """
        # Same implementation as entity_evolution for now
        return await self._entity_evolution(query, num_results, group_ids)

    async def _path_finder(
        self,
        source_entity: str,
        target_entity: str,
        num_results: int,
        group_ids: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Find paths connecting two entities.

        Uses graph traversal to find relationship chains.
        """
        # First, find both entities
        source_results = await self._graphiti.search(
            query=source_entity,
            group_ids=group_ids,
            num_results=3,
        )

        target_results = await self._graphiti.search(
            query=target_entity,
            group_ids=group_ids,
            num_results=3,
        )

        if not source_results or not target_results:
            error_msg = (
                f"Could not find entities for path_finder: "
                f"source='{source_entity}' ({'found' if source_results else 'NOT FOUND'}), "
                f"target='{target_entity}' ({'found' if target_results else 'NOT FOUND'})"
            )
            logger.warning(error_msg)
            return {
                "facts": [],
                "entities": [],
                "total": 0,
                "error": error_msg,
            }

        # Use center_node_uuid to find paths near source
        center_uuid = source_results[0].source_node_uuid

        # Search with graph distance reranking
        path_results = await self._graphiti.search(
            query=target_entity,
            center_node_uuid=center_uuid,
            group_ids=group_ids,
            num_results=num_results,
        )

        facts = []
        for edge in path_results:
            facts.append({
                "uuid": edge.uuid,
                "fact": edge.fact,
                "name": edge.name,
                "source_uuid": edge.source_node_uuid,
                "target_uuid": edge.target_node_uuid,
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
            })

        return {
            "facts": facts,
            "entities": [],
            "total": len(facts),
            "source_entity": source_entity,
            "target_entity": target_entity,
        }


# =============================================================================
# TOOL REGISTRATION
# =============================================================================

# Tool is registered via @register_tool decorator
# Instantiation happens at runtime with proper dependencies
