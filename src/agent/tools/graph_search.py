"""
Graph Search Tool

Auto-extracted and cleaned from tier2_advanced.py
"""

import logging
from typing import Any, Dict, List, Optional, Literal
from pydantic import Field

from ._base import BaseTool, ToolInput, ToolResult
from ._registry import register_tool
from ._utils import format_chunk_result, generate_citation, validate_k_parameter

logger = logging.getLogger(__name__)


class GraphSearchInput(ToolInput):
    """Input for unified graph_search tool with multiple modes."""

    mode: str = Field(
        ...,
        description=(
            "Search mode:\n"
            "- 'entity_mentions': Find chunks mentioning an entity (fast, ~300ms)\n"
            "- 'entity_details': Get entity details + relationships + mentions (medium, ~500ms)\n"
            "- 'relationships': Query relationships between entities (medium, ~400ms)\n"
            "- 'multi_hop': Multi-hop BFS graph traversal (slow, ~1-2s)"
        ),
    )

    entity_value: str = Field(
        ...,
        description=(
            "REQUIRED: Specific entity name to search for (e.g., 'GRI 306', 'GSSB', 'waste management'). "
            "This tool requires a concrete entity - it cannot search ALL entities at once. "
            "Use browse_entities first to find entities, then use graph_search on each one."
        ),
    )

    entity_type: Optional[str] = Field(
        None,
        description=(
            "Entity type filter (optional): 'standard', 'organization', 'date', 'clause', "
            "'topic', 'person', 'location', 'regulation', 'contract'"
        ),
    )

    k: int = Field(
        6,
        description="Maximum number of results to return",
        ge=1,
        le=50,
    )

    relationship_types: Optional[List[str]] = Field(
        None,
        description=(
            "Filter by relationship types (optional): 'superseded_by', 'references', 'issued_by', "
            "'effective_date', 'covers_topic', 'contains_clause', 'applies_to', 'part_of', etc."
        ),
    )

    direction: str = Field(
        "both",
        description="Relationship direction: 'outgoing' (from entity), 'incoming' (to entity), 'both'",
    )

    max_hops: int = Field(
        2,
        description="Maximum hops for multi-hop traversal (1-3)",
        ge=1,
        le=3,
    )

    min_confidence: float = Field(
        0.0,
        description="Minimum confidence score for entities/relationships (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    cross_document: bool = Field(
        True,
        description="Allow traversal across documents (True) or stay within same document (False)",
    )

    include_metadata: bool = Field(
        True,
        description="Include detailed entity/relationship metadata in results",
    )


@register_tool
class GraphSearchTool(BaseTool):
    """Unified knowledge graph search with multiple modes."""

    name = "graph_search"
    description = (
        "Search knowledge graph for ONE specific entity (e.g., 'GRI 306'). "
        "ALWAYS requires entity_value parameter - cannot search all entities at once. "
        "For bulk: use browse_entities first, then call this for each entity."
    )
    detailed_help = """
    Unified tool for searching the knowledge graph with 4 modes:

    **Mode 1: entity_mentions** ()
    Find all chunks mentioning a specific entity.
    Example: Find all mentions of "GRI 306" standard across documents.
    Use when: You know an entity and want to see where it's discussed.

    **Mode 2: entity_details** ()
    Get comprehensive entity information: properties, relationships, and chunk mentions.
    Example: Get full details about "GSSB" organization including what it issued, when, and where mentioned.
    Use when: You need complete information about a specific entity.

    **Mode 3: relationships** ()
    Query relationships for a SPECIFIC entity with filtering.
    Example: Find all "superseded_by" relationships for "GRI 306" standard.
    ⚠️  Requires entity_value: Cannot search ALL relationships at once - must specify one entity.
    For bulk: Use browse_entities first to get entity list, then call graph_search for each.
    Use when: You need to understand connections of a specific entity.

    **Mode 4: multi_hop** ()
    Multi-hop BFS traversal following relationships across the graph.
    Example: "What topics are covered by standards issued by GSSB?" (GSSB → issued_by → Standards → covers_topic → Topics)
    Use when: Complex queries requiring following chains of relationships.

    **Parameters:**
    - entity_value: REQUIRED - Specific entity to search (e.g., "GRI 306", "GSSB") ⚠️ CANNOT BE OMITTED
    - entity_type: Optional type filter (standard, organization, topic, etc.)
    - k: Max results to return (default: 6)
    - relationship_types: Filter relationships (e.g., ["superseded_by", "references"])
    - direction: "outgoing", "incoming", or "both" (default: both)
    - max_hops: For multi_hop mode, how many hops (1-3, default: 2)
    - min_confidence: Filter low-confidence extractions (0.0-1.0, default: 0.0)
    - cross_document: Allow cross-document traversal (default: True)
    - include_metadata: Include detailed metadata (default: True)

    **⚠️  IMPORTANT: This tool works on ONE entity at a time**
    To search multiple entities:
    1. Use browse_entities to get list of entities
    2. Loop: call graph_search for each entity individually

    **Entity Types:**
    standard, organization, date, clause, topic, person, location, regulation, contract

    **Relationship Types:**
    superseded_by, supersedes, references, referenced_by, issued_by, developed_by, published_by,
    effective_date, expiry_date, signed_on, covers_topic, contains_clause, applies_to,
    part_of, contains, mentioned_in, defined_in

    **Examples:**
    1. entity_mentions: {"mode": "entity_mentions", "entity_value": "GRI 306", "k": 10}
    2. entity_details: {"mode": "entity_details", "entity_value": "GSSB", "entity_type": "organization"}
    3. relationships: {"mode": "relationships", "entity_value": "GRI 306", "relationship_types": ["superseded_by", "supersedes"]}
    4. multi_hop: {"mode": "multi_hop", "entity_value": "GSSB", "relationship_types": ["issued_by", "covers_topic"], "max_hops": 2}

    **❌ WRONG - This will FAIL with validation error:**
    {"mode": "relationships", "relationship_types": ["superseded_by"]}  # Missing entity_value!

    **✅ CORRECT - For "find all regulations with supersession relationships":**
    Step 1: browse_entities(entity_type="standard", limit=50)
    Step 2: For each entity from step 1:
            graph_search(mode="relationships", entity_value=<entity.value>, relationship_types=["superseded_by"])
    """

    input_schema = GraphSearchInput
    requires_kg = True

    def execute_impl(
        self,
        mode: str,
        entity_value: str,
        entity_type: Optional[str] = None,
        k: int = 6,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        max_hops: int = 2,
        min_confidence: float = 0.0,
        cross_document: bool = True,
        include_metadata: bool = True,
    ) -> ToolResult:
        """Execute graph search based on mode."""

        # Validate knowledge graph availability
        if not self.knowledge_graph:
            return ToolResult(
                success=False,
                data=None,
                error="Knowledge graph not available. Run indexing with enable_knowledge_graph=True.",
                metadata={"mode": mode, "entity_value": entity_value},
            )

        # Validate mode
        valid_modes = ["entity_mentions", "entity_details", "relationships", "multi_hop"]
        if mode not in valid_modes:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}",
                metadata={"mode": mode},
            )

        logger.info(
            f"graph_search: mode={mode}, entity={entity_value}, type={entity_type}, k={k}"
        )

        # Dispatch to appropriate mode handler
        try:
            if mode == "entity_mentions":
                return self._entity_mentions_search(
                    entity_value, entity_type, k, min_confidence, include_metadata
                )
            elif mode == "entity_details":
                return self._entity_details_search(
                    entity_value, entity_type, direction, min_confidence, include_metadata
                )
            elif mode == "relationships":
                return self._relationships_search(
                    entity_value,
                    entity_type,
                    relationship_types,
                    direction,
                    k,
                    min_confidence,
                    include_metadata,
                )
            elif mode == "multi_hop":
                return self._multi_hop_bfs(
                    entity_value,
                    entity_type,
                    relationship_types,
                    max_hops,
                    k,
                    min_confidence,
                    cross_document,
                    include_metadata,
                )
        except Exception as e:
            logger.error(f"graph_search failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Graph search failed: {str(e)}",
                metadata={"mode": mode, "entity_value": entity_value},
            )

    def _find_entity(
        self, entity_value: str, entity_type: Optional[str] = None, min_confidence: float = 0.0
    ) -> Optional[object]:
        """Find entity in knowledge graph by value and optional type."""
        entity_value_lower = entity_value.lower().strip()

        # Search through all entities
        candidates = []
        for entity in self.knowledge_graph.entities:
            if entity.confidence < min_confidence:
                continue

            # Check type match if specified
            if entity_type:
                # Handle both EntityType enum (entity.type) and string (mocks)
                entity_type_value = entity.type if hasattr(entity.type, 'value') else entity.type
                if entity_type_value != entity_type:
                    continue

            # Check value match (normalized or original)
            # Handle None/missing values gracefully - use try/except for robustness
            try:
                # Ensure we have a string, not a Mock or other object
                norm_val = entity.normalized_value
                entity_norm_lower = norm_val.lower() if norm_val and isinstance(norm_val, str) else ""
            except (AttributeError, TypeError):
                entity_norm_lower = ""

            try:
                # Ensure we have a string, not a Mock or other object
                val = entity.value
                entity_val_lower = val.lower() if val and isinstance(val, str) else ""
            except (AttributeError, TypeError):
                entity_val_lower = ""

            # Match if query appears in either normalized or original value
            # Check that we have actual strings before doing substring matching
            if (entity_norm_lower and entity_value_lower in entity_norm_lower) or \
               (entity_val_lower and entity_value_lower in entity_val_lower):
                candidates.append(entity)

        # Return best match (highest confidence)
        if candidates:
            candidates.sort(key=lambda e: e.confidence, reverse=True)
            return candidates[0]

        return None

    def _get_entity_not_found_error(self, entity_value: str, entity_type: Optional[str] = None) -> str:
        """Generate helpful error message when entity is not found."""
        import re

        # Check if entity_value looks like a document ID (e.g., Sb_1997_18, BZ_VR1)
        doc_id_pattern = r'^[A-Z][a-z]?_\d{4}_\d+.*$|^[A-Z]{2,}_[A-Z0-9]+.*$'
        is_likely_doc_id = bool(re.match(doc_id_pattern, entity_value))

        error_msg = f"Entity '{entity_value}' not found in knowledge graph"
        if entity_type:
            error_msg += f" with type '{entity_type}'"

        if is_likely_doc_id:
            error_msg += (
                f". NOTE: '{entity_value}' looks like a document ID. "
                "graph_search expects entity names (e.g., 'GRI 306', 'GSSB', 'zákon č. 18/1997 Sb.'), "
                "not document IDs. Try using get_document_info() or search() instead."
            )
        else:
            # Suggest browsing entities to find correct name
            error_msg += (
                ". Use browse_entities() to list available entities, "
                "or try a different entity name/type."
            )

        return error_msg

    def _entity_mentions_search(
        self,
        entity_value: str,
        entity_type: Optional[str],
        k: int,
        min_confidence: float,
        include_metadata: bool,
    ) -> ToolResult:
        """Mode 1: Find chunks mentioning an entity."""

        # Find entity
        entity = self._find_entity(entity_value, entity_type, min_confidence)

        if not entity:
            return ToolResult(
                success=False,
                data=None,
                error=self._get_entity_not_found_error(entity_value, entity_type),
                metadata={"entity_value": entity_value, "entity_type": entity_type},
            )

        # Get chunk IDs mentioning this entity
        chunk_ids = list(entity.source_chunk_ids)[:k]

        if not chunk_ids:
            return ToolResult(
                success=True,
                data={
                    "entity": {
                        "id": entity.id,
                        "type": entity.type,
                        "value": entity.value,
                        "confidence": entity.confidence,
                    },
                    "chunks": [],
                    "count": 0,
                },
                metadata={"mode": "entity_mentions", "entity_id": entity.id},
            )

        # Retrieve actual chunks from vector store metadata
        chunks = []
        # Get Layer 3 metadata (same pattern as expand_context tool)
        layer3_chunks = []
        if hasattr(self.vector_store, "metadata_layer3"):
            layer3_chunks = self.vector_store.metadata_layer3
        elif hasattr(self.vector_store, "faiss_store"):
            layer3_chunks = self.vector_store.faiss_store.metadata_layer3

        for chunk_id in chunk_ids:
            # Find chunk in metadata
            for meta in layer3_chunks:
                if meta.get("chunk_id") == chunk_id:
                    formatted = format_chunk_result(meta, include_score=False)
                    chunks.append(formatted)
                    break

        # Build result
        result_data = {
            "entity": {
                "id": entity.id,
                "type": entity.type,
                "value": entity.value,
                "normalized_value": entity.normalized_value,
                "confidence": entity.confidence,
            },
            "chunks": chunks,
            "count": len(chunks),
        }

        if include_metadata:
            result_data["entity"]["metadata"] = entity.metadata
            result_data["entity"]["extraction_method"] = entity.extraction_method

        citations = [f"{c['document_id']}:{c['chunk_id']}" for c in chunks]

        return ToolResult(
            success=True,
            data=result_data,
            citations=citations,
            metadata={
                "mode": "entity_mentions",
                "entity_id": entity.id,
                "chunks_found": len(chunks),
            },
        )

    def _entity_details_search(
        self,
        entity_value: str,
        entity_type: Optional[str],
        direction: str,
        min_confidence: float,
        include_metadata: bool,
    ) -> ToolResult:
        """Mode 2: Get comprehensive entity details."""

        # Find entity
        entity = self._find_entity(entity_value, entity_type, min_confidence)

        if not entity:
            return ToolResult(
                success=False,
                data=None,
                error=self._get_entity_not_found_error(entity_value, entity_type),
                metadata={"entity_value": entity_value, "entity_type": entity_type},
            )

        # Get relationships
        if direction == "outgoing":
            relationships = self.knowledge_graph.get_outgoing_relationships(entity.id)
        elif direction == "incoming":
            relationships = self.knowledge_graph.get_incoming_relationships(entity.id)
        else:  # both
            relationships = self.knowledge_graph.get_relationships_for_entity(entity.id)

        # Filter by confidence
        relationships = [r for r in relationships if r.confidence >= min_confidence]

        # Format relationships
        formatted_relationships = []
        for rel in relationships:
            source = self.knowledge_graph.get_entity(rel.source_entity_id)
            target = self.knowledge_graph.get_entity(rel.target_entity_id)

            rel_data = {
                "type": rel.type.value,
                "source": {"id": source.id, "value": source.value, "type": source.type}
                if source
                else None,
                "target": {"id": target.id, "value": target.value, "type": target.type.value}
                if target
                else None,
                "confidence": rel.confidence,
            }

            if include_metadata:
                rel_data["evidence_text"] = rel.evidence_text
                rel_data["source_chunk_id"] = rel.source_chunk_id
                rel_data["properties"] = rel.properties

            formatted_relationships.append(rel_data)

        # Build result
        result_data = {
            "entity": {
                "id": entity.id,
                "type": entity.type if hasattr(entity.type, 'value') else entity.type,
                "value": entity.value,
                "normalized_value": entity.normalized_value,
                "confidence": entity.confidence,
                "source_chunk_ids": list(entity.source_chunk_ids) if isinstance(entity.source_chunk_ids, set) else entity.source_chunk_ids,
                "first_mention_chunk_id": entity.first_mention_chunk_id,
            },
            "relationships": formatted_relationships,
            "relationship_count": len(formatted_relationships),
        }

        if include_metadata:
            result_data["entity"]["metadata"] = entity.metadata
            result_data["entity"]["document_id"] = entity.document_id
            result_data["entity"]["section_path"] = entity.section_path
            result_data["entity"]["extraction_method"] = entity.extraction_method

        return ToolResult(
            success=True,
            data=result_data,
            metadata={
                "mode": "entity_details",
                "entity_id": entity.id,
                "relationships_found": len(formatted_relationships),
            },
        )

    def _relationships_search(
        self,
        entity_value: str,
        entity_type: Optional[str],
        relationship_types: Optional[List[str]],
        direction: str,
        k: int,
        min_confidence: float,
        include_metadata: bool,
    ) -> ToolResult:
        """Mode 3: Query relationships for an entity."""

        # Find entity
        entity = self._find_entity(entity_value, entity_type, min_confidence)

        if not entity:
            return ToolResult(
                success=False,
                data=None,
                error=self._get_entity_not_found_error(entity_value, entity_type),
                metadata={"entity_value": entity_value, "entity_type": entity_type},
            )

        # Get relationships based on direction
        if direction == "outgoing":
            relationships = self.knowledge_graph.get_outgoing_relationships(entity.id)
        elif direction == "incoming":
            relationships = self.knowledge_graph.get_incoming_relationships(entity.id)
        else:  # both
            relationships = self.knowledge_graph.get_relationships_for_entity(entity.id)

        # Filter by confidence
        relationships = [r for r in relationships if r.confidence >= min_confidence]

        # Filter by relationship types if specified
        if relationship_types:
            rel_types_lower = [rt.lower() for rt in relationship_types]
            relationships = [r for r in relationships if r.type.value in rel_types_lower]

        # Limit to k
        relationships = relationships[:k]

        # Format relationships with full entity details
        formatted_relationships = []
        for rel in relationships:
            source = self.knowledge_graph.get_entity(rel.source_entity_id)
            target = self.knowledge_graph.get_entity(rel.target_entity_id)

            rel_data = {
                "type": rel.type.value,
                "source": {
                    "id": source.id,
                    "value": source.value,
                    "normalized_value": source.normalized_value,
                    "type": source.type,
                }
                if source
                else None,
                "target": {
                    "id": target.id,
                    "value": target.value,
                    "normalized_value": target.normalized_value,
                    "type": target.type.value,
                }
                if target
                else None,
                "confidence": rel.confidence,
            }

            if include_metadata:
                rel_data["evidence_text"] = rel.evidence_text
                rel_data["source_chunk_id"] = rel.source_chunk_id
                rel_data["properties"] = rel.properties
                rel_data["extraction_method"] = rel.extraction_method

            formatted_relationships.append(rel_data)

        result_data = {
            "entity": {
                "id": entity.id,
                "type": entity.type,  # entity.type is already a string
                "value": entity.value,
                "normalized_value": entity.normalized_value,
            },
            "relationships": formatted_relationships,
            "count": len(formatted_relationships),
            "filters": {
                "relationship_types": relationship_types,
                "direction": direction,
                "min_confidence": min_confidence,
            },
        }

        return ToolResult(
            success=True,
            data=result_data,
            metadata={
                "mode": "relationships",
                "entity_id": entity.id,
                "relationships_found": len(formatted_relationships),
            },
        )

    def _multi_hop_bfs(
        self,
        entity_value: str,
        entity_type: Optional[str],
        relationship_types: Optional[List[str]],
        max_hops: int,
        k: int,
        min_confidence: float,
        cross_document: bool,
        include_metadata: bool,
    ) -> ToolResult:
        """Mode 4: Multi-hop BFS graph traversal."""

        # Find starting entity
        start_entity = self._find_entity(entity_value, entity_type, min_confidence)

        if not start_entity:
            return ToolResult(
                success=False,
                data=None,
                error=f"Starting entity '{entity_value}' not found in knowledge graph"
                + (f" with type '{entity_type}'" if entity_type else ""),
                metadata={"entity_value": entity_value, "entity_type": entity_type},
            )

        # BFS traversal
        from collections import deque

        visited_entities = {start_entity.id}
        entity_queue = deque([(start_entity.id, 0)])  # (entity_id, hop_distance)
        entity_distances = {start_entity.id: 0}  # Track distances for scoring

        # Track discovered entities and relationships by hop
        entities_by_hop = {0: [start_entity]}  # hop 0 is the starting entity
        relationships_by_hop = {}  # hop N contains relationships from hop N-1 to hop N

        # Limit entities per hop to prevent explosion
        MAX_ENTITIES_PER_HOP = 20
        MAX_TOTAL_ENTITIES = 200

        total_entities = 1

        while entity_queue and total_entities < MAX_TOTAL_ENTITIES:
            current_entity_id, current_hop = entity_queue.popleft()

            # Don't expand beyond max_hops
            if current_hop >= max_hops:
                continue

            # Get outgoing relationships from current entity
            relationships = self.knowledge_graph.get_outgoing_relationships(current_entity_id)

            # Filter by confidence
            relationships = [r for r in relationships if r.confidence >= min_confidence]

            # Filter by relationship types if specified
            if relationship_types:
                rel_types_lower = [rt.lower() for rt in relationship_types]
                relationships = [r for r in relationships if r.type.value in rel_types_lower]

            # Process relationships
            hop_relationships = []
            hop_entities = []
            entities_added_this_hop = 0

            for rel in relationships:
                target_entity = self.knowledge_graph.get_entity(rel.target_entity_id)

                if not target_entity:
                    continue

                # Cross-document filtering
                if not cross_document:
                    source_doc = self.knowledge_graph.get_entity(current_entity_id).document_id
                    target_doc = target_entity.document_id
                    if source_doc != target_doc:
                        continue

                # Track this relationship
                hop_relationships.append(rel)

                # Add target entity if not visited
                if rel.target_entity_id not in visited_entities:
                    visited_entities.add(rel.target_entity_id)
                    entity_distances[rel.target_entity_id] = current_hop + 1

                    # Add to queue for further expansion
                    entity_queue.append((rel.target_entity_id, current_hop + 1))
                    hop_entities.append(target_entity)

                    entities_added_this_hop += 1
                    total_entities += 1

                    # Limit entities per hop
                    if entities_added_this_hop >= MAX_ENTITIES_PER_HOP:
                        break

            # Store discovered entities and relationships for this hop
            next_hop = current_hop + 1
            if hop_relationships:
                relationships_by_hop[next_hop] = relationships_by_hop.get(next_hop, []) + hop_relationships
            if hop_entities:
                entities_by_hop[next_hop] = entities_by_hop.get(next_hop, []) + hop_entities

        # Collect all chunks from discovered entities with distance-based scoring
        chunk_scores = {}  # chunk_id -> score
        chunk_entities = {}  # chunk_id -> list of entity values mentioning it

        for hop, entities in entities_by_hop.items():
            # Distance-based boost: closer entities get higher scores
            # hop 0: +0.5, hop 1: +0.3, hop 2: +0.15, hop 3: +0.05
            hop_boost = 0.5 / (2**hop)

            for entity in entities:
                for chunk_id in entity.source_chunk_ids:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + hop_boost

                    if chunk_id not in chunk_entities:
                        chunk_entities[chunk_id] = []
                    chunk_entities[chunk_id].append(
                        {"value": entity.value, "type": entity.type, "hop": hop}
                    )

        # Sort chunks by score (ascending: lowest confidence first, highest last)
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=False)[:k]

        # Get Layer 3 metadata for chunk retrieval
        layer3_chunks = []
        if hasattr(self.vector_store, "metadata_layer3"):
            layer3_chunks = self.vector_store.metadata_layer3
        elif hasattr(self.vector_store, "faiss_store"):
            layer3_chunks = self.vector_store.faiss_store.metadata_layer3

        # Build chunk lookup dictionary for O(1) access (avoid N+1 pattern)
        chunk_lookup = {meta.get("chunk_id"): meta for meta in layer3_chunks if meta.get("chunk_id")}

        # Retrieve actual chunks
        chunks = []
        for chunk_id, score in sorted_chunks:
            # Direct lookup instead of linear search
            chunk = chunk_lookup.get(chunk_id)

            if chunk:
                formatted = format_chunk_result(chunk, include_score=True)
                formatted["graph_score"] = round(score, 4)
                formatted["mentioned_entities"] = chunk_entities.get(chunk_id, [])
                chunks.append(formatted)

        # Format traversal summary
        traversal_summary = {
            "start_entity": {
                "id": start_entity.id,
                "value": start_entity.value,
                "type": start_entity.type,
            },
            "total_entities_discovered": len(visited_entities),
            "total_relationships_traversed": sum(len(rels) for rels in relationships_by_hop.values()),
            "max_hop_reached": max(entities_by_hop.keys()),
            "entities_by_hop": {hop: len(entities) for hop, entities in entities_by_hop.items()},
        }

        if include_metadata:
            # Add detailed traversal path
            traversal_summary["relationships_by_hop"] = {}
            for hop, rels in relationships_by_hop.items():
                traversal_summary["relationships_by_hop"][hop] = [
                    {
                        "type": r.type.value,
                        "source_id": r.source_entity_id,
                        "target_id": r.target_entity_id,
                        "confidence": r.confidence,
                    }
                    for r in rels[:10]  # Limit to first 10 per hop
                ]

        result_data = {
            "traversal": traversal_summary,
            "chunks": chunks,
            "count": len(chunks),
        }

        citations = [f"{c['document_id']}:{c['chunk_id']}" for c in chunks]

        return ToolResult(
            success=True,
            data=result_data,
            citations=citations,
            metadata={
                "mode": "multi_hop",
                "start_entity_id": start_entity.id,
                "entities_discovered": len(visited_entities),
                "chunks_found": len(chunks),
                "max_hops": max_hops,
            },
        )


# ============================================================================
# Multi-Document Synthesis Tool (NEW 2025-01)
# ============================================================================


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


