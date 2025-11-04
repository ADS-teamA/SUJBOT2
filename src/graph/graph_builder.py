"""
Graph builders for storing knowledge graphs.

Provides multiple backend implementations:
- Neo4jGraphBuilder: Production Neo4j database
- SimpleGraphBuilder: JSON-based in-memory graph for development
- NetworkXGraphBuilder: NetworkX for lightweight operations
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import GraphStorageConfig, Neo4jConfig
from .models import Entity, EntityType, KnowledgeGraph, Relationship, RelationshipType

logger = logging.getLogger(__name__)


class GraphBuilder(ABC):
    """
    Abstract base class for graph builders.

    Defines interface for storing and querying knowledge graphs.
    """

    @abstractmethod
    def add_entities(self, entities: List[Entity]) -> None:
        """Add entities to the graph."""
        pass

    @abstractmethod
    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships to the graph."""
        pass

    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        pass

    @abstractmethod
    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity."""
        pass

    @abstractmethod
    def export_to_knowledge_graph(self) -> KnowledgeGraph:
        """Export graph to KnowledgeGraph object."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save graph to file."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connections and cleanup."""
        pass


class SimpleGraphBuilder(GraphBuilder):
    """
    Simple in-memory graph builder using Python data structures.

    Stores graph in memory and exports to JSON. Fast for development and testing.
    """

    def __init__(self, config: GraphStorageConfig):
        """Initialize simple graph builder."""
        self.config = config

        # Storage
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}

        # Indexes
        self.entity_by_type: Dict[EntityType, Set[str]] = {}
        self.relationships_by_source: Dict[str, Set[str]] = {}
        self.relationships_by_target: Dict[str, Set[str]] = {}
        # Index for O(1) deduplication: (type, normalized_value) -> entity_id
        self.entity_by_normalized_value: Dict[tuple, str] = {}

        logger.info("Initialized SimpleGraphBuilder")

    def add_entities(self, entities: List[Entity]) -> None:
        """Add entities to the graph."""
        for entity in entities:
            # Store entity
            self.entities[entity.id] = entity

            # Index by type
            if entity.type not in self.entity_by_type:
                self.entity_by_type[entity.type] = set()
            self.entity_by_type[entity.type].add(entity.id)

            # Index by (type, normalized_value) for deduplication
            key = (entity.type, entity.normalized_value)
            self.entity_by_normalized_value[key] = entity.id

        logger.info(f"Added {len(entities)} entities to graph (total: {len(self.entities)})")

    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships to the graph."""
        for rel in relationships:
            # Store relationship
            self.relationships[rel.id] = rel

            # Index by source entity
            if rel.source_entity_id not in self.relationships_by_source:
                self.relationships_by_source[rel.source_entity_id] = set()
            self.relationships_by_source[rel.source_entity_id].add(rel.id)

            # Index by target entity
            if rel.target_entity_id not in self.relationships_by_target:
                self.relationships_by_target[rel.target_entity_id] = set()
            self.relationships_by_target[rel.target_entity_id].add(rel.id)

        logger.info(
            f"Added {len(relationships)} relationships to graph (total: {len(self.relationships)})"
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self.entity_by_type.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids]

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity (as source or target)."""
        rel_ids = set()

        # Get relationships where entity is source
        if entity_id in self.relationships_by_source:
            rel_ids.update(self.relationships_by_source[entity_id])

        # Get relationships where entity is target
        if entity_id in self.relationships_by_target:
            rel_ids.update(self.relationships_by_target[entity_id])

        return [self.relationships[rid] for rid in rel_ids]

    def get_outgoing_relationships(
        self, entity_id: str, rel_type: Optional[RelationshipType] = None
    ) -> List[Relationship]:
        """Get outgoing relationships from an entity."""
        rel_ids = self.relationships_by_source.get(entity_id, set())
        rels = [self.relationships[rid] for rid in rel_ids]

        if rel_type:
            rels = [r for r in rels if r.type == rel_type]

        return rels

    def get_incoming_relationships(
        self, entity_id: str, rel_type: Optional[RelationshipType] = None
    ) -> List[Relationship]:
        """Get incoming relationships to an entity."""
        rel_ids = self.relationships_by_target.get(entity_id, set())
        rels = [self.relationships[rid] for rid in rel_ids]

        if rel_type:
            rels = [r for r in rels if r.type == rel_type]

        return rels

    def get_neighbors(
        self, entity_id: str, rel_type: Optional[RelationshipType] = None
    ) -> List[Entity]:
        """Get neighbor entities connected by relationships."""
        neighbors = []

        # Outgoing relationships
        outgoing = self.get_outgoing_relationships(entity_id, rel_type)
        for rel in outgoing:
            target = self.get_entity(rel.target_entity_id)
            if target:
                neighbors.append(target)

        # Incoming relationships
        incoming = self.get_incoming_relationships(entity_id, rel_type)
        for rel in incoming:
            source = self.get_entity(rel.source_entity_id)
            if source:
                neighbors.append(source)

        return neighbors

    def export_to_knowledge_graph(self) -> KnowledgeGraph:
        """Export to KnowledgeGraph object."""
        kg = KnowledgeGraph(
            entities=list(self.entities.values()),
            relationships=list(self.relationships.values()),
        )

        # Compute statistics
        kg.compute_stats()

        return kg

    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        kg = self.export_to_knowledge_graph()

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        kg.save_json(path)
        logger.info(f"Saved graph to {path}")

    def close(self) -> None:
        """Cleanup (no-op for simple builder)."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        entity_type_counts = {}
        for entity_type, entity_ids in self.entity_by_type.items():
            entity_type_counts[entity_type.value] = len(entity_ids)

        relationship_type_counts = {}
        for rel in self.relationships.values():
            rel_type = rel.type.value
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1

        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_type_counts": entity_type_counts,
            "relationship_type_counts": relationship_type_counts,
        }

    def merge(self, other: "SimpleGraphBuilder") -> Dict[str, Any]:
        """
        Merge another graph into this one with entity deduplication.

        Deduplication strategy:
        - Exact match on (entity_type, normalized_value)
        - Merge source_chunk_ids for duplicate entities
        - Update relationship entity IDs after remapping

        Args:
            other: Graph to merge into this one

        Returns:
            Merge statistics with deduplication counts

        Example:
            >>> target_graph = SimpleGraphBuilder(config)
            >>> source_graph = SimpleGraphBuilder(config)
            >>> stats = target_graph.merge(source_graph)
            >>> print(f"Added {stats['entities_added']} entities")
        """
        from .deduplicator import EntityDeduplicator

        deduplicator = EntityDeduplicator()

        # Track statistics
        stats = {
            "entities_added": 0,
            "entities_deduplicated": 0,
            "relationships_added": 0,
            "entity_id_remapping": {},  # old_id -> new_id
        }

        # Phase 1: Merge entities with deduplication
        for entity in other.entities.values():
            # Check for duplicate
            duplicate_id = deduplicator.find_duplicate(entity, self)

            if duplicate_id:
                # Found duplicate - merge source_chunk_ids
                existing_entity = self.entities[duplicate_id]

                # Merge source_chunk_ids (deduplicate)
                merged_chunk_ids = list(
                    set(existing_entity.source_chunk_ids + entity.source_chunk_ids)
                )
                existing_entity.source_chunk_ids = merged_chunk_ids

                # Track remapping
                stats["entity_id_remapping"][entity.id] = duplicate_id
                stats["entities_deduplicated"] += 1

                logger.debug(
                    f"Deduplicated entity: {entity.type.value}='{entity.value}' "
                    f"({entity.id} -> {duplicate_id})"
                )
            else:
                # No duplicate - add entity
                self.entities[entity.id] = entity

                # Update indexes
                if entity.type not in self.entity_by_type:
                    self.entity_by_type[entity.type] = set()
                self.entity_by_type[entity.type].add(entity.id)

                # Update normalized_value index
                key = (entity.type, entity.normalized_value)
                self.entity_by_normalized_value[key] = entity.id

                stats["entities_added"] += 1

        # Phase 2: Merge relationships with entity ID remapping
        for rel in other.relationships.values():
            # Remap source entity ID if deduplicated
            source_id = stats["entity_id_remapping"].get(rel.source_entity_id, rel.source_entity_id)
            target_id = stats["entity_id_remapping"].get(rel.target_entity_id, rel.target_entity_id)

            # Update relationship entity IDs
            rel.source_entity_id = source_id
            rel.target_entity_id = target_id

            # Add relationship
            self.relationships[rel.id] = rel

            # Update indexes
            if source_id not in self.relationships_by_source:
                self.relationships_by_source[source_id] = set()
            self.relationships_by_source[source_id].add(rel.id)

            if target_id not in self.relationships_by_target:
                self.relationships_by_target[target_id] = set()
            self.relationships_by_target[target_id].add(rel.id)

            stats["relationships_added"] += 1

        logger.info(
            f"Graph merge complete: "
            f"+{stats['entities_added']} entities, "
            f"~{stats['entities_deduplicated']} deduplicated, "
            f"+{stats['relationships_added']} relationships"
        )

        return stats


class Neo4jGraphBuilder(GraphBuilder):
    """
    Neo4j graph database builder for production use.

    Stores entities as nodes and relationships as edges in Neo4j.
    Supports Cypher queries, batch operations, and automatic retry logic.

    Features:
    - Batch entity insertion (1000 per batch) for performance
    - Automatic retry on transient failures
    - Health checks before operations
    - Connection pooling with Neo4jManager
    """

    def __init__(self, config: GraphStorageConfig):
        """Initialize Neo4j graph builder with health check."""
        self.config = config

        if not self.config.neo4j_config:
            raise ValueError("Neo4j config required for Neo4jGraphBuilder")

        self.neo4j_config: Neo4jConfig = self.config.neo4j_config

        # Use Neo4jManager for connection pooling and retry logic
        from .neo4j_manager import Neo4jManager

        self.manager = Neo4jManager(self.neo4j_config)

        # Health check (fail fast if Neo4j unavailable)
        health = self.manager.health_check()
        if not health.get("healthy", False):
            raise ValueError(f"Neo4j health check failed: {health.get('error')}")

        # Create indexes and constraints
        if self.neo4j_config.create_indexes:
            self._create_indexes()

        # Initialize deduplication if enabled
        from .config import EntityDeduplicationConfig

        dedup_config = self.config.deduplication_config or EntityDeduplicationConfig()

        if dedup_config.enabled:
            from .neo4j_deduplicator import Neo4jDeduplicator

            self.neo4j_dedup = Neo4jDeduplicator(self.manager, dedup_config)

            # Create uniqueness constraints for deduplication
            if dedup_config.create_uniqueness_constraints:
                self.neo4j_dedup.create_uniqueness_constraints()

            logger.info(
                f"Deduplication enabled: "
                f"exact={dedup_config.exact_match_enabled}, "
                f"embeddings={dedup_config.use_embeddings}, "
                f"acronyms={dedup_config.use_acronym_expansion}"
            )
        else:
            self.neo4j_dedup = None
            logger.info("Deduplication disabled")

        # Initialize ID aliases dict for relationship remapping
        self._id_aliases: Dict[str, str] = {}

        logger.info(
            f"Initialized Neo4jGraphBuilder (uri={self.neo4j_config.uri}, "
            f"health check: {health['response_time_ms']:.0f}ms)"
        )

    def _create_indexes(self):
        """Create indexes and constraints for fast lookups."""
        # Create unique constraint on Entity.id
        self.manager.execute(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )

        # Create indexes on Entity properties
        self.manager.execute(
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)"
        )

        self.manager.execute(
            "CREATE INDEX entity_normalized_value_idx IF NOT EXISTS "
            "FOR (e:Entity) ON (e.normalized_value)"
        )

        logger.info("Created Neo4j indexes and constraints")

    def add_entities(self, entities: List[Entity]) -> None:
        """
        Add entities to Neo4j as nodes using batch operations.

        When deduplication is enabled, uses Neo4jDeduplicator for incremental
        duplicate detection and property merging.

        When deduplication is disabled, falls back to basic batch insertion.

        Processes entities in batches of 1000 for optimal performance.
        Uses UNWIND for bulk insertion (10-20x faster than individual inserts).
        """
        if not entities:
            logger.warning("No entities to add")
            return

        # Use deduplication if enabled
        if self.neo4j_dedup:
            logger.info(f"Adding {len(entities)} entities with deduplication...")
            stats = self.neo4j_dedup.add_entities_with_dedup(entities)

            # Store ID aliases for relationship remapping
            self._id_aliases = stats.get("id_aliases", {})
            if self._id_aliases:
                logger.info(f"Stored {len(self._id_aliases)} ID aliases for relationship remapping")

            logger.info(
                f"Deduplication complete: "
                f"added={stats['entities_added']}, "
                f"merged={stats['entities_merged']}, "
                f"failed={stats['entities_failed']}"
            )
            return

        # Fallback: Basic batch insertion without deduplication
        logger.info(f"Adding {len(entities)} entities without deduplication...")
        batch_size = 1000
        total_added = 0

        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]

            # Convert batch to list of dicts
            entities_data = [
                {
                    "id": e.id,
                    "type": e.type.value,
                    "value": e.value,
                    "normalized_value": e.normalized_value,
                    "confidence": e.confidence,
                    "source_chunk_ids": e.source_chunk_ids,
                    "first_mention_chunk_id": e.first_mention_chunk_id,
                    "document_id": e.document_id,
                    "section_path": e.section_path,
                    "extraction_method": e.extraction_method,
                }
                for e in batch
            ]

            # Batch insert using UNWIND
            result = self.manager.execute(
                """
                UNWIND $entities as entity
                MERGE (e:Entity {id: entity.id})
                SET e = entity
                RETURN COUNT(e) as count
                """,
                {"entities": entities_data},
            )

            # Safely extract count from result (Bug #1 fix)
            batch_count = (
                result[0].get("count", len(batch)) if result and len(result) > 0 else len(batch)
            )
            total_added += batch_count

            logger.debug(f"Batch {i//batch_size + 1}: Added {batch_count} entities")

        # Bug #2 fix: Use math.ceil for correct batch count
        import math

        actual_batches = math.ceil(len(entities) / batch_size)
        logger.info(f"Added {total_added} entities to Neo4j (in {actual_batches} batches)")

    def add_relationships(self, relationships: List[Relationship]) -> None:
        """
        Add relationships to Neo4j as edges using batch operations.

        Processes relationships in batches of 500 for optimal performance.
        Groups by relationship type for efficient Cypher generation.
        """
        batch_size = 500
        total_added = 0

        # Group relationships by type for batch processing
        from collections import defaultdict

        rels_by_type = defaultdict(list)
        for rel in relationships:
            rels_by_type[rel.type.value].append(rel)

        # Process each type separately
        for rel_type, rels_list in rels_by_type.items():
            rel_type_upper = rel_type.upper().replace("-", "_")

            for i in range(0, len(rels_list), batch_size):
                batch = rels_list[i : i + batch_size]

                # Convert batch to list of dicts with ID remapping
                rels_data = [
                    {
                        "id": r.id,
                        "source_id": self._id_aliases.get(r.source_entity_id, r.source_entity_id),
                        "target_id": self._id_aliases.get(r.target_entity_id, r.target_entity_id),
                        "confidence": r.confidence,
                        "source_chunk_id": r.source_chunk_id,
                        "evidence_text": r.evidence_text,
                        "extraction_method": r.extraction_method,
                        "type_value": r.type.value,
                    }
                    for r in batch
                ]

                # Bug #4 fix: Detect missing entities and log warnings
                result = self.manager.execute(
                    f"""
                    UNWIND $rels as rel
                    OPTIONAL MATCH (source:Entity {{id: rel.source_id}})
                    OPTIONAL MATCH (target:Entity {{id: rel.target_id}})
                    WITH rel, source, target,
                         CASE WHEN source IS NULL THEN rel.source_id ELSE NULL END as missing_source,
                         CASE WHEN target IS NULL THEN rel.target_id ELSE NULL END as missing_target
                    WHERE source IS NOT NULL AND target IS NOT NULL
                    MERGE (source)-[r:{rel_type_upper}]->(target)
                    SET r.id = rel.id,
                        r.type = rel.type_value,
                        r.confidence = rel.confidence,
                        r.source_chunk_id = rel.source_chunk_id,
                        r.evidence_text = rel.evidence_text,
                        r.extraction_method = rel.extraction_method
                    RETURN COUNT(r) as count,
                           COLLECT(DISTINCT missing_source) + COLLECT(DISTINCT missing_target) as missing_entities
                    """,
                    {"rels": rels_data},
                )

                # Warn about missing entities
                if result and len(result) > 0:
                    missing = [e for e in result[0].get("missing_entities", []) if e is not None]
                    if missing:
                        logger.warning(
                            f"Skipped {len(missing)} relationships due to missing entities: {missing[:5]}"
                            + (f" (and {len(missing) - 5} more)" if len(missing) > 5 else "")
                        )

                # Safely extract count from result (Bug #1 fix)
                batch_count = (
                    result[0].get("count", len(batch)) if result and len(result) > 0 else len(batch)
                )
                total_added += batch_count

                logger.debug(
                    f"Batch {i//batch_size + 1} ({rel_type}): Added {batch_count} relationships"
                )

        # Bug #3 fix: Use math.ceil for correct batch count
        import math

        total_batches = sum(math.ceil(len(rels) / batch_size) for rels in rels_by_type.values())
        logger.info(f"Added {total_added} relationships to Neo4j (in {total_batches} batches)")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID from Neo4j."""
        result = self.manager.execute("MATCH (e:Entity {id: $id}) RETURN e", {"id": entity_id})

        if not result:
            return None

        return self._node_to_entity(result[0]["e"])

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity."""
        result = self.manager.execute(
            """
            MATCH (e:Entity {id: $id})-[r]-(other)
            RETURN r, startNode(r) as source, endNode(r) as target
            """,
            {"id": entity_id},
        )

        relationships = []
        for record in result:
            rel = self._edge_to_relationship(
                record["r"],
                record["source"]["id"],
                record["target"]["id"],
            )
            relationships.append(rel)

        return relationships

    def export_to_knowledge_graph(self) -> KnowledgeGraph:
        """Export all data from Neo4j to KnowledgeGraph object."""
        # Get all entities
        entity_result = self.manager.execute("MATCH (e:Entity) RETURN e")
        entities = [self._node_to_entity(record["e"]) for record in entity_result]

        # Get all relationships
        rel_result = self.manager.execute(
            """
            MATCH (source:Entity)-[r]->(target:Entity)
            RETURN r, source.id as source_id, target.id as target_id
            """
        )
        relationships = [
            self._edge_to_relationship(record["r"], record["source_id"], record["target_id"])
            for record in rel_result
        ]

        kg = KnowledgeGraph(entities=entities, relationships=relationships)
        kg.compute_stats()

        logger.info(
            f"Exported {len(entities)} entities and {len(relationships)} relationships from Neo4j"
        )

        return kg

    def save(self, path: str) -> None:
        """Export graph to JSON file."""
        kg = self.export_to_knowledge_graph()

        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        kg.save_json(path)
        logger.info(f"Exported Neo4j graph to {path}")

    def close(self) -> None:
        """Close Neo4j connection via manager."""
        if hasattr(self, "manager") and self.manager:
            self.manager.close()

    def _node_to_entity(self, node) -> Entity:
        """Convert Neo4j node to Entity object."""
        # Deserialize metadata from JSON string if present
        metadata_json = node.get("metadata", "{}")
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                f"Failed to deserialize metadata for entity {node.get('id')}: {metadata_json}"
            )
            metadata = {}

        return Entity(
            id=node["id"],
            type=EntityType(node["type"]),
            value=node["value"],
            normalized_value=node["normalized_value"],
            confidence=node["confidence"],
            source_chunk_ids=node.get("source_chunk_ids", []),
            first_mention_chunk_id=node.get("first_mention_chunk_id"),
            document_id=node.get("document_id"),
            section_path=node.get("section_path"),
            metadata=metadata,
            extraction_method=node.get("extraction_method", "unknown"),
        )

    def _edge_to_relationship(self, edge, source_id: str, target_id: str) -> Relationship:
        """Convert Neo4j edge to Relationship object."""
        return Relationship(
            id=edge["id"],
            type=RelationshipType(edge["type"]),
            source_entity_id=source_id,
            target_entity_id=target_id,
            confidence=edge["confidence"],
            source_chunk_id=edge.get("source_chunk_id", ""),
            evidence_text=edge.get("evidence_text", ""),
            extraction_method=edge.get("extraction_method", "unknown"),
        )


class NetworkXGraphBuilder(GraphBuilder):
    """
    NetworkX-based graph builder for lightweight operations.

    Uses NetworkX DiGraph for in-memory graph operations and algorithms.
    """

    def __init__(self, config: GraphStorageConfig):
        """Initialize NetworkX graph builder."""
        self.config = config

        try:
            import networkx as nx

            self.nx = nx
            self.graph = nx.DiGraph()
        except ImportError:
            raise ImportError("networkx package not installed. Install with: pip install networkx")

        logger.info("Initialized NetworkXGraphBuilder")

    def add_entities(self, entities: List[Entity]) -> None:
        """Add entities as nodes."""
        for entity in entities:
            self.graph.add_node(
                entity.id,
                entity_obj=entity,
                type=entity.type.value,
                value=entity.value,
                normalized_value=entity.normalized_value,
            )

        logger.info(f"Added {len(entities)} entities to NetworkX graph")

    def add_relationships(self, relationships: List[Relationship]) -> None:
        """Add relationships as edges."""
        for rel in relationships:
            self.graph.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                relationship_obj=rel,
                type=rel.type.value,
                confidence=rel.confidence,
            )

        logger.info(f"Added {len(relationships)} relationships to NetworkX graph")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        if entity_id in self.graph.nodes:
            return self.graph.nodes[entity_id].get("entity_obj")
        return None

    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity."""
        relationships = []

        # Outgoing edges
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            rel = data.get("relationship_obj")
            if rel:
                relationships.append(rel)

        # Incoming edges
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            rel = data.get("relationship_obj")
            if rel:
                relationships.append(rel)

        return relationships

    def export_to_knowledge_graph(self) -> KnowledgeGraph:
        """Export to KnowledgeGraph object."""
        entities = [
            data["entity_obj"] for _, data in self.graph.nodes(data=True) if "entity_obj" in data
        ]

        relationships = []
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relationship_obj")
            if rel:
                relationships.append(rel)

        kg = KnowledgeGraph(entities=entities, relationships=relationships)
        kg.compute_stats()

        return kg

    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        kg = self.export_to_knowledge_graph()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        kg.save_json(path)

        logger.info(f"Saved NetworkX graph to {path}")

    def close(self) -> None:
        """Cleanup (no-op)."""
        pass


def create_graph_builder(config: GraphStorageConfig) -> GraphBuilder:
    """
    Factory function to create appropriate graph builder.

    Args:
        config: Graph storage configuration

    Returns:
        GraphBuilder instance
    """
    from .config import GraphBackend

    if config.backend == GraphBackend.NEO4J:
        return Neo4jGraphBuilder(config)
    elif config.backend == GraphBackend.SIMPLE:
        return SimpleGraphBuilder(config)
    elif config.backend == GraphBackend.NETWORKX:
        return NetworkXGraphBuilder(config)
    else:
        raise ValueError(f"Unsupported graph backend: {config.backend}")
