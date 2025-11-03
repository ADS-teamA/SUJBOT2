"""
Neo4j incremental entity deduplication.

Provides incremental deduplication during entity insertion into Neo4j.
Uses APOC procedures when available, falls back to pure Cypher.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .config import EntityDeduplicationConfig
    from .models import Entity
    from .neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class Neo4jDeduplicator:
    """
    Incremental Neo4j deduplication with APOC/Cypher fallback.

    Features:
    - APOC optimization when available
    - Pure Cypher fallback (compatible with all Neo4j versions)
    - Property merging (confidence MAX, chunks UNION)
    - Document provenance tracking
    - Uniqueness constraints

    Performance:
    - APOC: ~10-20ms per 1000 entities
    - Pure Cypher: ~20-50ms per 1000 entities
    - Constraint enforcement: ~5ms overhead

    Example:
        >>> manager = Neo4jManager(neo4j_config)
        >>> config = EntityDeduplicationConfig()
        >>> dedup = Neo4jDeduplicator(manager, config)
        >>>
        >>> # Setup constraints once
        >>> dedup.create_uniqueness_constraints()
        >>>
        >>> # Add entities with dedup
        >>> stats = dedup.add_entities_with_dedup(entities)
        >>> print(f"Added {stats['entities_added']}, merged {stats['entities_merged']}")
    """

    def __init__(
        self,
        manager: "Neo4jManager",
        config: "EntityDeduplicationConfig",
    ):
        """
        Initialize Neo4j deduplicator.

        Args:
            manager: Neo4jManager for database operations
            config: EntityDeduplicationConfig
        """
        self.manager = manager
        self.config = config
        self.apoc_available = self._check_apoc() if config.apoc_enabled else False

        logger.info(
            f"Neo4jDeduplicator initialized "
            f"(apoc_available={self.apoc_available}, "
            f"create_constraints={config.create_uniqueness_constraints})"
        )

    def create_uniqueness_constraints(self) -> None:
        """
        Create Neo4j uniqueness constraints for deduplication.

        Constraints created:
        1. Composite unique on (type, normalized_value)
        2. Unique on Entity.id (already exists)

        Should be called ONCE at pipeline initialization.
        """
        if not self.config.create_uniqueness_constraints:
            logger.info("Uniqueness constraint creation disabled")
            return

        try:
            # Try composite constraint (Neo4j 4.2+)
            self.manager.execute(
                """
                CREATE CONSTRAINT entity_type_normalized_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE (e.type, e.normalized_value) IS UNIQUE
                """
            )
            logger.info("Created composite uniqueness constraint on (type, normalized_value)")

        except Exception as e:
            logger.warning(f"Could not create composite constraint (may require Neo4j 4.2+): {e}")

            # Fallback: Create composite index
            try:
                self.manager.execute(
                    """
                    CREATE INDEX entity_type_normalized_idx IF NOT EXISTS
                    FOR (e:Entity) ON (e.type, e.normalized_value)
                    """
                )
                logger.info("Created composite index as constraint fallback")
            except Exception as e2:
                logger.warning(f"Could not create composite index: {e2}")

    def add_entities_with_dedup(
        self,
        entities: List["Entity"],
        batch_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Add entities to Neo4j with incremental deduplication.

        Main entry point for pipeline integration.

        Args:
            entities: List of Entity objects from entity extractor
            batch_size: Batch size for Neo4j operations

        Returns:
            Statistics dict with:
            - entities_added: Number of new entities created
            - entities_merged: Number of duplicates merged
            - entities_failed: Number of failed operations
            - merge_details: List of merge operations
            - id_aliases: Dict mapping duplicate entity IDs to canonical IDs

        Example:
            >>> entities = entity_extractor.extract_from_chunks(chunks)
            >>> stats = dedup.add_entities_with_dedup(entities)
        """
        stats: Dict[str, Any] = {
            "entities_added": 0,
            "entities_merged": 0,
            "entities_failed": 0,
            "entities_unknown": 0,
            "merge_details": [],
            "id_aliases": {},  # Maps duplicate_id → canonical_id
        }

        # Build entity lookup by (type, normalized_value) for ID mapping
        entity_lookup: Dict[tuple, str] = {}  # (type, normalized_value) → entity.id
        for entity in entities:
            key = (entity.type.value, entity.normalized_value)
            if key not in entity_lookup:
                entity_lookup[key] = entity.id  # First occurrence is canonical
            else:
                # This is a duplicate - map its ID to the canonical ID
                stats["id_aliases"][entity.id] = entity_lookup[key]

        # Process in batches
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]

            try:
                batch_stats = self._process_batch_with_dedup(batch)

                # Accumulate stats
                stats["entities_added"] += batch_stats.get("entities_added", 0)
                stats["entities_merged"] += batch_stats.get("entities_merged", 0)
                stats["entities_failed"] += batch_stats.get("entities_failed", 0)
                stats["entities_unknown"] += batch_stats.get("entities_unknown", 0)
                stats["merge_details"].extend(batch_stats.get("merge_details", []))

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                stats["entities_failed"] += len(batch)

        logger.info(
            f"Deduplication complete: "
            f"added={stats['entities_added']}, "
            f"merged={stats['entities_merged']}, "
            f"failed={stats['entities_failed']}, "
            f"unknown={stats['entities_unknown']}"
        )

        return stats

    def _process_batch_with_dedup(self, batch: List["Entity"]) -> Dict[str, Any]:
        """
        Process single batch with deduplication.

        Try APOC first, fallback to pure Cypher if APOC unavailable.

        Args:
            batch: List of entities to process

        Returns:
            Batch statistics
        """
        try:
            if self.apoc_available:
                return self._apoc_batch_merge(batch)
            else:
                return self._cypher_batch_merge(batch)

        except Exception as e:
            # If APOC fails, try Cypher fallback
            if self.apoc_available and "apoc" in str(e).lower():
                logger.warning(f"APOC merge failed: {e}, falling back to Cypher")
                self.apoc_available = False  # Disable APOC for future batches
                return self._cypher_batch_merge(batch)
            else:
                raise

    def _apoc_batch_merge(self, batch: List["Entity"]) -> Dict[str, Any]:
        """
        Use APOC apoc.coll.union for efficient batch merging.

        APOC Query Strategy:
        - UNWIND batch
        - MERGE on (type, normalized_value)
        - ON CREATE: Set all properties
        - ON MATCH: Merge arrays using apoc.coll.union, MAX confidence

        Args:
            batch: Entities to merge

        Returns:
            Statistics dict
        """
        entities_data = [self._entity_to_dict(e) for e in batch]

        result = self.manager.execute(
            """
            UNWIND $entities as entity
            MERGE (e:Entity {type: entity.type, normalized_value: entity.normalized_value})
            ON CREATE SET
              e.id = entity.id,
              e.value = entity.value,
              e.confidence = entity.confidence,
              e.source_chunk_ids = entity.source_chunk_ids,
              e.first_mention_chunk_id = entity.first_mention_chunk_id,
              e.document_id = entity.document_id,
              e.section_path = entity.section_path,
              e.extraction_method = entity.extraction_method,
              e.metadata = entity.metadata,
              e.merged_from = [],
              e.created_at = datetime(),
              e._is_new = true
            ON MATCH SET
              e.source_chunk_ids = apoc.coll.union(e.source_chunk_ids, entity.source_chunk_ids),
              e.confidence = CASE WHEN entity.confidence > e.confidence THEN entity.confidence ELSE e.confidence END,
              e.merged_from = apoc.coll.union(e.merged_from, [entity.id]),
              e.metadata = CASE
                WHEN entity.metadata IS NOT NULL
                THEN apoc.map.merge(coalesce(e.metadata, {}), entity.metadata)
                ELSE e.metadata
              END,
              e.updated_at = datetime(),
              e._is_new = false
            RETURN
              SUM(CASE WHEN e._is_new THEN 1 ELSE 0 END) as created_count,
              SUM(CASE WHEN NOT e._is_new THEN 1 ELSE 0 END) as merged_count,
              COUNT(*) as total_count
            """,
            {"entities": entities_data},
        )

        if not result or len(result) == 0:
            logger.error(
                f"Neo4j APOC merge returned empty result for batch of {len(batch)} entities. "
                f"Database state unknown - this may indicate constraint violation, "
                f"transaction rollback, or query timeout. Manual verification recommended."
            )
            return {
                "entities_added": 0,
                "entities_merged": 0,
                "entities_failed": 0,
                "entities_unknown": len(batch),
                "merge_details": [],
            }

        record = result[0]
        created = record.get("created_count", 0)
        merged = record.get("merged_count", 0)

        return {
            "entities_added": created,
            "entities_merged": merged,
            "entities_failed": 0,
            "entities_unknown": 0,
            "merge_details": [],
        }

    def _cypher_batch_merge(self, batch: List["Entity"]) -> Dict[str, Any]:
        """
        Pure Cypher fallback without APOC.

        Uses manual array manipulation instead of apoc.coll.union.

        Args:
            batch: Entities to merge

        Returns:
            Statistics dict
        """
        stats: Dict[str, Any] = {
            "entities_added": 0,
            "entities_merged": 0,
            "entities_failed": 0,
            "entities_unknown": 0,
            "merge_details": [],
        }

        entities_data = [self._entity_to_dict(e) for e in batch]

        # Process batch with pure Cypher
        result = self.manager.execute(
            """
            UNWIND $entities as entity
            MERGE (e:Entity {type: entity.type, normalized_value: entity.normalized_value})
            ON CREATE SET
              e.id = entity.id,
              e.value = entity.value,
              e.confidence = entity.confidence,
              e.source_chunk_ids = entity.source_chunk_ids,
              e.first_mention_chunk_id = entity.first_mention_chunk_id,
              e.document_id = entity.document_id,
              e.section_path = entity.section_path,
              e.extraction_method = entity.extraction_method,
              e.metadata = entity.metadata,
              e.merged_from = [],
              e.created_at = datetime(),
              e._is_new = true
            ON MATCH SET
              e.source_chunk_ids = reduce(acc = [], chunk IN e.source_chunk_ids + entity.source_chunk_ids |
                CASE WHEN chunk IN acc THEN acc ELSE acc + [chunk] END),
              e.confidence = CASE WHEN entity.confidence > e.confidence THEN entity.confidence ELSE e.confidence END,
              e.merged_from = CASE WHEN NOT entity.id IN e.merged_from THEN e.merged_from + [entity.id] ELSE e.merged_from END,
              e.metadata = CASE
                WHEN entity.metadata IS NOT NULL AND e.metadata IS NOT NULL
                THEN entity.metadata
                WHEN entity.metadata IS NOT NULL
                THEN entity.metadata
                ELSE e.metadata
              END,
              e.updated_at = datetime(),
              e._is_new = false,
              e._metadata_merge_warning = CASE
                WHEN entity.metadata IS NOT NULL AND e.metadata IS NOT NULL
                THEN "Pure Cypher mode: metadata replaced, not merged. Use APOC for full metadata merging."
                ELSE null
              END
            RETURN
              SUM(CASE WHEN e._is_new THEN 1 ELSE 0 END) as created_count,
              SUM(CASE WHEN NOT e._is_new THEN 1 ELSE 0 END) as merged_count
            """,
            {"entities": entities_data},
        )

        if result and len(result) > 0:
            record = result[0]
            stats["entities_added"] = record.get("created_count", 0)
            stats["entities_merged"] = record.get("merged_count", 0)

        return stats

    def _entity_to_dict(self, entity: "Entity") -> Dict[str, Any]:
        """
        Convert Entity to dict for Neo4j.

        Args:
            entity: Entity object

        Returns:
            Dict with all entity properties
        """
        return {
            "id": entity.id,
            "type": entity.type.value,
            "value": entity.value,
            "normalized_value": entity.normalized_value,
            "confidence": entity.confidence,
            "source_chunk_ids": entity.source_chunk_ids or [],
            "first_mention_chunk_id": entity.first_mention_chunk_id,
            "document_id": entity.document_id,
            "section_path": entity.section_path,
            "extraction_method": entity.extraction_method,
            "metadata": entity.metadata or {},
        }

    def _check_apoc(self) -> bool:
        """
        Check if APOC procedures are available.

        Returns:
            True if APOC is available, False otherwise

        Raises:
            Exception: Re-raises authentication and connection errors
        """
        try:
            result = self.manager.execute("RETURN apoc.version() as version")
            if result and len(result) > 0:
                version = result[0].get("version")
                logger.info(f"APOC available: version {version}")
                return True
            return False

        except Exception as e:
            error_msg = str(e).lower()

            # Categorize errors properly to avoid masking real problems
            if "apoc" in error_msg or "procedure" in error_msg or "function" in error_msg:
                # APOC not installed or not enabled
                logger.info("APOC not available, will use pure Cypher fallback")
                return False
            elif "auth" in error_msg or "credential" in error_msg or "password" in error_msg:
                # Authentication failure - don't mask this!
                logger.error(f"Neo4j authentication failed during APOC check: {e}")
                raise
            elif "timeout" in error_msg or "connection" in error_msg or "refused" in error_msg:
                # Connection failure - don't mask this!
                logger.error(f"Neo4j connection failed during APOC check: {e}")
                raise
            else:
                # Unknown error - log warning but assume APOC unavailable
                logger.warning(
                    f"APOC check failed with unexpected error (assuming unavailable): {e}"
                )
                return False
