"""
Unified Knowledge Graph Manager.

Manages unified knowledge graph construction with:
- Cross-document entity deduplication
- Cross-document relationship detection
- Incremental merging of document graphs
- Document tracking per entity

Architecture: Pragmatic Balance (research-based, production-ready)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

from .models import Entity, Relationship, KnowledgeGraph, EntityType, RelationshipType
from .deduplicator import EntityDeduplicator

logger = logging.getLogger(__name__)


class UnifiedKnowledgeGraphManager:
    """
    Manages unified knowledge graph with cross-document entity deduplication.

    Features:
    - Incremental document merging (add new documents without rebuilding)
    - Entity deduplication with document tracking (metadata["document_ids"])
    - Cross-document relationship detection
    - Backward compatibility (saves per-document backups)

    Storage:
    - Primary: vector_db/unified_kg.json
    - Backup: vector_db/{document_id}_kg.json (per-document snapshots)

    Usage:
        manager = UnifiedKnowledgeGraphManager(storage_dir="vector_db")

        # Load existing unified graph (or create new)
        unified_kg = manager.load_or_create()

        # Merge new document graph
        new_kg = KnowledgeGraph.load_json("vector_db/BZ_VR1_kg.json")
        unified_kg = manager.merge_document_graph(unified_kg, new_kg, "BZ_VR1")

        # Save unified + per-document backup
        manager.save(unified_kg, document_id="BZ_VR1")
    """

    def __init__(self, storage_dir: str = "vector_db"):
        """
        Initialize unified KG manager.

        Args:
            storage_dir: Directory for KG storage (default: vector_db)
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.unified_kg_path = self.storage_dir / "unified_kg.json"
        self.deduplicator = EntityDeduplicator()

        logger.info(f"UnifiedKnowledgeGraphManager initialized: {self.storage_dir}")

    def load_or_create(self) -> KnowledgeGraph:
        """
        Load existing unified KG or create new empty graph.

        Returns:
            KnowledgeGraph: Unified graph (may be empty)
        """
        if self.unified_kg_path.exists():
            logger.info(f"Loading unified KG from {self.unified_kg_path}")
            kg = KnowledgeGraph.load_json(str(self.unified_kg_path))
            logger.info(
                f"Loaded unified KG: {len(kg.entities)} entities, "
                f"{len(kg.relationships)} relationships"
            )
            return kg
        else:
            logger.info("Creating new unified KG")
            return KnowledgeGraph(
                source_document_id="unified",
                created_at=datetime.now(),
            )

    def merge_document_graph(
        self,
        unified_kg: KnowledgeGraph,
        document_kg: KnowledgeGraph,
        document_id: str,
        cross_doc_detector: Optional["CrossDocumentRelationshipDetector"] = None,
    ) -> KnowledgeGraph:
        """
        Merge a document graph into unified graph with deduplication.

        Process:
        1. Deduplicate entities (exact match on type + normalized_value)
        2. Track document_id in metadata["document_ids"]
        3. Remap entity IDs in relationships
        4. Detect cross-document relationships (optional)
        5. Add new entities and relationships

        Args:
            unified_kg: Existing unified graph
            document_kg: New document graph to merge
            document_id: ID of source document
            cross_doc_detector: Optional detector for cross-document relationships

        Returns:
            KnowledgeGraph: Updated unified graph
        """
        logger.info(
            f"Merging document '{document_id}' into unified KG "
            f"(unified: {len(unified_kg.entities)} entities, "
            f"document: {len(document_kg.entities)} entities)"
        )

        # Build lookup index from unified graph: (type, normalized_value) -> entity
        entity_index: Dict[Tuple, Entity] = {}
        for entity in unified_kg.entities:
            key = (entity.type, entity.normalized_value)
            entity_index[key] = entity

        # Track entity ID remapping (old_id -> new_id)
        entity_id_remapping: Dict[str, str] = {}

        # Statistics
        stats = {
            "entities_added": 0,
            "entities_merged": 0,
            "relationships_added": 0,
            "cross_doc_relationships_added": 0,
        }

        # Phase 1: Deduplicate entities
        logger.info("Phase 1: Deduplicating entities...")

        for entity in document_kg.entities:
            # Find duplicate using exact match
            key = (entity.type, entity.normalized_value)
            duplicate = entity_index.get(key)

            if duplicate:
                # Entity already exists - merge metadata
                stats["entities_merged"] += 1

                # Track document ID
                if "document_ids" not in duplicate.metadata:
                    duplicate.metadata["document_ids"] = [duplicate.document_id]

                if document_id not in duplicate.metadata["document_ids"]:
                    duplicate.metadata["document_ids"].append(document_id)

                # Merge source chunks
                for chunk_id in entity.source_chunk_ids:
                    if chunk_id not in duplicate.source_chunk_ids:
                        duplicate.source_chunk_ids.append(chunk_id)

                # Merge confidence (take max)
                duplicate.confidence = max(duplicate.confidence, entity.confidence)

                # Track ID remapping
                entity_id_remapping[entity.id] = duplicate.id

            else:
                # New entity - add to unified graph
                stats["entities_added"] += 1

                # Initialize document tracking
                entity.metadata["document_ids"] = [document_id]
                entity.document_id = document_id  # Keep original for provenance

                # Add to unified graph
                unified_kg.entities.append(entity)

                # Update index for next iteration
                entity_index[key] = entity

                # No remapping needed (ID stays same)
                entity_id_remapping[entity.id] = entity.id

        logger.info(
            f"Phase 1 complete: {stats['entities_added']} added, "
            f"{stats['entities_merged']} merged"
        )

        # Phase 2: Remap relationships
        logger.info("Phase 2: Remapping relationships...")

        for rel in document_kg.relationships:
            # Remap source and target entity IDs
            new_source_id = entity_id_remapping.get(rel.source_entity_id)
            new_target_id = entity_id_remapping.get(rel.target_entity_id)

            if not new_source_id or not new_target_id:
                logger.warning(
                    f"Skipping relationship {rel.id}: missing entity mapping "
                    f"(source: {rel.source_entity_id} -> {new_source_id}, "
                    f"target: {rel.target_entity_id} -> {new_target_id})"
                )
                continue

            # Check if relationship already exists
            existing_rel = self._find_existing_relationship(
                unified_kg, new_source_id, new_target_id, rel.type
            )

            if existing_rel:
                # Relationship exists - update confidence
                existing_rel.confidence = max(existing_rel.confidence, rel.confidence)
            else:
                # New relationship - add with remapped IDs
                rel.source_entity_id = new_source_id
                rel.target_entity_id = new_target_id
                unified_kg.relationships.append(rel)
                stats["relationships_added"] += 1

        logger.info(f"Phase 2 complete: {stats['relationships_added']} relationships added")

        # Phase 3: Detect cross-document relationships (optional)
        if cross_doc_detector:
            logger.info("Phase 3: Detecting cross-document relationships...")

            cross_doc_rels = cross_doc_detector.detect_cross_document_relationships(
                unified_kg, document_id
            )

            for rel in cross_doc_rels:
                # Add cross-document relationship
                unified_kg.relationships.append(rel)
                stats["cross_doc_relationships_added"] += 1

            logger.info(
                f"Phase 3 complete: {stats['cross_doc_relationships_added']} "
                f"cross-document relationships added"
            )

        # Update metadata
        unified_kg.created_at = datetime.now()
        unified_kg.compute_stats()

        logger.info(
            f"Merge complete: unified KG now has {len(unified_kg.entities)} entities, "
            f"{len(unified_kg.relationships)} relationships"
        )
        logger.info(f"Merge stats: {stats}")

        return unified_kg

    def save(
        self,
        unified_kg: KnowledgeGraph,
        document_id: Optional[str] = None,
        save_per_document_backup: bool = True,
    ):
        """
        Save unified KG with optional per-document backup.

        Args:
            unified_kg: Unified knowledge graph to save
            document_id: Optional document ID for per-document backup
            save_per_document_backup: Whether to save per-document backup
        """
        # Save unified graph
        logger.info(f"Saving unified KG to {self.unified_kg_path}")
        unified_kg.save_json(str(self.unified_kg_path))

        # Save per-document backup (for audit trail)
        if save_per_document_backup and document_id:
            backup_path = self.storage_dir / f"{document_id}_kg.json"
            logger.info(f"Saving per-document backup to {backup_path}")

            # Extract entities and relationships for this document
            document_entities = [
                e
                for e in unified_kg.entities
                if document_id in e.metadata.get("document_ids", [])
            ]

            document_entity_ids = {e.id for e in document_entities}

            document_relationships = [
                r
                for r in unified_kg.relationships
                if r.source_entity_id in document_entity_ids
                or r.target_entity_id in document_entity_ids
            ]

            document_kg = KnowledgeGraph(
                entities=document_entities,
                relationships=document_relationships,
                source_document_id=document_id,
                created_at=datetime.now(),
            )
            document_kg.compute_stats()
            document_kg.save_json(str(backup_path))

            logger.info(
                f"Per-document backup saved: {len(document_entities)} entities, "
                f"{len(document_relationships)} relationships"
            )

    def get_document_statistics(self, unified_kg: KnowledgeGraph) -> Dict[str, Any]:
        """
        Compute statistics about document coverage in unified graph.

        Args:
            unified_kg: Unified knowledge graph

        Returns:
            Dict with statistics (total_documents, entities_per_document, etc.)
        """
        # Collect all document IDs
        document_ids: Set[str] = set()
        entities_per_document: Dict[str, int] = defaultdict(int)
        cross_document_entities = 0

        for entity in unified_kg.entities:
            doc_ids = entity.metadata.get("document_ids", [])
            document_ids.update(doc_ids)

            if len(doc_ids) > 1:
                cross_document_entities += 1

            for doc_id in doc_ids:
                entities_per_document[doc_id] += 1

        return {
            "total_documents": len(document_ids),
            "document_ids": sorted(list(document_ids)),
            "entities_per_document": dict(entities_per_document),
            "cross_document_entities": cross_document_entities,
            "cross_document_entity_percentage": (
                100 * cross_document_entities / len(unified_kg.entities)
                if unified_kg.entities
                else 0
            ),
        }

    def _find_existing_relationship(
        self,
        kg: KnowledgeGraph,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType,
    ) -> Optional[Relationship]:
        """
        Find existing relationship by source, target, and type.

        Args:
            kg: Knowledge graph to search
            source_id: Source entity ID
            target_id: Target entity ID
            rel_type: Relationship type

        Returns:
            Existing relationship or None
        """
        for rel in kg.relationships:
            if (
                rel.source_entity_id == source_id
                and rel.target_entity_id == target_id
                and rel.type == rel_type
            ):
                return rel
        return None
