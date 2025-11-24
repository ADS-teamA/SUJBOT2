"""
Knowledge Graph Pipeline orchestrator.

Main entry point for building knowledge graphs from legal document chunks.
Coordinates entity extraction, relationship extraction, and graph building.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import Entity, Relationship, KnowledgeGraph
from .config import KnowledgeGraphConfig
from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .graph_builder import create_graph_builder, GraphBuilder


logger = logging.getLogger(__name__)


class KnowledgeGraphPipeline:
    """
    Complete Knowledge Graph construction pipeline.

    Orchestrates the full workflow:
    1. Entity extraction from chunks
    2. Relationship extraction between entities
    3. Graph construction and storage
    4. Export to various formats

    Usage:
        config = KnowledgeGraphConfig.from_env()
        pipeline = KnowledgeGraphPipeline(config)

        # Build from chunks
        kg = pipeline.build_from_chunks(chunks)

        # Or build from phase3 output file
        kg = pipeline.build_from_phase3_file("data/phase3_chunks.json")
    """

    def __init__(self, config: KnowledgeGraphConfig):
        """
        Initialize Knowledge Graph pipeline.

        Args:
            config: Knowledge Graph configuration
        """
        self.config = config

        # Validate configuration
        self.config.validate()

        # Propagate regex prefill toggle to entity extraction config
        self.config.entity_extraction.enable_regex_prefill = self.config.enable_regex_prefill
        self.config.entity_extraction.enable_semantic_dedup = (
            self.config.enable_semantic_entity_dedup
        )

        # Initialize components
        self.entity_extractor: Optional[EntityExtractor] = None
        self.relationship_extractor: Optional[RelationshipExtractor] = None
        self.graph_builder: Optional[GraphBuilder] = None

        # Initialize extractors
        if self.config.enable_entity_extraction:
            self._initialize_entity_extractor()

        if self.config.enable_relationship_extraction:
            self._initialize_relationship_extractor()

        # Initialize graph builder
        self.graph_builder = create_graph_builder(self.config.graph_storage)

        logger.info("Initialized KnowledgeGraphPipeline")

    def _initialize_entity_extractor(self):
        """Initialize entity extractor with API keys."""
        api_key = None
        if self.config.entity_extraction.llm_provider == "openai":
            api_key = self.config.openai_api_key
        elif self.config.entity_extraction.llm_provider == "anthropic":
            api_key = self.config.anthropic_api_key

        self.entity_extractor = EntityExtractor(
            config=self.config.entity_extraction,
            api_key=api_key,
        )

    def _initialize_relationship_extractor(self):
        """Initialize relationship extractor with API keys."""
        api_key = None
        if self.config.relationship_extraction.llm_provider == "openai":
            api_key = self.config.openai_api_key
        elif self.config.relationship_extraction.llm_provider == "anthropic":
            api_key = self.config.anthropic_api_key

        self.relationship_extractor = RelationshipExtractor(
            config=self.config.relationship_extraction,
            api_key=api_key,
        )

    def build_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: Optional[str] = None,
    ) -> KnowledgeGraph:
        """
        Build knowledge graph from chunks.

        Args:
            chunks: List of chunk dictionaries with 'id', 'content', 'metadata'
            document_id: Optional document ID for metadata

        Returns:
            KnowledgeGraph object
        """
        logger.info(f"Building knowledge graph from {len(chunks)} chunks...")

        start_time = datetime.now()

        # Step 1: Extract entities
        entities = []
        if self.config.enable_entity_extraction and self.entity_extractor:
            logger.info("=" * 60)
            logger.info("STEP 1: Entity Extraction")
            logger.info("=" * 60)

            entities = self.entity_extractor.extract_from_chunks(chunks)
            logger.info(f"Extracted {len(entities)} entities")

            # Log entity type distribution
            entity_type_counts = {}
            for entity in entities:
                entity_type = entity.type.value
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

            logger.info(f"Entity type distribution: {entity_type_counts}")
        else:
            logger.warning("Entity extraction disabled")

        # Step 2: Extract relationships
        relationships = []
        if self.config.enable_relationship_extraction and self.relationship_extractor:
            logger.info("=" * 60)
            logger.info("STEP 2: Relationship Extraction")
            logger.info("=" * 60)

            if not entities:
                logger.warning("No entities found, skipping relationship extraction")
            else:
                relationships = self.relationship_extractor.extract_relationships(
                    entities=entities,
                    chunks=chunks,
                )
                logger.info(f"Extracted {len(relationships)} relationships")

                # Log relationship type distribution
                rel_type_counts = {}
                for rel in relationships:
                    rel_type = rel.type.value
                    rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1

                logger.info(f"Relationship type distribution: {rel_type_counts}")
        else:
            logger.warning("Relationship extraction disabled")

        # Step 3: Build graph
        logger.info("=" * 60)
        logger.info("STEP 3: Graph Construction")
        logger.info("=" * 60)

        self.graph_builder.add_entities(entities)
        self.graph_builder.add_relationships(relationships)

        # Step 4: Export to KnowledgeGraph
        logger.info("=" * 60)
        logger.info("STEP 4: Export")
        logger.info("=" * 60)

        kg = self.graph_builder.export_to_knowledge_graph()
        kg.source_document_id = document_id
        kg.created_at = datetime.now()
        kg.extraction_config = {
            "entity_extraction": {
                "llm_model": self.config.entity_extraction.llm_model,
                "enabled_types": [
                    et.value for et in self.config.entity_extraction.enabled_entity_types
                ],
            },
            "relationship_extraction": {
                "llm_model": self.config.relationship_extraction.llm_model,
                "enabled_types": [
                    rt.value
                    for rt in self.config.relationship_extraction.enabled_relationship_types
                ],
            },
            "graph_backend": self.config.graph_storage.backend.value,
        }

        # Compute statistics
        kg.compute_stats()

        # Save to JSON if configured
        if self.config.graph_storage.export_json:
            export_path = self.config.graph_storage.export_path
            self.graph_builder.save(export_path)

        # Log summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total entities: {len(kg.entities)}")
        logger.info(f"Total relationships: {len(kg.relationships)}")
        logger.info(f"Time elapsed: {elapsed:.2f}s")
        logger.info(f"Statistics: {kg.stats}")

        return kg

    def build_from_phase3_file(
        self, phase3_path: str, document_id: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Build knowledge graph from phase3 chunks JSON file.

        Args:
            phase3_path: Path to phase3_chunks.json file
            document_id: Optional document ID (auto-detected from file if not provided)

        Returns:
            KnowledgeGraph object
        """
        logger.info(f"Loading chunks from {phase3_path}...")

        # Load phase3 chunks
        with open(phase3_path, "r", encoding="utf-8") as f:
            phase3_data = json.load(f)

        # Extract chunks
        chunks = phase3_data.get("chunks", [])

        # Auto-detect document ID if not provided
        if not document_id and chunks:
            metadata = chunks[0].get("metadata", {})
            document_id = metadata.get("document_id", Path(phase3_path).stem)

        logger.info(f"Loaded {len(chunks)} chunks from {phase3_path}")

        # Build knowledge graph
        kg = self.build_from_chunks(chunks, document_id=document_id)
        kg.source_chunks_file = phase3_path

        return kg

    def build_from_multiple_documents(
        self,
        phase3_files: List[str],
        enable_cross_document_relationships: bool = False,
    ) -> KnowledgeGraph:
        """
        Build knowledge graph from multiple phase3 files.

        Args:
            phase3_files: List of paths to phase3_chunks.json files
            enable_cross_document_relationships: Extract relationships across documents

        Returns:
            Combined KnowledgeGraph object
        """
        logger.info(f"Building multi-document knowledge graph from {len(phase3_files)} files...")

        all_chunks = []
        all_document_ids = []

        # Load all chunks
        for phase3_path in phase3_files:
            with open(phase3_path, "r", encoding="utf-8") as f:
                phase3_data = json.load(f)

            chunks = phase3_data.get("chunks", [])
            all_chunks.extend(chunks)

            # Track document IDs
            if chunks:
                doc_id = chunks[0].get("metadata", {}).get("document_id", Path(phase3_path).stem)
                all_document_ids.append(doc_id)

        logger.info(f"Loaded {len(all_chunks)} total chunks from {len(phase3_files)} documents")

        # Build combined knowledge graph
        kg = self.build_from_chunks(
            all_chunks,
            document_id=f"multi_doc_{len(phase3_files)}",
        )

        kg.source_chunks_file = f"{len(phase3_files)} documents"
        kg.metadata = {
            "document_ids": all_document_ids,
            "cross_document_relationships": enable_cross_document_relationships,
        }

        return kg

    def merge_graphs(self, other_pipeline: "KnowledgeGraphPipeline") -> Dict[str, Any]:
        """
        Merge another knowledge graph pipeline's graph into this one.

        Entities are deduplicated using (type, normalized_value) matching.
        Relationships are updated to reference merged entity IDs.

        Args:
            other_pipeline: KnowledgeGraphPipeline to merge from

        Returns:
            Merge statistics with entity/relationship counts

        Example:
            >>> target_pipeline = KnowledgeGraphPipeline(config)
            >>> target_pipeline.build_from_phase3_file("doc1.json")
            >>>
            >>> source_pipeline = KnowledgeGraphPipeline(config)
            >>> source_pipeline.build_from_phase3_file("doc2.json")
            >>>
            >>> stats = target_pipeline.merge_graphs(source_pipeline)
            >>> print(f"Merged: +{stats['entities_added']} entities")
        """
        if not hasattr(self.graph_builder, "merge"):
            raise NotImplementedError(
                f"Graph builder {type(self.graph_builder).__name__} does not support merge"
            )

        logger.info("Merging knowledge graphs...")
        stats = self.graph_builder.merge(other_pipeline.graph_builder)

        logger.info(
            f"Merge complete: "
            f"+{stats['entities_added']} entities, "
            f"~{stats['entities_deduplicated']} deduplicated, "
            f"+{stats['relationships_added']} relationships"
        )

        return stats

    def query_graph(self, entity_id: str) -> Dict[str, Any]:
        """
        Query graph for entity and its relationships.

        Args:
            entity_id: Entity ID to query

        Returns:
            Dictionary with entity, relationships, and neighbors
        """
        entity = self.graph_builder.get_entity(entity_id)
        if not entity:
            return {"error": f"Entity {entity_id} not found"}

        relationships = self.graph_builder.get_relationships_for_entity(entity_id)

        # Get neighbor entities
        neighbors = []
        for rel in relationships:
            if rel.source_entity_id == entity_id:
                neighbor = self.graph_builder.get_entity(rel.target_entity_id)
                if neighbor:
                    neighbors.append(
                        {
                            "entity": neighbor,
                            "relationship": rel,
                            "direction": "outgoing",
                        }
                    )
            elif rel.target_entity_id == entity_id:
                neighbor = self.graph_builder.get_entity(rel.source_entity_id)
                if neighbor:
                    neighbors.append(
                        {
                            "entity": neighbor,
                            "relationship": rel,
                            "direction": "incoming",
                        }
                    )

        return {
            "entity": entity,
            "relationships": relationships,
            "neighbors": neighbors,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if hasattr(self.graph_builder, "get_statistics"):
            return self.graph_builder.get_statistics()
        else:
            kg = self.graph_builder.export_to_knowledge_graph()
            return kg.compute_stats()

    def close(self):
        """Close connections and cleanup."""
        if self.graph_builder:
            self.graph_builder.close()

        logger.info("Pipeline closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def build_unified_kg(
        self,
        storage_dir: str = "vector_db",
        enable_cross_document_relationships: bool = True,
    ) -> KnowledgeGraph:
        """
        Build unified knowledge graph from all existing per-document KGs.

        Features:
        - Loads all *_kg.json files from storage_dir
        - Deduplicates entities across documents
        - Detects cross-document relationships
        - Saves to unified_kg.json + per-document backups

        Args:
            storage_dir: Directory with existing KG files (default: vector_db)
            enable_cross_document_relationships: Enable cross-document relationship detection

        Returns:
            Unified KnowledgeGraph

        Example:
            >>> pipeline = KnowledgeGraphPipeline(config)
            >>> unified_kg = pipeline.build_unified_kg("vector_db")
            >>> print(f"Unified KG: {len(unified_kg.entities)} entities")
        """
        from .unified_kg_manager import UnifiedKnowledgeGraphManager
        from .cross_doc_detector import CrossDocumentRelationshipDetector

        logger.info("=" * 60)
        logger.info("BUILDING UNIFIED KNOWLEDGE GRAPH")
        logger.info("=" * 60)

        # Initialize manager and detector
        manager = UnifiedKnowledgeGraphManager(storage_dir=storage_dir)

        cross_doc_detector = None
        if enable_cross_document_relationships:
            cross_doc_detector = CrossDocumentRelationshipDetector(
                use_llm_validation=False,  # Pattern-based for speed
                confidence_threshold=0.7,
            )

        # Load existing unified KG or create new
        unified_kg = manager.load_or_create()

        # Find all per-document KG files
        storage_path = Path(storage_dir)
        kg_files = list(storage_path.glob("*_kg.json"))

        # Exclude unified_kg.json itself
        kg_files = [f for f in kg_files if f.name != "unified_kg.json"]

        logger.info(f"Found {len(kg_files)} document KGs to merge")

        # Merge each document KG
        for kg_file in kg_files:
            document_id = kg_file.stem.replace("_kg", "")
            logger.info(f"Merging document: {document_id}")

            # Load document KG
            doc_kg = KnowledgeGraph.load_json(str(kg_file))

            # Merge into unified KG
            unified_kg = manager.merge_document_graph(
                unified_kg=unified_kg,
                document_kg=doc_kg,
                document_id=document_id,
                cross_doc_detector=cross_doc_detector,
            )

        # Save unified KG + per-document backups
        logger.info("Saving unified KG...")
        for kg_file in kg_files:
            document_id = kg_file.stem.replace("_kg", "")
            manager.save(unified_kg, document_id=document_id)

        # Print statistics
        doc_stats = manager.get_document_statistics(unified_kg)

        logger.info("=" * 60)
        logger.info("UNIFIED KG COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total entities: {len(unified_kg.entities)}")
        logger.info(f"Total relationships: {len(unified_kg.relationships)}")
        logger.info(f"Documents: {doc_stats['total_documents']}")
        logger.info(f"Cross-document entities: {doc_stats['cross_document_entities']}")
        logger.info(
            f"Cross-document entity %: {doc_stats['cross_document_entity_percentage']:.1f}%"
        )
        logger.info(f"Entities per document: {doc_stats['entities_per_document']}")

        return unified_kg


def build_knowledge_graph_from_file(
    phase3_path: str,
    config: Optional[KnowledgeGraphConfig] = None,
    output_path: Optional[str] = None,
) -> KnowledgeGraph:
    """
    Convenience function to build knowledge graph from phase3 file.

    Args:
        phase3_path: Path to phase3_chunks.json file
        config: Optional configuration (uses default if not provided)
        output_path: Optional output path for JSON export

    Returns:
        KnowledgeGraph object
    """
    if config is None:
        config = KnowledgeGraphConfig.from_env()

    # Override output path if provided
    if output_path:
        config.graph_storage.export_json = True
        config.graph_storage.export_path = output_path

    with KnowledgeGraphPipeline(config) as pipeline:
        kg = pipeline.build_from_phase3_file(phase3_path)

    return kg
