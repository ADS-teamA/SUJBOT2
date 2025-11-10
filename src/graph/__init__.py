"""
Knowledge Graph module for legal document processing.

Implements entity extraction, relationship extraction, and graph construction
for creating knowledge graphs from legal documents (GRI standards, contracts, regulations).

Architecture:
- Entity Extraction: LLM-based extraction of legal entities (Standards, Organizations, Dates, etc.)
- Relationship Extraction: LLM-based extraction of semantic relationships
- Graph Construction: Neo4j or SimpleGraphStore backend
- Integration: Seamless integration with PHASE 1-4 RAG pipeline

Usage:
    from src.graph import KnowledgeGraphPipeline, KnowledgeGraphConfig

    config = KnowledgeGraphConfig()
    kg_pipeline = KnowledgeGraphPipeline(config)

    # Build graph from phase3 chunks
    graph = kg_pipeline.build_from_chunks(chunks)
"""

from .models import Entity, EntityType, Relationship, RelationshipType, KnowledgeGraph
from .config import (
    KnowledgeGraphConfig,
    EntityExtractionConfig,
    RelationshipExtractionConfig,
    GraphStorageConfig,
    GraphBackend,
    Neo4jConfig,
)
from .entity_extractor import EntityExtractor
from .relationship_extractor import RelationshipExtractor
from .graph_builder import GraphBuilder, Neo4jGraphBuilder, SimpleGraphBuilder
from .kg_pipeline import KnowledgeGraphPipeline
from .unified_kg_manager import UnifiedKnowledgeGraphManager
from .cross_doc_detector import CrossDocumentRelationshipDetector
from .deduplicator import EntityDeduplicator

# Neo4j integration
from .exceptions import (
    Neo4jError,
    Neo4jConnectionError,
    Neo4jQueryError,
    Neo4jTimeoutError,
    Neo4jAuthenticationError,
)
from .neo4j_manager import Neo4jManager
from .health_check import check_neo4j_health

__all__ = [
    # Models
    "Entity",
    "EntityType",
    "Relationship",
    "RelationshipType",
    "KnowledgeGraph",
    # Config
    "KnowledgeGraphConfig",
    "EntityExtractionConfig",
    "RelationshipExtractionConfig",
    "GraphStorageConfig",
    "GraphBackend",
    "Neo4jConfig",
    # Extractors
    "EntityExtractor",
    "RelationshipExtractor",
    # Builders
    "GraphBuilder",
    "Neo4jGraphBuilder",
    "SimpleGraphBuilder",
    # Pipeline
    "KnowledgeGraphPipeline",
    # Unified KG (Phase 5: Cross-Document)
    "UnifiedKnowledgeGraphManager",
    "CrossDocumentRelationshipDetector",
    "EntityDeduplicator",
    # Neo4j integration
    "Neo4jError",
    "Neo4jConnectionError",
    "Neo4jQueryError",
    "Neo4jTimeoutError",
    "Neo4jAuthenticationError",
    "Neo4jManager",
    "check_neo4j_health",
]
