"""
Knowledge Graph module for legal document processing.

Implements entity extraction, relationship extraction, and graph construction
for creating knowledge graphs from legal documents (GRI standards, contracts, regulations).

Architecture:
- Graphiti: Temporal KG extraction with GPT-4o-mini (primary)
- Graph Construction: Neo4j backend with Graphiti indices
- Integration: Seamless integration with PHASE 1-4 RAG pipeline

Usage:
    from src.graph import GraphitiExtractor, GraphitiExtractionResult

    extractor = GraphitiExtractor()
    result = await extractor.extract_from_phase3(phase3_path)
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
from .graph_builder import GraphBuilder, Neo4jGraphBuilder, SimpleGraphBuilder
from .unified_kg_manager import UnifiedKnowledgeGraphManager
from .cross_doc_detector import CrossDocumentRelationshipDetector
from .deduplicator import EntityDeduplicator

# Graphiti-based extraction (primary)
from .graphiti_extractor import (
    GraphitiExtractor,
    GraphitiExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    ChunkExtractionResult,
)
from .graphiti_types import GraphitiEntityType, get_all_entity_types

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
    # Graphiti Extractor (primary)
    "GraphitiExtractor",
    "GraphitiExtractionResult",
    "ExtractedEntity",
    "ExtractedRelationship",
    "ChunkExtractionResult",
    "GraphitiEntityType",
    "get_all_entity_types",
    # Builders
    "GraphBuilder",
    "Neo4jGraphBuilder",
    "SimpleGraphBuilder",
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
