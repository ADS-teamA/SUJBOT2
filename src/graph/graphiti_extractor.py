"""
Graphiti-based Knowledge Graph Extractor for SUJBOT2.

Replaces GeminiKGExtractor with Graphiti framework for temporal KG extraction.
Uses GPT-4o-mini for entity/relationship extraction with custom Pydantic types.

Features:
- Episode-based ingestion (each chunk = episode)
- 55 custom entity types for Czech legal/nuclear domain
- Bi-temporal model (valid_at + created_at)
- LLM-driven entity resolution and deduplication
- Parallel batch processing (~10 concurrent) for performance
- Bidirectional chunk-entity linking

Architecture:
    Chunks → Episodes → Graphiti → Neo4j
       ↓
    PostgreSQL (chunk_entity_mentions)
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from src.graph.graphiti_types import (
    ENTITY_TYPE_TO_MODEL,
    BezpecnostniDokumentaceEntity,
    ComplianceGapEntity,
    DocumentationRequirementEntity,
    EmergencyProcedureEntity,
    FrequencyEntity,
    GenericEntity,
    GraphitiEntityType,
    LimitniStavEntity,
    MaintenanceActivityEntity,
    MeasurementUnitEntity,
    MetodickyPokynEntity,
    MezniHodnotaEntity,
    MitigationMeasureEntity,
    NarizeniEntity,
    NumericThresholdEntity,
    PercentageEntity,
    PressureEntity,
    RadiationActivityEntity,
    RiskFactorEntity,
    SbirkaZakonuEntity,
    SujbRozhodnutiEntity,
    TemperatureEntity,
    TimePeriodEntity,
    TrainingRequirementEntity,
    VyhlaskaEntity,
    get_all_entity_types,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class ExtractedEntity:
    """Entity extracted from a chunk via Graphiti."""

    uuid: str
    name: str
    labels: List[str]
    summary: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_chunk_id: str = ""

    def __post_init__(self):
        """Validate entity data."""
        if not self.uuid:
            raise ValueError("Entity uuid cannot be empty")
        if not self.name:
            raise ValueError("Entity name cannot be empty")
        if not self.labels:
            raise ValueError("Entity must have at least one label")


@dataclass
class ExtractedRelationship:
    """Relationship extracted from a chunk via Graphiti."""

    uuid: str
    source_uuid: str
    target_uuid: str
    name: str  # Relationship type (e.g., "WORKS_FOR")
    fact: str  # Human-readable description
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    source_chunk_id: str = ""

    def __post_init__(self):
        """Validate relationship data."""
        if not self.uuid:
            raise ValueError("Relationship uuid cannot be empty")
        if not self.source_uuid:
            raise ValueError("Relationship source_uuid cannot be empty")
        if not self.target_uuid:
            raise ValueError("Relationship target_uuid cannot be empty")
        if not self.name:
            raise ValueError("Relationship name cannot be empty")
        if not self.fact:
            raise ValueError("Relationship fact cannot be empty")
        if self.valid_at and self.invalid_at and self.invalid_at < self.valid_at:
            raise ValueError(
                f"invalid_at ({self.invalid_at}) cannot be before valid_at ({self.valid_at})"
            )


@dataclass
class ChunkExtractionResult:
    """Result of extracting entities/relationships from a single chunk."""

    chunk_id: str
    episode_uuid: str
    entities: List[ExtractedEntity] = field(default_factory=list)
    relationships: List[ExtractedRelationship] = field(default_factory=list)
    processing_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class GraphitiExtractionResult:
    """Complete result of extracting KG from all chunks."""

    document_id: str
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    total_entities: int
    total_relationships: int
    unique_entity_count: int
    chunk_results: List[ChunkExtractionResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    entity_type_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Validate extraction result consistency."""
        if self.successful_chunks + self.failed_chunks != self.total_chunks:
            raise ValueError(
                f"Chunk count mismatch: {self.successful_chunks} + {self.failed_chunks} != {self.total_chunks}"
            )
        if self.unique_entity_count > self.total_entities:
            raise ValueError(
                f"unique_entity_count ({self.unique_entity_count}) cannot exceed total_entities ({self.total_entities})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "total_chunks": self.total_chunks,
            "successful_chunks": self.successful_chunks,
            "failed_chunks": self.failed_chunks,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
            "unique_entity_count": self.unique_entity_count,
            "processing_time_ms": self.processing_time_ms,
            "entity_type_counts": self.entity_type_counts,
        }


# =============================================================================
# GRAPHITI EXTRACTOR
# =============================================================================


class GraphitiExtractor:
    """
    Graphiti-based KG extractor replacing GeminiKGExtractor.

    Features:
    - Episode-based ingestion (each chunk = episode)
    - GPT-4o-mini with custom Pydantic entity types
    - Bi-temporal model (valid_at + created_at)
    - LLM-driven entity resolution and deduplication
    - Parallel batch processing for performance
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: str = "neo4j",
        neo4j_password: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 10,
        enable_temporal: bool = True,
        enable_deduplication: bool = True,
    ):
        """
        Initialize GraphitiExtractor.

        Args:
            neo4j_uri: Neo4j bolt URI (default from env NEO4J_URI or bolt://localhost:7687)
            neo4j_user: Neo4j username (default: neo4j)
            neo4j_password: Neo4j password (default from env NEO4J_PASSWORD)
            model_name: LLM model for extraction (default from env GRAPHITI_MODEL or gpt-4o-mini)
            batch_size: Number of concurrent chunk extractions (default: 10)
            enable_temporal: Enable bi-temporal tracking (default: True)
            enable_deduplication: Enable entity deduplication (default: True)
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.model_name = model_name or os.getenv("GRAPHITI_MODEL", "gpt-4o-mini")
        self.batch_size = batch_size
        self.enable_temporal = enable_temporal
        self.enable_deduplication = enable_deduplication

        # Lazy initialization
        self._graphiti = None
        self._initialized = False

        # Custom entity types for extraction
        self._entity_types = self._build_entity_types()

        logger.info(
            f"GraphitiExtractor initialized: model={self.model_name}, "
            f"batch_size={batch_size}, entity_types={len(self._entity_types)}"
        )

    def _build_entity_types(self) -> Dict[str, Type[BaseModel]]:
        """
        Build entity type mapping for Graphiti.

        Returns dict mapping type name to Pydantic model class.
        """
        entity_types = {}

        # Add specialized models from ENTITY_TYPE_TO_MODEL
        for entity_type, model_class in ENTITY_TYPE_TO_MODEL.items():
            # Use the enum value as key (e.g., "vyhlaska" not "VYHLASKA")
            entity_types[entity_type.value.title().replace("_", "")] = model_class

        # Add GenericEntity for types without specialized models
        for entity_type in get_all_entity_types():
            type_name = entity_type.value.title().replace("_", "")
            if type_name not in entity_types:
                # Create dynamic subclass with fixed type
                entity_types[type_name] = GenericEntity

        logger.debug(f"Built {len(entity_types)} entity types for Graphiti")
        return entity_types

    async def initialize(self) -> None:
        """
        Lazy async initialization of Graphiti client.

        Creates connection to Neo4j and builds necessary indices/constraints.
        Must be called before extract_from_chunks().
        """
        if self._initialized:
            return

        try:
            # Import here to avoid import errors if graphiti-core not installed
            from graphiti_core import Graphiti

            logger.info(f"Initializing Graphiti connection to {self.neo4j_uri}")

            self._graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            )

            # Build indices and constraints for efficient queries
            await self._graphiti.build_indices_and_constraints()

            self._initialized = True
            logger.info("Graphiti initialized successfully")

        except ImportError as e:
            logger.error(f"graphiti-core not installed: {e}")
            raise RuntimeError(
                "graphiti-core package required. Install with: uv add graphiti-core"
            ) from e
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise RuntimeError(
                f"Neo4j connection failed: {e}. "
                "Verify Neo4j is running (docker compose up neo4j) and "
                "NEO4J_URI/NEO4J_PASSWORD are set in .env file."
            ) from e
        except ValueError as e:
            logger.error(f"Invalid Graphiti configuration: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}", exc_info=True)
            raise RuntimeError(f"Graphiti initialization failed: {e}") from e

    async def close(self) -> None:
        """Close Graphiti connection."""
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None
            self._initialized = False
            logger.info("Graphiti connection closed")

    async def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        reference_time: Optional[datetime] = None,
    ) -> GraphitiExtractionResult:
        """
        Extract entities and relationships from chunks using episode ingestion.

        Each chunk becomes a Graphiti episode with:
        - episode_body: chunk raw_content
        - group_id: document_id for provenance
        - reference_time: document timestamp for temporal queries
        - entity_types: custom Pydantic schemas for extraction

        Args:
            chunks: List of chunk dicts with 'chunk_id', 'raw_content', etc.
            document_id: Document ID for grouping
            reference_time: Document timestamp (default: now)

        Returns:
            GraphitiExtractionResult with all extracted entities/relationships
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now(timezone.utc)
        reference_time = reference_time or start_time

        logger.info(
            f"Starting Graphiti extraction for {len(chunks)} chunks from {document_id}"
        )

        # Process chunks in parallel batches
        chunk_results: List[ChunkExtractionResult] = []
        all_entities: List[ExtractedEntity] = []
        all_relationships: List[ExtractedRelationship] = []

        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            logger.debug(
                f"Processing batch {batch_start // self.batch_size + 1}: "
                f"chunks {batch_start + 1}-{batch_end}"
            )

            # Process batch concurrently with asyncio.gather
            batch_tasks = [
                self._process_chunk(chunk, document_id, reference_time)
                for chunk in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    chunk_id = batch[i].get("chunk_id", f"unknown_{batch_start + i}")
                    # Log with traceback for debugging
                    logger.error(
                        f"Failed to process chunk {chunk_id}: {result}",
                        exc_info=(type(result), result, result.__traceback__) if hasattr(result, '__traceback__') else True
                    )
                    chunk_results.append(
                        ChunkExtractionResult(
                            chunk_id=chunk_id,
                            episode_uuid="",
                            error=str(result),
                        )
                    )
                else:
                    chunk_results.append(result)
                    all_entities.extend(result.entities)
                    all_relationships.extend(result.relationships)

        # Compute statistics
        successful_chunks = sum(1 for r in chunk_results if r.error is None)
        failed_chunks = len(chunk_results) - successful_chunks

        # Count unique entities by UUID
        unique_entity_uuids = set(e.uuid for e in all_entities)

        # Count entities by type
        entity_type_counts: Dict[str, int] = {}
        for entity in all_entities:
            for label in entity.labels:
                if label != "Entity":  # Skip generic label
                    entity_type_counts[label] = entity_type_counts.get(label, 0) + 1

        processing_time_ms = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        result = GraphitiExtractionResult(
            document_id=document_id,
            total_chunks=len(chunks),
            successful_chunks=successful_chunks,
            failed_chunks=failed_chunks,
            total_entities=len(all_entities),
            total_relationships=len(all_relationships),
            unique_entity_count=len(unique_entity_uuids),
            chunk_results=chunk_results,
            processing_time_ms=processing_time_ms,
            entity_type_counts=entity_type_counts,
        )

        logger.info(
            f"Graphiti extraction complete: {result.unique_entity_count} unique entities, "
            f"{result.total_relationships} relationships in {processing_time_ms:.0f}ms"
        )

        return result

    async def _process_chunk(
        self,
        chunk: Dict[str, Any],
        document_id: str,
        reference_time: datetime,
    ) -> ChunkExtractionResult:
        """
        Process a single chunk as a Graphiti episode.

        Args:
            chunk: Chunk dict with 'chunk_id', 'raw_content', etc.
            document_id: Document ID for grouping
            reference_time: Reference time for temporal queries

        Returns:
            ChunkExtractionResult with extracted entities/relationships
        """
        from graphiti_core.nodes import EpisodeType

        chunk_id = chunk.get("chunk_id", "unknown")
        raw_content = chunk.get("raw_content", chunk.get("content", ""))

        if not raw_content or len(raw_content.strip()) < 10:
            return ChunkExtractionResult(
                chunk_id=chunk_id,
                episode_uuid="",
                error="Chunk content too short",
            )

        start_time = datetime.now(timezone.utc)

        try:
            # Add chunk as episode
            result = await self._graphiti.add_episode(
                name=f"chunk_{chunk_id}",
                episode_body=raw_content,
                source=EpisodeType.text,
                source_description=f"Document chunk from {document_id}",
                reference_time=reference_time,
                group_id=document_id,
                entity_types=self._entity_types,
            )

            # Convert Graphiti result to our dataclasses
            entities = []
            for node in result.nodes:
                entities.append(
                    ExtractedEntity(
                        uuid=node.uuid,
                        name=node.name,
                        labels=node.labels,
                        summary=getattr(node, "summary", None),
                        attributes=getattr(node, "attributes", {}),
                        source_chunk_id=chunk_id,
                    )
                )

            relationships = []
            for edge in result.edges:
                relationships.append(
                    ExtractedRelationship(
                        uuid=edge.uuid,
                        source_uuid=edge.source_node_uuid,
                        target_uuid=edge.target_node_uuid,
                        name=edge.name,
                        fact=edge.fact,
                        valid_at=getattr(edge, "valid_at", None),
                        invalid_at=getattr(edge, "invalid_at", None),
                        source_chunk_id=chunk_id,
                    )
                )

            processing_time_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return ChunkExtractionResult(
                chunk_id=chunk_id,
                episode_uuid=result.episode.uuid,
                entities=entities,
                relationships=relationships,
                processing_time_ms=processing_time_ms,
            )

        except (ConnectionError, TimeoutError) as e:
            # Expected network errors - log and continue
            logger.warning(f"Network error processing chunk {chunk_id}: {e}")
            return ChunkExtractionResult(
                chunk_id=chunk_id,
                episode_uuid="",
                error=f"Network error: {str(e)}",
            )
        except Exception as e:
            # Unexpected errors - log with traceback for debugging
            logger.error(f"Unexpected error processing chunk {chunk_id}: {e}", exc_info=True)
            return ChunkExtractionResult(
                chunk_id=chunk_id,
                episode_uuid="",
                error=str(e),
            )

    async def extract_from_phase3(
        self,
        phase3_path: Path,
        document_id: Optional[str] = None,
        reference_time: Optional[datetime] = None,
    ) -> GraphitiExtractionResult:
        """
        Extract KG from phase3_chunks.json file.

        Convenience method that loads chunks from JSON and calls extract_from_chunks().

        Args:
            phase3_path: Path to phase3_chunks.json
            document_id: Document ID (default: derived from filename)
            reference_time: Document timestamp

        Returns:
            GraphitiExtractionResult
        """
        if not phase3_path.exists():
            raise FileNotFoundError(f"Phase 3 chunks not found: {phase3_path}")

        with open(phase3_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, list):
            chunks = data
        elif isinstance(data, dict):
            chunks = data.get("chunks", data.get("items", []))
        else:
            raise ValueError(f"Unexpected phase3 format: {type(data)}")

        # Derive document_id from filename if not provided
        if document_id is None:
            document_id = phase3_path.stem.replace("_phase3_chunks", "")

        logger.info(f"Loaded {len(chunks)} chunks from {phase3_path}")

        return await self.extract_from_chunks(
            chunks=chunks,
            document_id=document_id,
            reference_time=reference_time,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def extract_kg_with_graphiti(
    chunks: List[Dict[str, Any]],
    document_id: str,
    reference_time: Optional[datetime] = None,
    batch_size: int = 10,
) -> GraphitiExtractionResult:
    """
    Convenience function for one-shot KG extraction.

    Creates temporary GraphitiExtractor, extracts, and cleans up.

    Args:
        chunks: List of chunk dicts
        document_id: Document ID
        reference_time: Document timestamp
        batch_size: Concurrent batch size

    Returns:
        GraphitiExtractionResult
    """
    extractor = GraphitiExtractor(batch_size=batch_size)
    try:
        await extractor.initialize()
        return await extractor.extract_from_chunks(
            chunks=chunks,
            document_id=document_id,
            reference_time=reference_time,
        )
    finally:
        await extractor.close()


def sync_extract_kg_with_graphiti(
    chunks: List[Dict[str, Any]],
    document_id: str,
    reference_time: Optional[datetime] = None,
    batch_size: int = 10,
) -> GraphitiExtractionResult:
    """
    Synchronous wrapper for extract_kg_with_graphiti.

    For use in non-async contexts.
    """
    return asyncio.run(
        extract_kg_with_graphiti(
            chunks=chunks,
            document_id=document_id,
            reference_time=reference_time,
            batch_size=batch_size,
        )
    )
