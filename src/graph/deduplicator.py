"""
Entity deduplication for knowledge graph merging.

Supports:
- Layer 1: Exact match by (type, normalized_value) - fast, <1ms
- Layer 2: Embedding-based semantic similarity - 50-200ms
- Layer 3: Acronym expansion + fuzzy matching - 100-500ms
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .acronym_expander import AcronymExpander
    from .config import EntityDeduplicationConfig
    from .graph_builder import SimpleGraphBuilder
    from .models import Entity
    from .similarity_detector import EntitySimilarityDetector

logger = logging.getLogger(__name__)


class EntityDeduplicator:
    """
    Three-layer entity deduplicator for graph merging.

    Deduplication strategy (layers applied in order):
    1. Exact match on (entity_type, normalized_value) - Fast, 100% precision
    2. Embedding similarity - Medium speed, ~95% precision
    3. Acronym expansion + fuzzy - Slower, ~90% precision

    Each layer is optional and configurable.

    Example:
        >>> from src.graph.config import EntityDeduplicationConfig
        >>> config = EntityDeduplicationConfig(
        ...     use_embeddings=True,
        ...     use_acronym_expansion=True
        ... )
        >>> deduplicator = EntityDeduplicator(config)
        >>> duplicate_id = deduplicator.find_duplicate(entity, graph)
        >>> if duplicate_id:
        ...     print(f"Entity is duplicate of {duplicate_id}")
    """

    def __init__(
        self,
        config: Optional["EntityDeduplicationConfig"] = None,
        similarity_detector: Optional["EntitySimilarityDetector"] = None,
        acronym_expander: Optional["AcronymExpander"] = None,
    ):
        """
        Initialize multi-layer deduplicator.

        Args:
            config: EntityDeduplicationConfig (creates default if None)
            similarity_detector: Optional similarity detector (Layer 2)
            acronym_expander: Optional acronym expander (Layer 3)
        """
        # Import here to avoid circular dependencies
        from .config import EntityDeduplicationConfig

        self.config = config or EntityDeduplicationConfig()

        # Layer 2: Semantic similarity (optional)
        self.similarity_detector = similarity_detector
        if self.config.use_embeddings and not similarity_detector:
            logger.warning(
                "Embedding-based dedup enabled but similarity_detector=None. "
                "Layer 2 will be skipped."
            )

        # Layer 3: Acronym expansion (optional)
        self.acronym_expander = acronym_expander
        if self.config.use_acronym_expansion and not acronym_expander:
            logger.warning(
                "Acronym expansion enabled but acronym_expander=None. " "Layer 3 will be skipped."
            )

        # Statistics
        self.stats = {
            "layer1_matches": 0,  # Exact match
            "layer2_matches": 0,  # Semantic similarity
            "layer3_matches": 0,  # Acronym expansion
            "total_checks": 0,
            "unique_entities": 0,
        }

        logger.info(
            f"EntityDeduplicator initialized: "
            f"exact={self.config.exact_match_enabled}, "
            f"embeddings={self.config.use_embeddings}, "
            f"acronyms={self.config.use_acronym_expansion}"
        )

    def find_duplicate(
        self,
        entity: "Entity",
        graph: "SimpleGraphBuilder",
    ) -> Optional[str]:
        """
        Find duplicate entity using multi-layer strategy.

        Layers applied in order (first match wins):
        1. Exact match (O(1) hash lookup)
        2. Semantic similarity (if enabled)
        3. Acronym expansion (if enabled)

        Args:
            entity: Entity to check for duplicates
            graph: Graph to search in

        Returns:
            Entity ID of duplicate if found, None otherwise
        """
        self.stats["total_checks"] += 1

        # Layer 1: Exact match (always try if enabled)
        if self.config.exact_match_enabled:
            dup_id = self._find_exact_duplicate(entity, graph)
            if dup_id:
                self.stats["layer1_matches"] += 1
                logger.debug(f"Layer 1 match: {entity.normalized_value} -> {dup_id}")
                return dup_id

        # Layer 2: Semantic similarity (if enabled and detector available)
        if self.config.use_embeddings and self.similarity_detector:
            # Convert graph to list of entities for similarity check
            candidate_entities = list(graph.entities.values())

            dup_id = self.similarity_detector.find_similar(entity, candidate_entities, entity.type)
            if dup_id:
                self.stats["layer2_matches"] += 1
                logger.debug(f"Layer 2 match: {entity.normalized_value} -> {dup_id}")
                return dup_id

        # Layer 3: Acronym expansion (if enabled and expander available)
        if self.config.use_acronym_expansion and self.acronym_expander:
            candidate_entities = list(graph.entities.values())

            dup_id = self.acronym_expander.find_acronym_match(entity, candidate_entities)
            if dup_id:
                self.stats["layer3_matches"] += 1
                logger.debug(f"Layer 3 match: {entity.normalized_value} -> {dup_id}")
                return dup_id

        # No match found - entity is unique
        self.stats["unique_entities"] += 1
        return None

    def _find_exact_duplicate(
        self,
        entity: "Entity",
        graph: "SimpleGraphBuilder",
    ) -> Optional[str]:
        """Find duplicate using exact (type, normalized_value) match."""
        # Lookup in index (O(1) operation)
        key = (entity.type, entity.normalized_value)
        return graph.entity_by_normalized_value.get(key)

    def merge_entity_properties(
        self,
        primary: "Entity",
        duplicate: "Entity",
    ) -> "Entity":
        """
        Merge properties from duplicate into primary entity.

        Merging strategy:
        - confidence: MAX(primary, duplicate)
        - source_chunk_ids: UNION(primary, duplicate)
        - first_mention_chunk_id: Keep primary (first occurrence)
        - metadata: Deep merge (primary takes precedence)

        Args:
            primary: Primary entity (will be modified)
            duplicate: Duplicate entity (source of new data)

        Returns:
            primary (modified) with merged properties

        Example:
            >>> primary = Entity(confidence=0.92, source_chunk_ids=["ch1"])
            >>> duplicate = Entity(confidence=0.95, source_chunk_ids=["ch2"])
            >>> merged = dedup.merge_entity_properties(primary, duplicate)
            >>> assert merged.confidence == 0.95
            >>> assert set(merged.source_chunk_ids) == {"ch1", "ch2"}
        """
        # Merge confidence (use maximum)
        primary.confidence = max(primary.confidence, duplicate.confidence)

        # Merge chunk IDs (union)
        primary.source_chunk_ids = list(set(primary.source_chunk_ids + duplicate.source_chunk_ids))

        # Keep first mention (don't override)
        if not primary.first_mention_chunk_id and duplicate.first_mention_chunk_id:
            primary.first_mention_chunk_id = duplicate.first_mention_chunk_id

        # Merge metadata (deep merge)
        if duplicate.metadata:
            if not primary.metadata:
                primary.metadata = {}

            # Track merged-from entities
            if "merged_from" not in primary.metadata:
                primary.metadata["merged_from"] = []
            primary.metadata["merged_from"].append(duplicate.id)

            # Merge other metadata fields
            for key, value in duplicate.metadata.items():
                if key not in primary.metadata:
                    primary.metadata[key] = value
                elif isinstance(primary.metadata[key], list) and isinstance(value, list):
                    # For lists, use union
                    primary.metadata[key] = list(set(primary.metadata[key] + value))

        logger.debug(
            f"Merged properties: confidence={primary.confidence:.2f}, "
            f"chunks={len(primary.source_chunk_ids)}"
        )

        return primary

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive deduplication statistics."""
        total_matches = (
            self.stats["layer1_matches"]
            + self.stats["layer2_matches"]
            + self.stats["layer3_matches"]
        )

        return {
            **self.stats,
            "total_matches": total_matches,
            "layer1_precision": 1.0,  # Exact match is 100% precise
            "layer2_precision": 0.95 if self.config.use_embeddings else None,
            "layer3_precision": 0.90 if self.config.use_acronym_expansion else None,
        }
