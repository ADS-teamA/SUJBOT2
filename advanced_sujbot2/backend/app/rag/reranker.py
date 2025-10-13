"""
Reranking Module for Legal Compliance System

This module implements a multi-stage reranking pipeline that combines:
1. Cross-encoder semantic relevance scoring (multilingual)
2. Graph-aware structural scoring (proximity, centrality, authority)
3. Legal precedence weighting (constitutional > statutory > regulatory > contractual)
4. Ensemble fusion for robust final ranking

Key Components:
- CrossEncoderReranker: Deep semantic relevance using mmarco-mMiniLMv2-L12-H384-v1
- GraphAwareReranker: Graph structure-based scoring
- LegalPrecedenceReranker: Legal hierarchy and temporal precedence
- EnsembleFusion: Weighted score combination
- RerankingPipeline: Orchestrates all rerankers

Based on specification: 07_reranking.md
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from sentence_transformers import CrossEncoder

try:
    from .device_utils import get_device
except ImportError:
    # Fallback if device_utils not available
    def get_device(device_str: str) -> str:
        """Simple fallback device selection"""
        if device_str == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device_str


# ============================================================================
# Data Structures
# ============================================================================

class PrecedenceLevel(Enum):
    """Legal hierarchy levels (Czech legal system)."""
    CONSTITUTIONAL = 5  # Ústava, Listina základních práv
    STATUTORY = 4       # Zákony
    REGULATORY = 3      # Vyhlášky, nařízení vlády
    CONTRACTUAL = 2     # Smlouvy
    GUIDANCE = 1        # Metodiky, doporučení


@dataclass
class RerankingScores:
    """Individual scores from different rerankers."""
    cross_encoder_score: float  # 0.0 to 1.0 (normalized)
    graph_score: float          # 0.0 to 1.0
    precedence_score: float     # 0.0 to 1.0
    ensemble_score: float       # 0.0 to 1.0 (final)

    # Score metadata
    cross_encoder_confidence: float = 0.0  # 0.0 to 1.0
    graph_features: Dict[str, float] = field(default_factory=dict)
    precedence_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search result from hybrid retrieval (input to reranking)."""
    chunk_id: str
    content: str
    legal_reference: str = ""
    document_id: str = ""
    document_type: str = "law_code"  # law_code | contract | regulation
    hierarchy_path: str = ""
    rank: int = 0
    hybrid_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedResult:
    """Final reranked result with all scoring information."""
    chunk_id: str
    content: str
    legal_reference: str
    document_id: str
    document_type: str  # 'law_code' | 'contract'

    # Scores
    scores: RerankingScores
    final_rank: int  # 1 to K

    # Confidence and explanation
    confidence: float  # 0.0 to 1.0
    reranking_explanation: str  # Human-readable explanation

    # Original retrieval metadata
    original_rank: int
    original_hybrid_score: float

    # Graph context
    graph_neighbors: List[str] = field(default_factory=list)
    reference_path: Optional[List[str]] = None

    @property
    def rank_improvement(self) -> int:
        """How many positions this result moved up."""
        return self.original_rank - self.final_rank


@dataclass
class RerankingConfig:
    """Configuration for reranking pipeline."""

    # Cross-encoder settings
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cross_encoder_batch_size: int = 16
    cross_encoder_device: str = "cpu"  # cpu | cuda | mps
    cross_encoder_max_length: int = 512

    # Graph-aware settings
    enable_graph_reranking: bool = True
    graph_proximity_weight: float = 0.4
    graph_centrality_weight: float = 0.3
    graph_authority_weight: float = 0.3
    max_hop_distance: int = 3  # For proximity calculation

    # Legal precedence settings
    enable_precedence_weighting: bool = True
    precedence_weights: Dict[str, float] = field(default_factory=dict)
    temporal_decay_factor: float = 0.95  # Newer laws slightly favored

    # Ensemble settings
    ensemble_method: str = "weighted_average"  # weighted_average | borda_count | rrf
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

    # Output settings
    final_top_k: int = 5
    min_confidence_threshold: float = 0.1
    explain_reranking: bool = True

    # Score calibration (for cross-encoder normalization)
    # Min-max normalization bounds for mMiniLMv2 model
    # Typical range: -2.0 to +2.0 (can be adjusted based on empirical data)
    cross_encoder_score_min: float = -3.0  # Conservative lower bound
    cross_encoder_score_max: float = 3.0   # Conservative upper bound

    # Legacy sigmoid normalization parameters (deprecated)
    cross_encoder_score_mean: float = 0.0
    cross_encoder_score_std: float = 5.0

    def __post_init__(self):
        """Set defaults for dict fields."""
        if not self.precedence_weights:
            self.precedence_weights = {
                "constitutional": 1.0,
                "statutory": 0.9,
                "regulatory": 0.7,
                "contractual": 0.5,
                "guidance": 0.3
            }

        if not self.ensemble_weights:
            self.ensemble_weights = {
                "cross_encoder": 0.5,
                "graph": 0.3,
                "precedence": 0.2
            }

    @classmethod
    def from_yaml(cls, config_dict: Dict[str, Any]) -> 'RerankingConfig':
        """Create config from YAML dictionary."""
        reranking_config = config_dict.get("reranking", {})
        return cls(
            cross_encoder_model=reranking_config.get(
                "cross_encoder_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            ),
            cross_encoder_batch_size=reranking_config.get("cross_encoder_batch_size", 16),
            cross_encoder_device=reranking_config.get("cross_encoder_device", "cpu"),
            cross_encoder_max_length=reranking_config.get("cross_encoder_max_length", 512),
            enable_graph_reranking=reranking_config.get("enable_graph_reranking", True),
            graph_proximity_weight=reranking_config.get("graph_proximity_weight", 0.4),
            graph_centrality_weight=reranking_config.get("graph_centrality_weight", 0.3),
            graph_authority_weight=reranking_config.get("graph_authority_weight", 0.3),
            max_hop_distance=reranking_config.get("max_hop_distance", 3),
            enable_precedence_weighting=reranking_config.get("enable_precedence_weighting", True),
            precedence_weights=reranking_config.get("precedence_weights", {}),
            temporal_decay_factor=reranking_config.get("temporal_decay_factor", 0.95),
            ensemble_method=reranking_config.get("ensemble_method", "weighted_average"),
            ensemble_weights=reranking_config.get("ensemble_weights", {}),
            final_top_k=reranking_config.get("final_top_k", 5),
            min_confidence_threshold=reranking_config.get("min_confidence_threshold", 0.1),
            explain_reranking=reranking_config.get("explain_reranking", True),
        )


# ============================================================================
# Cross-Encoder Reranker
# ============================================================================

class CrossEncoderReranker:
    """Rerank results using multilingual cross-encoder.

    Uses mmarco-mMiniLMv2-L12-H384-v1 for deep semantic relevance scoring.
    This model provides much better relevance assessment than bi-encoders
    because it sees both query and document together.
    """

    def __init__(self, config: RerankingConfig):
        """Initialize cross-encoder reranker.

        Args:
            config: Reranking configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Resolve device (convert "auto" to actual device)
        self.device = get_device(config.cross_encoder_device)

        try:
            self.model = CrossEncoder(
                config.cross_encoder_model,
                device=self.device,
                max_length=config.cross_encoder_max_length
            )
            self.logger.info(
                f"Loaded cross-encoder model: {config.cross_encoder_model} "
                f"on device: {self.device}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {e}")
            raise

        # Score normalization params (min-max bounds for mMiniLMv2)
        self.score_min = config.cross_encoder_score_min
        self.score_max = config.cross_encoder_score_max

        # Legacy sigmoid params (for backward compatibility)
        self.score_mean = config.cross_encoder_score_mean
        self.score_std = config.cross_encoder_score_std

    async def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float]]:
        """Rerank search results using cross-encoder.

        Args:
            query: User query
            results: Initial search results

        Returns:
            List of (result, cross_encoder_score) sorted by score descending
        """
        if not results:
            return []

        self.logger.debug(f"Cross-encoder reranking {len(results)} results")

        # Prepare query-document pairs
        pairs = []
        for result in results:
            # Construct input with legal context
            document_text = self._prepare_document_text(result)
            pairs.append([query, document_text])

        # Batch prediction (run in thread pool to avoid blocking)
        try:
            scores = await asyncio.to_thread(
                self.model.predict,
                pairs,
                batch_size=self.config.cross_encoder_batch_size,
                show_progress_bar=False
            )
        except Exception as e:
            self.logger.error(f"Cross-encoder prediction failed: {e}")
            # Fallback: return original order with neutral scores
            return [(r, 0.5) for r in results]

        # Normalize scores to [0, 1]
        normalized_scores = self._normalize_scores(scores)

        # Sort by score
        scored_results = list(zip(results, normalized_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        self.logger.debug(
            f"Cross-encoder scores: min={normalized_scores.min():.3f}, "
            f"max={normalized_scores.max():.3f}, mean={normalized_scores.mean():.3f}"
        )

        return scored_results

    def _prepare_document_text(self, result: SearchResult) -> str:
        """Prepare document text for cross-encoder input.

        Include legal context for better scoring:
        - Legal reference (e.g., [§89 odst. 1])
        - Hierarchy path (e.g., (Část II > Hlava III > §89))
        - Main content

        Args:
            result: Search result

        Returns:
            Formatted document text (truncated to ~500 tokens)
        """
        parts = []

        # Legal reference
        if result.legal_reference:
            parts.append(f"[{result.legal_reference}]")

        # Hierarchy context
        if result.hierarchy_path:
            parts.append(f"({result.hierarchy_path})")

        # Main content
        parts.append(result.content)

        # Truncate to avoid exceeding model max_length
        # Rough estimate: 1 token ≈ 4 chars, max_length=512 tokens ≈ 2000 chars
        text = " ".join(parts)
        max_chars = self.config.cross_encoder_max_length * 4
        return text[:max_chars]

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize cross-encoder logits to [0, 1] range.

        Uses min-max normalization with calibrated bounds.
        For mmarco-mMiniLMv2-L12-H384-v1, typical range is [-2, +2].

        Args:
            scores: Raw logit scores from cross-encoder

        Returns:
            Normalized scores in [0, 1]
        """
        # Log actual score range for calibration
        actual_min = float(scores.min())
        actual_max = float(scores.max())
        if actual_min < self.score_min or actual_max > self.score_max:
            self.logger.warning(
                f"Cross-encoder scores outside calibrated range: "
                f"actual=[{actual_min:.2f}, {actual_max:.2f}], "
                f"expected=[{self.score_min:.2f}, {self.score_max:.2f}]. "
                f"Consider adjusting cross_encoder_score_min/max in config."
            )

        # Min-max normalization with clipping
        normalized = (scores - self.score_min) / (self.score_max - self.score_min)
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized

    def get_confidence(self, score: float) -> float:
        """Estimate confidence in cross-encoder score.

        Higher absolute scores (further from decision boundary 0.5) = higher confidence.

        Args:
            score: Normalized cross-encoder score [0, 1]

        Returns:
            Confidence estimate [0, 1]
        """
        # Confidence based on distance from decision boundary (0.5)
        # score=0.0 or 1.0 → confidence=1.0
        # score=0.5 → confidence=0.0
        confidence = abs(2.0 * score - 1.0)
        return float(confidence)


# ============================================================================
# Graph-Aware Reranker
# ============================================================================

class GraphAwareReranker:
    """Rerank using knowledge graph structure.

    Computes scores based on:
    1. Proximity: Distance to query-relevant chunks in graph
    2. Centrality: Betweenness centrality (foundational provisions)
    3. Authority: PageRank-like score (highly referenced provisions)
    """

    def __init__(
        self,
        config: RerankingConfig,
        knowledge_graph: Optional['LegalKnowledgeGraph'] = None
    ):
        """Initialize graph-aware reranker.

        Args:
            config: Reranking configuration
            knowledge_graph: Legal knowledge graph (if available)
        """
        self.config = config
        self.graph = knowledge_graph
        self.logger = logging.getLogger(__name__)

        # Precompute centrality measures if graph available
        if knowledge_graph and hasattr(knowledge_graph, 'graph'):
            try:
                self._centrality_cache = self._compute_centrality()
                self._authority_cache = self._compute_authority()
                self.logger.info("Precomputed graph centrality and authority scores")
            except Exception as e:
                self.logger.warning(f"Failed to precompute graph metrics: {e}")
                self._centrality_cache = {}
                self._authority_cache = {}
        else:
            self._centrality_cache = {}
            self._authority_cache = {}
            if config.enable_graph_reranking:
                self.logger.warning("Graph reranking enabled but no knowledge graph provided")

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[SearchResult, float, Dict[str, float]]]:
        """Rerank based on graph structure.

        Args:
            query: User query (may contain legal references)
            results: Initial search results
            query_context: Optional context (e.g., anchor chunks)

        Returns:
            List of (result, graph_score, features_dict) sorted by score
        """
        if not self.graph or not hasattr(self.graph, 'graph'):
            self.logger.warning("No graph available, returning uniform scores")
            return [(r, 0.5, {}) for r in results]

        # Identify anchor chunks (highly relevant to query)
        anchor_chunks = self._identify_anchors(query, results, query_context)
        self.logger.debug(f"Identified {len(anchor_chunks)} anchor chunks")

        # Score each result
        scored_results = []
        for result in results:
            chunk_id = result.chunk_id

            # Compute graph features
            proximity_score = self._compute_proximity(chunk_id, anchor_chunks)
            centrality_score = self._centrality_cache.get(chunk_id, 0.0)
            authority_score = self._authority_cache.get(chunk_id, 0.0)

            # Weighted combination
            graph_score = (
                self.config.graph_proximity_weight * proximity_score +
                self.config.graph_centrality_weight * centrality_score +
                self.config.graph_authority_weight * authority_score
            )

            features = {
                "proximity": proximity_score,
                "centrality": centrality_score,
                "authority": authority_score
            }

            scored_results.append((result, graph_score, features))

        # Sort by graph score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results

    def _identify_anchors(
        self,
        query: str,
        results: List[SearchResult],
        query_context: Optional[Dict[str, Any]]
    ) -> Set[str]:
        """Identify anchor chunks (reference points for proximity).

        Args:
            query: User query
            results: Search results
            query_context: Optional explicit anchors

        Returns:
            Set of chunk IDs to use as anchors
        """
        anchors = set()

        # Extract legal references from query (e.g., "§89", "Zákon č. 89/2012")
        references = self._extract_references(query)
        if references and self.graph and hasattr(self.graph, 'get_chunks_by_reference'):
            for ref in references:
                chunk_ids = self.graph.get_chunks_by_reference(ref)
                anchors.update(chunk_ids)

        # Top-3 results are likely relevant
        anchors.update(r.chunk_id for r in results[:3])

        # Explicit anchors from context
        if query_context and "anchor_chunks" in query_context:
            anchors.update(query_context["anchor_chunks"])

        return anchors

    def _compute_proximity(
        self,
        chunk_id: str,
        anchor_chunks: Set[str]
    ) -> float:
        """Compute proximity score based on graph distance to anchors.

        Args:
            chunk_id: Target chunk ID
            anchor_chunks: Set of anchor chunk IDs

        Returns:
            Proximity score [0, 1] (1 = very close, 0 = far or disconnected)
        """
        if not anchor_chunks or chunk_id in anchor_chunks:
            return 1.0  # Maximum proximity

        if not self.graph or not hasattr(self.graph, 'graph'):
            return 0.5  # Neutral if no graph

        # Find shortest path to any anchor
        min_distance = float('inf')
        for anchor in anchor_chunks:
            try:
                distance = nx.shortest_path_length(
                    self.graph.graph,
                    source=chunk_id,
                    target=anchor
                )
                min_distance = min(min_distance, distance)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        # Convert distance to score (closer = higher score)
        if min_distance == float('inf'):
            return 0.0

        # Exponential decay with distance
        max_hops = self.config.max_hop_distance
        if min_distance > max_hops:
            return 0.0

        proximity_score = np.exp(-0.5 * min_distance)
        return float(proximity_score)

    def _compute_centrality(self) -> Dict[str, float]:
        """Precompute betweenness centrality for all chunks.

        Betweenness centrality measures how often a node appears on
        shortest paths between other nodes. High centrality = foundational.

        Returns:
            Dict mapping chunk_id to normalized centrality score
        """
        if not self.graph or not hasattr(self.graph, 'graph'):
            return {}

        try:
            centrality = nx.betweenness_centrality(self.graph.graph)

            # Normalize to [0, 1]
            max_centrality = max(centrality.values()) if centrality else 1.0
            if max_centrality > 0:
                normalized = {
                    node: score / max_centrality
                    for node, score in centrality.items()
                }
            else:
                normalized = {node: 0.0 for node in centrality}

            return normalized
        except Exception as e:
            self.logger.warning(f"Failed to compute centrality: {e}")
            return {}

    def _compute_authority(self) -> Dict[str, float]:
        """Compute authority scores using PageRank-like algorithm.

        Authority flows from highly-referenced chunks to their neighbors.

        Returns:
            Dict mapping chunk_id to normalized authority score
        """
        if not self.graph or not hasattr(self.graph, 'graph'):
            return {}

        try:
            # Filter to reference edges only (if method available)
            if hasattr(self.graph, 'get_subgraph_by_edge_type'):
                reference_graph = self.graph.get_subgraph_by_edge_type('REFERENCES')
            else:
                reference_graph = self.graph.graph

            # Run PageRank
            try:
                authority = nx.pagerank(reference_graph, alpha=0.85)
            except nx.PowerIterationFailedConvergence:
                # Fallback: uniform scores
                authority = {node: 1.0 for node in reference_graph.nodes()}

            # Normalize
            max_authority = max(authority.values()) if authority else 1.0
            if max_authority > 0:
                normalized = {
                    node: score / max_authority
                    for node, score in authority.items()
                }
            else:
                normalized = {node: 0.0 for node in authority}

            return normalized
        except Exception as e:
            self.logger.warning(f"Failed to compute authority: {e}")
            return {}

    def _extract_references(self, query: str) -> List[str]:
        """Extract legal references from query.

        Looks for patterns like:
        - §89
        - §89 odst. 1
        - Zákon č. 89/2012 Sb.

        Args:
            query: User query

        Returns:
            List of extracted references
        """
        import re

        references = []

        # Pattern: §89, §89 odst. 1, etc.
        paragraph_pattern = r'§\s*\d+(?:\s+odst\.\s*\d+)?'
        references.extend(re.findall(paragraph_pattern, query))

        # Pattern: Zákon č. 89/2012 Sb.
        law_pattern = r'[Zz]ákon\s+č\.\s*\d+/\d+'
        references.extend(re.findall(law_pattern, query))

        return references


# ============================================================================
# Legal Precedence Reranker
# ============================================================================

class LegalPrecedenceReranker:
    """Rerank based on legal authority hierarchy.

    Czech legal system hierarchy (highest to lowest):
    1. Constitutional law (Ústava, Listina)
    2. Statutory law (Zákony)
    3. Regulatory law (Vyhlášky, nařízení)
    4. Contractual provisions
    5. Guidance documents
    """

    def __init__(self, config: RerankingConfig):
        """Initialize legal precedence reranker.

        Args:
            config: Reranking configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Document type → precedence level mapping
        self.type_to_level = {
            "constitution": PrecedenceLevel.CONSTITUTIONAL,
            "law_code": PrecedenceLevel.STATUTORY,
            "law": PrecedenceLevel.STATUTORY,
            "regulation": PrecedenceLevel.REGULATORY,
            "contract": PrecedenceLevel.CONTRACTUAL,
            "guidance": PrecedenceLevel.GUIDANCE,
        }

        # Convert string keys to PrecedenceLevel keys
        self.precedence_weights = {}
        for key, value in config.precedence_weights.items():
            if key == "constitutional":
                self.precedence_weights[PrecedenceLevel.CONSTITUTIONAL] = value
            elif key == "statutory":
                self.precedence_weights[PrecedenceLevel.STATUTORY] = value
            elif key == "regulatory":
                self.precedence_weights[PrecedenceLevel.REGULATORY] = value
            elif key == "contractual":
                self.precedence_weights[PrecedenceLevel.CONTRACTUAL] = value
            elif key == "guidance":
                self.precedence_weights[PrecedenceLevel.GUIDANCE] = value

    async def rerank(
        self,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float, Dict[str, float]]]:
        """Rerank based on legal precedence.

        Args:
            results: Initial search results

        Returns:
            List of (result, precedence_score, factors_dict)
        """
        scored_results = []

        for result in results:
            # Base score from hierarchy
            precedence_level = self._get_precedence_level(result)
            hierarchy_score = self.precedence_weights.get(precedence_level, 0.5)

            # Temporal adjustment (newer slightly favored)
            temporal_score = self._compute_temporal_score(result)

            # Specificity bonus (more specific = higher)
            specificity_score = self._compute_specificity(result)

            # Combined precedence score
            precedence_score = (
                0.6 * hierarchy_score +
                0.2 * temporal_score +
                0.2 * specificity_score
            )

            factors = {
                "hierarchy": hierarchy_score,
                "temporal": temporal_score,
                "specificity": specificity_score
            }

            scored_results.append((result, precedence_score, factors))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results

    def _get_precedence_level(self, result: SearchResult) -> PrecedenceLevel:
        """Determine precedence level from document type.

        Args:
            result: Search result

        Returns:
            Precedence level enum
        """
        doc_type = result.document_type.lower()
        return self.type_to_level.get(doc_type, PrecedenceLevel.GUIDANCE)

    def _compute_temporal_score(self, result: SearchResult) -> float:
        """Score based on document age.

        Newer documents get slightly higher scores (lex posterior).

        Args:
            result: Search result

        Returns:
            Temporal score [0, 1]
        """
        effective_date = result.metadata.get("effective_date")
        if not effective_date:
            return 0.5  # Neutral if date unknown

        # Handle both datetime and string dates
        if isinstance(effective_date, str):
            try:
                effective_date = datetime.fromisoformat(effective_date)
            except ValueError:
                return 0.5

        years_old = (datetime.now() - effective_date).days / 365.25

        # Exponential decay (very slow)
        temporal_score = self.config.temporal_decay_factor ** years_old
        return float(temporal_score)

    def _compute_specificity(self, result: SearchResult) -> float:
        """More specific provisions (deeper hierarchy) score higher.

        Example hierarchy depths:
        - Část I (Part) = depth 1
        - Část I > Hlava II (Chapter) = depth 2
        - Část I > Hlava II > §89 = depth 3
        - Část I > Hlava II > §89 > odst. 1 = depth 4

        Args:
            result: Search result

        Returns:
            Specificity score [0, 1]
        """
        hierarchy_path = result.hierarchy_path or ""

        # Count hierarchy depth (separator: ">")
        depth = hierarchy_path.count(">") + 1 if hierarchy_path else 0

        # Normalize to [0, 1]
        # Typical depths: 1 (Part) to 5 (Letter)
        specificity_score = min(1.0, depth / 5.0)

        return specificity_score


# ============================================================================
# Ensemble Fusion
# ============================================================================

class EnsembleFusion:
    """Combine scores from multiple rerankers.

    Supports three fusion methods:
    1. Weighted Average: Linear combination of normalized scores
    2. Borda Count: Rank-based aggregation
    3. Reciprocal Rank Fusion (RRF): Sum of 1/(k+rank)
    """

    def __init__(self, config: RerankingConfig):
        """Initialize ensemble fusion.

        Args:
            config: Reranking configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def fuse(
        self,
        cross_encoder_results: List[Tuple[SearchResult, float]],
        graph_results: List[Tuple[SearchResult, float, Dict]],
        precedence_results: List[Tuple[SearchResult, float, Dict]]
    ) -> List[RankedResult]:
        """Fuse scores from all rerankers.

        Args:
            cross_encoder_results: [(result, score), ...]
            graph_results: [(result, score, features), ...]
            precedence_results: [(result, score, factors), ...]

        Returns:
            List of RankedResult sorted by ensemble score
        """
        method = self.config.ensemble_method

        if method == "weighted_average":
            return self._weighted_average_fusion(
                cross_encoder_results,
                graph_results,
                precedence_results
            )
        elif method == "borda_count":
            return self._borda_count_fusion(
                cross_encoder_results,
                graph_results,
                precedence_results
            )
        elif method == "rrf":
            return self._rrf_fusion(
                cross_encoder_results,
                graph_results,
                precedence_results
            )
        else:
            self.logger.warning(
                f"Unknown ensemble method: {method}, using weighted_average"
            )
            return self._weighted_average_fusion(
                cross_encoder_results,
                graph_results,
                precedence_results
            )

    def _weighted_average_fusion(
        self,
        cross_encoder_results: List[Tuple[SearchResult, float]],
        graph_results: List[Tuple[SearchResult, float, Dict]],
        precedence_results: List[Tuple[SearchResult, float, Dict]]
    ) -> List[RankedResult]:
        """Weighted average of normalized scores."""
        weights = self.config.ensemble_weights

        # Build score maps
        ce_scores = {r.chunk_id: score for r, score in cross_encoder_results}
        graph_scores = {r.chunk_id: score for r, score, _ in graph_results}
        prec_scores = {r.chunk_id: score for r, score, _ in precedence_results}
        graph_features = {r.chunk_id: feats for r, _, feats in graph_results}
        prec_features = {r.chunk_id: feats for r, _, feats in precedence_results}

        # Compute ensemble scores
        ranked_results = []
        for result, ce_score in cross_encoder_results:
            chunk_id = result.chunk_id

            graph_score = graph_scores.get(chunk_id, 0.0)
            prec_score = prec_scores.get(chunk_id, 0.0)

            ensemble_score = (
                weights["cross_encoder"] * ce_score +
                weights["graph"] * graph_score +
                weights["precedence"] * prec_score
            )

            # Build RerankingScores
            scores = RerankingScores(
                cross_encoder_score=ce_score,
                graph_score=graph_score,
                precedence_score=prec_score,
                ensemble_score=ensemble_score,
                cross_encoder_confidence=self._get_ce_confidence(ce_score),
                graph_features=graph_features.get(chunk_id, {}),
                precedence_factors=prec_features.get(chunk_id, {})
            )

            # Build RankedResult
            ranked_result = RankedResult(
                chunk_id=result.chunk_id,
                content=result.content,
                legal_reference=result.legal_reference,
                document_id=result.document_id,
                document_type=result.document_type,
                scores=scores,
                final_rank=0,  # Will be assigned after sorting
                confidence=self._compute_confidence(scores),
                reranking_explanation=self._explain_reranking(result, scores),
                original_rank=result.rank,
                original_hybrid_score=result.hybrid_score,
                graph_neighbors=result.metadata.get("neighbors", []),
                reference_path=result.metadata.get("reference_path")
            )

            ranked_results.append(ranked_result)

        # Sort by ensemble score
        ranked_results.sort(key=lambda x: x.scores.ensemble_score, reverse=True)

        # Assign final ranks
        for rank, result in enumerate(ranked_results, start=1):
            result.final_rank = rank

        # Take top-K
        top_k = self.config.final_top_k
        return ranked_results[:top_k]

    def _borda_count_fusion(
        self,
        cross_encoder_results: List[Tuple[SearchResult, float]],
        graph_results: List[Tuple[SearchResult, float, Dict]],
        precedence_results: List[Tuple[SearchResult, float, Dict]]
    ) -> List[RankedResult]:
        """Rank aggregation using Borda count.

        Each reranker assigns ranks. Points = (N - rank + 1).
        Sum points across rerankers.
        """
        # Convert all result lists to rankings
        ce_ranking = {r.chunk_id: rank for rank, (r, _) in enumerate(cross_encoder_results, 1)}
        graph_ranking = {r.chunk_id: rank for rank, (r, _, _) in enumerate(graph_results, 1)}
        prec_ranking = {r.chunk_id: rank for rank, (r, _, _) in enumerate(precedence_results, 1)}

        all_rankings = [ce_ranking, graph_ranking, prec_ranking]

        # Compute Borda scores
        all_chunk_ids = set()
        for ranking in all_rankings:
            all_chunk_ids.update(ranking.keys())

        N = len(all_chunk_ids)
        borda_scores = {}

        for chunk_id in all_chunk_ids:
            points = 0
            for ranking in all_rankings:
                rank = ranking.get(chunk_id, N + 1)  # Unranked = last place
                points += (N - rank + 1)
            borda_scores[chunk_id] = points / (3 * N)  # Normalize to [0, 1]

        # Build RankedResults using borda_scores as ensemble_score
        return self._build_ranked_results(
            cross_encoder_results,
            graph_results,
            precedence_results,
            borda_scores
        )

    def _rrf_fusion(
        self,
        cross_encoder_results: List[Tuple[SearchResult, float]],
        graph_results: List[Tuple[SearchResult, float, Dict]],
        precedence_results: List[Tuple[SearchResult, float, Dict]]
    ) -> List[RankedResult]:
        """Reciprocal rank fusion.

        RRF score = Σ 1/(k + rank_i) where k=60
        """
        k = 60  # RRF constant

        # Convert to rankings
        ce_ranking = {r.chunk_id: rank for rank, (r, _) in enumerate(cross_encoder_results, 1)}
        graph_ranking = {r.chunk_id: rank for rank, (r, _, _) in enumerate(graph_results, 1)}
        prec_ranking = {r.chunk_id: rank for rank, (r, _, _) in enumerate(precedence_results, 1)}

        all_rankings = [ce_ranking, graph_ranking, prec_ranking]

        # Compute RRF scores
        all_chunk_ids = set()
        for ranking in all_rankings:
            all_chunk_ids.update(ranking.keys())

        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            score = 0
            for ranking in all_rankings:
                rank = ranking.get(chunk_id, 1000)  # Unranked = very low
                score += 1 / (k + rank)
            rrf_scores[chunk_id] = score

        # Normalize RRF scores to [0, 1]
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        if max_rrf > 0:
            rrf_scores = {k: v / max_rrf for k, v in rrf_scores.items()}

        # Build RankedResults
        return self._build_ranked_results(
            cross_encoder_results,
            graph_results,
            precedence_results,
            rrf_scores
        )

    def _build_ranked_results(
        self,
        cross_encoder_results: List[Tuple[SearchResult, float]],
        graph_results: List[Tuple[SearchResult, float, Dict]],
        precedence_results: List[Tuple[SearchResult, float, Dict]],
        ensemble_scores: Dict[str, float]
    ) -> List[RankedResult]:
        """Helper to build RankedResult objects from ensemble scores."""
        # Build score maps
        ce_scores = {r.chunk_id: score for r, score in cross_encoder_results}
        graph_scores = {r.chunk_id: score for r, score, _ in graph_results}
        prec_scores = {r.chunk_id: score for r, score, _ in precedence_results}
        graph_features = {r.chunk_id: feats for r, _, feats in graph_results}
        prec_features = {r.chunk_id: feats for r, _, feats in precedence_results}

        ranked_results = []
        for result, ce_score in cross_encoder_results:
            chunk_id = result.chunk_id

            graph_score = graph_scores.get(chunk_id, 0.0)
            prec_score = prec_scores.get(chunk_id, 0.0)
            ensemble_score = ensemble_scores.get(chunk_id, 0.0)

            scores = RerankingScores(
                cross_encoder_score=ce_score,
                graph_score=graph_score,
                precedence_score=prec_score,
                ensemble_score=ensemble_score,
                cross_encoder_confidence=self._get_ce_confidence(ce_score),
                graph_features=graph_features.get(chunk_id, {}),
                precedence_factors=prec_features.get(chunk_id, {})
            )

            ranked_result = RankedResult(
                chunk_id=result.chunk_id,
                content=result.content,
                legal_reference=result.legal_reference,
                document_id=result.document_id,
                document_type=result.document_type,
                scores=scores,
                final_rank=0,
                confidence=self._compute_confidence(scores),
                reranking_explanation=self._explain_reranking(result, scores),
                original_rank=result.rank,
                original_hybrid_score=result.hybrid_score,
                graph_neighbors=result.metadata.get("neighbors", []),
                reference_path=result.metadata.get("reference_path")
            )

            ranked_results.append(ranked_result)

        # Sort by ensemble score
        ranked_results.sort(key=lambda x: x.scores.ensemble_score, reverse=True)

        # Assign final ranks
        for rank, result in enumerate(ranked_results, start=1):
            result.final_rank = rank

        # Take top-K
        return ranked_results[:self.config.final_top_k]

    def _compute_confidence(self, scores: RerankingScores) -> float:
        """Compute overall confidence in ranking.

        Based on agreement between rerankers:
        - High mean score + low variance = high confidence
        - Low mean score or high variance = low confidence

        Args:
            scores: Reranking scores

        Returns:
            Confidence estimate [0, 1]
        """
        score_values = np.array([
            scores.cross_encoder_score,
            scores.graph_score,
            scores.precedence_score
        ])

        mean_score = float(np.mean(score_values))
        std_score = float(np.std(score_values))

        # High mean and low std = high confidence
        confidence = mean_score * (1 - std_score / (mean_score + 1e-6))
        confidence = np.clip(confidence, 0.0, 1.0)

        return float(confidence)

    def _explain_reranking(
        self,
        result: SearchResult,
        scores: RerankingScores
    ) -> str:
        """Generate human-readable explanation of reranking.

        Args:
            result: Search result
            scores: Reranking scores

        Returns:
            Human-readable explanation string
        """
        if not self.config.explain_reranking:
            return ""

        parts = []

        # Dominant score
        score_components = [
            ("semantic relevance", scores.cross_encoder_score),
            ("graph structure", scores.graph_score),
            ("legal precedence", scores.precedence_score)
        ]
        max_component = max(score_components, key=lambda x: x[1])

        parts.append(
            f"Ranked highly due to {max_component[0]} (score: {max_component[1]:.2f})"
        )

        # Graph features
        if scores.graph_features:
            if scores.graph_features.get("proximity", 0) > 0.7:
                parts.append("close proximity to query-relevant provisions")
            if scores.graph_features.get("centrality", 0) > 0.7:
                parts.append("foundational provision in legal structure")
            if scores.graph_features.get("authority", 0) > 0.7:
                parts.append("highly referenced by other provisions")

        # Precedence
        if scores.precedence_score > 0.8:
            parts.append("high legal authority")

        return "; ".join(parts)

    def _get_ce_confidence(self, score: float) -> float:
        """Confidence in cross-encoder score.

        Higher absolute deviation from 0.5 = higher confidence.

        Args:
            score: Normalized cross-encoder score [0, 1]

        Returns:
            Confidence [0, 1]
        """
        # 0.5 → 0 confidence, 0 or 1 → 1 confidence
        return float(abs(2 * score - 1))


# ============================================================================
# Reranking Pipeline
# ============================================================================

class RerankingPipeline:
    """Orchestrate all reranking components.

    Workflow:
    1. Cross-encoder reranking (semantic relevance)
    2. Graph-aware reranking (structural features)
    3. Legal precedence reranking (hierarchy)
    4. Ensemble fusion (combine all signals)
    5. Filter by confidence threshold
    6. Return top-K results
    """

    def __init__(
        self,
        config: RerankingConfig,
        knowledge_graph: Optional['LegalKnowledgeGraph'] = None
    ):
        """Initialize reranking pipeline.

        Args:
            config: Reranking configuration
            knowledge_graph: Legal knowledge graph (optional)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize rerankers
        self.cross_encoder = CrossEncoderReranker(config)
        self.logger.info("Initialized CrossEncoderReranker")

        if config.enable_graph_reranking:
            self.graph_reranker = GraphAwareReranker(config, knowledge_graph)
            self.logger.info("Initialized GraphAwareReranker")
        else:
            self.graph_reranker = None
            self.logger.info("Graph reranking disabled")

        if config.enable_precedence_weighting:
            self.precedence_reranker = LegalPrecedenceReranker(config)
            self.logger.info("Initialized LegalPrecedenceReranker")
        else:
            self.precedence_reranker = None
            self.logger.info("Legal precedence reranking disabled")

        self.fusion = EnsembleFusion(config)
        self.logger.info(f"Initialized EnsembleFusion (method: {config.ensemble_method})")

    async def rerank(
        self,
        query: str,
        initial_results: List[SearchResult],
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[RankedResult]:
        """Rerank initial retrieval results.

        Args:
            query: User query
            initial_results: Results from HybridRetriever
            query_context: Optional context for graph reranking

        Returns:
            Top-K reranked results with scoring details
        """
        if not initial_results:
            self.logger.warning("No initial results to rerank")
            return []

        self.logger.info(f"Reranking {len(initial_results)} results")

        # Run rerankers in parallel
        tasks = [
            self.cross_encoder.rerank(query, initial_results)
        ]

        if self.graph_reranker:
            tasks.append(
                self.graph_reranker.rerank(query, initial_results, query_context)
            )
        else:
            # Placeholder: uniform graph scores
            tasks.append(self._uniform_graph_scores(initial_results))

        if self.precedence_reranker:
            tasks.append(
                self.precedence_reranker.rerank(initial_results)
            )
        else:
            # Placeholder: uniform precedence scores
            tasks.append(self._uniform_precedence_scores(initial_results))

        try:
            reranker_results = await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            raise

        # Unpack results
        ce_results = reranker_results[0]
        graph_results = reranker_results[1]
        prec_results = reranker_results[2]

        self.logger.debug(
            f"Reranker outputs: CE={len(ce_results)}, "
            f"Graph={len(graph_results)}, Precedence={len(prec_results)}"
        )

        # Fuse scores
        final_results = self.fusion.fuse(ce_results, graph_results, prec_results)

        # Filter by confidence threshold
        filtered_results = [
            r for r in final_results
            if r.confidence >= self.config.min_confidence_threshold
        ]

        self.logger.info(
            f"Reranking complete: {len(filtered_results)}/{len(final_results)} "
            f"results above threshold (min_confidence={self.config.min_confidence_threshold})"
        )

        # Log rank improvements
        for result in filtered_results[:5]:  # Top-5
            if result.rank_improvement > 0:
                self.logger.debug(
                    f"Chunk {result.chunk_id}: rank {result.original_rank} → "
                    f"{result.final_rank} (+{result.rank_improvement})"
                )

        return filtered_results

    async def _uniform_graph_scores(
        self,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float, Dict]]:
        """Placeholder when graph reranking is disabled."""
        return [(r, 0.5, {}) for r in results]

    async def _uniform_precedence_scores(
        self,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float, Dict]]:
        """Placeholder when precedence reranking is disabled."""
        return [(r, 0.5, {}) for r in results]


# ============================================================================
# Utility Functions
# ============================================================================

def create_reranking_pipeline(
    config_dict: Dict[str, Any],
    knowledge_graph: Optional['LegalKnowledgeGraph'] = None
) -> RerankingPipeline:
    """Factory function to create RerankingPipeline from config.

    Args:
        config_dict: Configuration dictionary (from YAML)
        knowledge_graph: Legal knowledge graph (optional)

    Returns:
        Initialized RerankingPipeline
    """
    config = RerankingConfig.from_yaml(config_dict)
    return RerankingPipeline(config, knowledge_graph)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Data structures
    "PrecedenceLevel",
    "RerankingScores",
    "SearchResult",
    "RankedResult",
    "RerankingConfig",
    # Rerankers
    "CrossEncoderReranker",
    "GraphAwareReranker",
    "LegalPrecedenceReranker",
    "EnsembleFusion",
    "RerankingPipeline",
    # Factory
    "create_reranking_pipeline",
]
