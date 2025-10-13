# 07. Reranking Specification

## 1. Purpose

**Objective**: Refine initial retrieval results using sophisticated relevance scoring that considers semantic similarity, graph structure, and legal precedence rules to surface the most relevant chunks for compliance analysis.

**Why Reranking?**
- Initial hybrid retrieval casts a wide net (top-20 candidates)
- Cross-encoders provide deeper semantic understanding than bi-encoders
- Graph-aware scoring leverages structural relationships between legal provisions
- Legal precedence rules ensure higher-authority provisions are prioritized
- Ensemble methods combine multiple signals for robust ranking

**Key Capabilities**:
1. **Multilingual Cross-Encoder** - Deep semantic relevance for Czech language
2. **Graph-Aware Scoring** - Boost chunks based on graph proximity and centrality
3. **Legal Precedence Weighting** - Prioritize constitutional law > statutes > regulations
4. **Ensemble Reranking** - Combine multiple rerankers for robust results
5. **Confidence Calibration** - Normalized scores with uncertainty estimates

---

## 2. Reranking Architecture

### High-Level Flow

```
┌─────────────────────────────────────┐
│  Hybrid Retrieval Results           │
│  (20 candidates)                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Cross-Encoder Reranker             │
│  - Query-document pairs             │
│  - Multilingual mMiniLMv2           │
│  → Semantic relevance scores        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Graph-Aware Reranker               │
│  - Reference proximity              │
│  - Betweenness centrality           │
│  - Authority propagation            │
│  → Graph structure scores           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Legal Precedence Reranker          │
│  - Document hierarchy               │
│  - Authority weighting              │
│  - Temporal precedence              │
│  → Legal authority scores           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Ensemble Fusion                    │
│  - Weighted score combination       │
│  - Rank aggregation                 │
│  - Confidence calibration           │
│  → Final unified scores             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Top-K Selection                    │
│  (5-10 final results)               │
└─────────────────────────────────────┘
```

### Component Interaction

```
HybridRetriever → [SearchResult] → RerankingPipeline
                                    │
                                    ├→ CrossEncoderReranker
                                    ├→ GraphAwareReranker
                                    ├→ LegalPrecedenceReranker
                                    │
                                    └→ EnsembleFusion → [RankedResult]
```

---

## 3. Data Structures

### 3.1 RerankingResult

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

@dataclass
class RerankingScores:
    """Individual scores from different rerankers."""
    cross_encoder_score: float  # -15 to 15 (logit)
    graph_score: float          # 0.0 to 1.0
    precedence_score: float     # 0.0 to 1.0
    ensemble_score: float       # 0.0 to 1.0 (final)

    # Score metadata
    cross_encoder_confidence: float  # 0.0 to 1.0
    graph_features: Dict[str, float]  # proximity, centrality, etc.
    precedence_factors: Dict[str, float]  # hierarchy level, temporal, etc.

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
    graph_neighbors: List[str]  # Chunk IDs of related chunks
    reference_path: Optional[List[str]]  # Path to query-relevant chunk

    @property
    def rank_improvement(self) -> int:
        """How many positions this result moved up."""
        return self.original_rank - self.final_rank

class PrecedenceLevel(Enum):
    """Legal hierarchy levels."""
    CONSTITUTIONAL = 5  # Ústava, Listina
    STATUTORY = 4       # Zákony
    REGULATORY = 3      # Vyhlášky, nařízení
    CONTRACTUAL = 2     # Smlouvy
    GUIDANCE = 1        # Metodiky, doporučení
```

### 3.2 Configuration

```python
@dataclass
class RerankingConfig:
    """Configuration for reranking pipeline."""

    # Cross-encoder settings
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cross_encoder_batch_size: int = 16
    cross_encoder_device: str = "cpu"  # cpu | cuda | mps

    # Graph-aware settings
    enable_graph_reranking: bool = True
    graph_proximity_weight: float = 0.4
    graph_centrality_weight: float = 0.3
    graph_authority_weight: float = 0.3
    max_hop_distance: int = 3  # For proximity calculation

    # Legal precedence settings
    enable_precedence_weighting: bool = True
    precedence_weights: Dict[PrecedenceLevel, float] = None  # Default in __post_init__
    temporal_decay_factor: float = 0.95  # Newer laws slightly favored

    # Ensemble settings
    ensemble_method: str = "weighted_average"  # weighted_average | borda_count | rrf
    ensemble_weights: Dict[str, float] = None  # cross_encoder, graph, precedence

    # Output settings
    final_top_k: int = 5
    min_confidence_threshold: float = 0.1
    explain_reranking: bool = True

    def __post_init__(self):
        if self.precedence_weights is None:
            self.precedence_weights = {
                PrecedenceLevel.CONSTITUTIONAL: 1.0,
                PrecedenceLevel.STATUTORY: 0.9,
                PrecedenceLevel.REGULATORY: 0.7,
                PrecedenceLevel.CONTRACTUAL: 0.5,
                PrecedenceLevel.GUIDANCE: 0.3
            }

        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "cross_encoder": 0.5,
                "graph": 0.3,
                "precedence": 0.2
            }
```

---

## 4. Cross-Encoder Reranker

### 4.1 Architecture

**Model**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Multilingual (Czech support)
- 12-layer MiniLM architecture
- 384 hidden dimensions
- Trained on MS MARCO multilingual dataset
- Output: Single relevance logit (typically -15 to +15)

**Key Difference from Bi-Encoders**:
- **Bi-encoder** (BGE-M3): Encodes query and document separately → fast but less accurate
- **Cross-encoder**: Sees both query and document together → slower but much more accurate

### 4.2 Implementation

```python
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Tuple

class CrossEncoderReranker:
    """Rerank results using multilingual cross-encoder."""

    def __init__(self, config: RerankingConfig):
        self.config = config
        self.model = CrossEncoder(
            config.cross_encoder_model,
            device=config.cross_encoder_device,
            max_length=512  # Token limit for input
        )

        # Score normalization params (calibrated on validation set)
        self.score_mean = 0.0
        self.score_std = 5.0

    async def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float]]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: User query
            results: Initial search results

        Returns:
            List of (result, cross_encoder_score) sorted by score
        """
        if not results:
            return []

        # Prepare query-document pairs
        pairs = []
        for result in results:
            # Construct input with legal context
            document_text = self._prepare_document_text(result)
            pairs.append([query, document_text])

        # Batch prediction
        scores = await asyncio.to_thread(
            self.model.predict,
            pairs,
            batch_size=self.config.cross_encoder_batch_size,
            show_progress_bar=False
        )

        # Normalize scores to [0, 1]
        normalized_scores = self._normalize_scores(scores)

        # Sort by score
        scored_results = list(zip(results, normalized_scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results

    def _prepare_document_text(self, result: SearchResult) -> str:
        """
        Prepare document text for cross-encoder input.
        Include legal context for better scoring.
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
        text = " ".join(parts)
        return text[:2000]  # Roughly 500 tokens

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize cross-encoder logits to [0, 1] range.
        Uses sigmoid-like transformation with calibrated params.
        """
        # Z-score normalization
        z_scores = (scores - self.score_mean) / self.score_std

        # Sigmoid transformation
        normalized = 1 / (1 + np.exp(-z_scores))

        return normalized

    def get_confidence(self, score: float) -> float:
        """
        Estimate confidence in cross-encoder score.
        Higher absolute scores = higher confidence.
        """
        # Confidence based on distance from decision boundary (0)
        raw_score = score * self.score_std + self.score_mean
        confidence = min(1.0, abs(raw_score) / 10.0)  # Max at |score| = 10
        return confidence
```

---

## 5. Graph-Aware Reranker

### 5.1 Graph Features

**Proximity Score**: How close is the chunk to query-relevant chunks in the graph?
- Uses shortest path distance
- Considers reference edges (REFERENCES, PART_OF)

**Centrality Score**: How important is this chunk in the overall legal structure?
- Betweenness centrality: chunk is on paths between many other chunks
- High centrality = foundational provision

**Authority Propagation**: Score flows from high-authority chunks to neighbors
- Similar to PageRank
- Chunks referenced by many others get higher scores

### 5.2 Implementation

```python
from typing import Dict, List, Set
import networkx as nx

class GraphAwareReranker:
    """Rerank using knowledge graph structure."""

    def __init__(
        self,
        config: RerankingConfig,
        knowledge_graph: 'LegalKnowledgeGraph'
    ):
        self.config = config
        self.graph = knowledge_graph

        # Precompute centrality measures
        self._centrality_cache = self._compute_centrality()
        self._authority_cache = self._compute_authority()

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[SearchResult, float]]:
        """
        Rerank based on graph structure.

        Args:
            query: User query (may contain legal references)
            results: Initial search results
            query_context: Optional context (e.g., anchor chunks)

        Returns:
            List of (result, graph_score) sorted by score
        """
        # Identify anchor chunks (highly relevant to query)
        anchor_chunks = self._identify_anchors(query, results, query_context)

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

            scored_results.append((
                result,
                graph_score,
                {
                    "proximity": proximity_score,
                    "centrality": centrality_score,
                    "authority": authority_score
                }
            ))

        # Sort by graph score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results

    def _identify_anchors(
        self,
        query: str,
        results: List[SearchResult],
        query_context: Optional[Dict[str, Any]]
    ) -> Set[str]:
        """
        Identify anchor chunks (reference points for proximity).
        """
        anchors = set()

        # Extract legal references from query
        references = self._extract_references(query)
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
        """
        Compute proximity score based on graph distance to anchors.
        """
        if not anchor_chunks or chunk_id in anchor_chunks:
            return 1.0  # Maximum proximity

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
            except nx.NetworkXNoPath:
                continue

        # Convert distance to score (closer = higher score)
        if min_distance == float('inf'):
            return 0.0

        # Exponential decay with distance
        max_hops = self.config.max_hop_distance
        if min_distance > max_hops:
            return 0.0

        proximity_score = np.exp(-0.5 * min_distance)
        return proximity_score

    def _compute_centrality(self) -> Dict[str, float]:
        """
        Precompute betweenness centrality for all chunks.
        """
        centrality = nx.betweenness_centrality(self.graph.graph)

        # Normalize to [0, 1]
        max_centrality = max(centrality.values()) if centrality else 1.0
        normalized = {
            node: score / max_centrality
            for node, score in centrality.items()
        }

        return normalized

    def _compute_authority(self) -> Dict[str, float]:
        """
        Compute authority scores using PageRank-like algorithm.
        """
        # Filter to reference edges only
        reference_graph = self.graph.get_subgraph_by_edge_type('REFERENCES')

        # Run PageRank
        try:
            authority = nx.pagerank(reference_graph, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            # Fallback: uniform scores
            authority = {node: 1.0 for node in reference_graph.nodes()}

        # Normalize
        max_authority = max(authority.values()) if authority else 1.0
        normalized = {
            node: score / max_authority
            for node, score in authority.items()
        }

        return normalized

    def _extract_references(self, query: str) -> List[str]:
        """Extract legal references from query."""
        # Reuse regex patterns from DocumentReader
        from src.document_reader import ReferenceExtractor
        extractor = ReferenceExtractor()
        return extractor.extract(query)
```

---

## 6. Legal Precedence Reranker

### 6.1 Precedence Rules

**Hierarchy Levels** (highest to lowest):
1. **Constitutional**: Ústava ČR, Listina základních práv
2. **Statutory**: Zákony (e.g., Zákon č. 89/2012 Sb.)
3. **Regulatory**: Vyhlášky, nařízení vlády
4. **Contractual**: Smlouvy
5. **Guidance**: Metodiky, doporučení

**Temporal Precedence**:
- Newer laws override older laws (lex posterior)
- Specific laws override general laws (lex specialis)

### 6.2 Implementation

```python
class LegalPrecedenceReranker:
    """Rerank based on legal authority hierarchy."""

    def __init__(self, config: RerankingConfig):
        self.config = config

        # Document type → precedence level mapping
        self.type_to_level = {
            "constitution": PrecedenceLevel.CONSTITUTIONAL,
            "law_code": PrecedenceLevel.STATUTORY,
            "regulation": PrecedenceLevel.REGULATORY,
            "contract": PrecedenceLevel.CONTRACTUAL,
            "guidance": PrecedenceLevel.GUIDANCE
        }

    async def rerank(
        self,
        results: List[SearchResult]
    ) -> List[Tuple[SearchResult, float]]:
        """
        Rerank based on legal precedence.

        Returns:
            List of (result, precedence_score)
        """
        scored_results = []

        for result in results:
            # Base score from hierarchy
            precedence_level = self._get_precedence_level(result)
            hierarchy_score = self.config.precedence_weights[precedence_level]

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

            scored_results.append((
                result,
                precedence_score,
                {
                    "hierarchy": hierarchy_score,
                    "temporal": temporal_score,
                    "specificity": specificity_score
                }
            ))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results

    def _get_precedence_level(self, result: SearchResult) -> PrecedenceLevel:
        """Determine precedence level from document type."""
        doc_type = result.document_type
        return self.type_to_level.get(doc_type, PrecedenceLevel.GUIDANCE)

    def _compute_temporal_score(self, result: SearchResult) -> float:
        """
        Score based on document age.
        Newer documents get slightly higher scores.
        """
        if not result.metadata.get("effective_date"):
            return 0.5  # Neutral if date unknown

        effective_date = result.metadata["effective_date"]
        years_old = (datetime.now() - effective_date).days / 365.25

        # Exponential decay (very slow)
        temporal_score = self.config.temporal_decay_factor ** years_old
        return temporal_score

    def _compute_specificity(self, result: SearchResult) -> float:
        """
        More specific provisions (deeper hierarchy) score higher.
        """
        hierarchy_path = result.hierarchy_path or ""

        # Count hierarchy depth
        depth = hierarchy_path.count(">") + 1 if hierarchy_path else 0

        # Normalize to [0, 1]
        # Typical depths: 1 (Part) to 5 (Letter)
        specificity_score = min(1.0, depth / 5.0)

        return specificity_score
```

---

## 7. Ensemble Fusion

### 7.1 Fusion Methods

**Weighted Average**:
```
final_score = w1·cross_encoder + w2·graph + w3·precedence
```

**Borda Count** (Rank Aggregation):
- Each reranker assigns ranks (1st, 2nd, 3rd, ...)
- Points = (N - rank + 1)
- Sum points across rerankers

**Reciprocal Rank Fusion (RRF)**:
```
score(chunk) = Σ 1 / (k + rank_i)
```
- k = 60 (constant)
- rank_i = rank from reranker i

### 7.2 Implementation

```python
class EnsembleFusion:
    """Combine scores from multiple rerankers."""

    def __init__(self, config: RerankingConfig):
        self.config = config

    def fuse(
        self,
        cross_encoder_results: List[Tuple[SearchResult, float]],
        graph_results: List[Tuple[SearchResult, float, Dict]],
        precedence_results: List[Tuple[SearchResult, float, Dict]]
    ) -> List[RankedResult]:
        """
        Fuse scores from all rerankers.

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
            raise ValueError(f"Unknown ensemble method: {method}")

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

            # Build RankedResult
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

    def _borda_count_fusion(self, *args) -> List[RankedResult]:
        """Rank aggregation using Borda count."""
        # Convert all result lists to rankings
        all_rankings = []
        for results in args:
            ranking = {r.chunk_id: rank for rank, (r, _) in enumerate(results, 1)}
            all_rankings.append(ranking)

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
            borda_scores[chunk_id] = points

        # Build RankedResults (similar to weighted_average)
        # ... (code similar to above, using borda_scores as ensemble_score)
        pass  # Implementation analogous to weighted_average

    def _rrf_fusion(self, *args) -> List[RankedResult]:
        """Reciprocal rank fusion."""
        k = 60  # RRF constant

        # Convert to rankings
        all_rankings = []
        for results in args:
            ranking = {r.chunk_id: rank for rank, (r, _) in enumerate(results, 1)}
            all_rankings.append(ranking)

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

        # Build RankedResults
        # ... (similar to weighted_average)
        pass

    def _compute_confidence(self, scores: RerankingScores) -> float:
        """
        Compute overall confidence in ranking.
        Based on agreement between rerankers.
        """
        # If all scores are high, confidence is high
        score_values = [
            scores.cross_encoder_score,
            scores.graph_score,
            scores.precedence_score
        ]

        mean_score = np.mean(score_values)
        std_score = np.std(score_values)

        # High mean and low std = high confidence
        confidence = mean_score * (1 - std_score / (mean_score + 1e-6))
        confidence = np.clip(confidence, 0.0, 1.0)

        return confidence

    def _explain_reranking(
        self,
        result: SearchResult,
        scores: RerankingScores
    ) -> str:
        """Generate human-readable explanation of reranking."""
        if not self.config.explain_reranking:
            return ""

        parts = []

        # Dominant score
        max_component = max([
            ("semantic relevance", scores.cross_encoder_score),
            ("graph structure", scores.graph_score),
            ("legal precedence", scores.precedence_score)
        ], key=lambda x: x[1])

        parts.append(f"Ranked highly due to {max_component[0]} (score: {max_component[1]:.2f})")

        # Graph features
        if scores.graph_features:
            if scores.graph_features.get("proximity", 0) > 0.7:
                parts.append("close proximity to query-relevant provisions")
            if scores.graph_features.get("centrality", 0) > 0.7:
                parts.append("foundational provision in legal structure")

        # Precedence
        if scores.precedence_score > 0.8:
            parts.append("high legal authority")

        return "; ".join(parts)

    def _get_ce_confidence(self, score: float) -> float:
        """Confidence in cross-encoder score."""
        # Higher absolute score = higher confidence
        # Assume scores normalized to [0, 1]
        return abs(2 * score - 1)  # 0.5 → 0 confidence, 0 or 1 → 1 confidence
```

---

## 8. Reranking Pipeline

### 8.1 Orchestration

```python
class RerankingPipeline:
    """Orchestrate all reranking components."""

    def __init__(
        self,
        config: RerankingConfig,
        knowledge_graph: 'LegalKnowledgeGraph'
    ):
        self.config = config

        # Initialize rerankers
        self.cross_encoder = CrossEncoderReranker(config)

        if config.enable_graph_reranking:
            self.graph_reranker = GraphAwareReranker(config, knowledge_graph)
        else:
            self.graph_reranker = None

        if config.enable_precedence_weighting:
            self.precedence_reranker = LegalPrecedenceReranker(config)
        else:
            self.precedence_reranker = None

        self.fusion = EnsembleFusion(config)

        self.logger = logging.getLogger(__name__)

    async def rerank(
        self,
        query: str,
        initial_results: List[SearchResult],
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[RankedResult]:
        """
        Rerank initial retrieval results.

        Args:
            query: User query
            initial_results: Results from HybridRetriever
            query_context: Optional context for graph reranking

        Returns:
            Top-K reranked results
        """
        if not initial_results:
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

        if self.precedence_reranker:
            tasks.append(
                self.precedence_reranker.rerank(initial_results)
            )

        reranker_results = await asyncio.gather(*tasks)

        # Unpack results
        ce_results = reranker_results[0]
        graph_results = reranker_results[1] if self.graph_reranker else []
        prec_results = reranker_results[2] if self.precedence_reranker else []

        # Fuse scores
        final_results = self.fusion.fuse(ce_results, graph_results, prec_results)

        # Filter by confidence threshold
        final_results = [
            r for r in final_results
            if r.confidence >= self.config.min_confidence_threshold
        ]

        self.logger.info(f"Reranking complete: {len(final_results)} results above threshold")

        return final_results
```

---

## 9. Configuration

### config.yaml

```yaml
reranking:
  # Cross-encoder
  cross_encoder_model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
  cross_encoder_batch_size: 16
  cross_encoder_device: "cpu"  # cpu | cuda | mps

  # Graph-aware reranking
  enable_graph_reranking: true
  graph_proximity_weight: 0.4
  graph_centrality_weight: 0.3
  graph_authority_weight: 0.3
  max_hop_distance: 3

  # Legal precedence
  enable_precedence_weighting: true
  temporal_decay_factor: 0.95
  precedence_weights:
    constitutional: 1.0
    statutory: 0.9
    regulatory: 0.7
    contractual: 0.5
    guidance: 0.3

  # Ensemble fusion
  ensemble_method: "weighted_average"  # weighted_average | borda_count | rrf
  ensemble_weights:
    cross_encoder: 0.5
    graph: 0.3
    precedence: 0.2

  # Output
  final_top_k: 5
  min_confidence_threshold: 0.1
  explain_reranking: true
```

---

## 10. Usage Examples

### 10.1 Basic Reranking

```python
from src.reranking_pipeline import RerankingPipeline, RerankingConfig
from src.knowledge_graph import LegalKnowledgeGraph

# Initialize
config = RerankingConfig.from_yaml("config.yaml")
graph = LegalKnowledgeGraph.load("indexes/graph.pkl")
pipeline = RerankingPipeline(config, graph)

# Get initial results from HybridRetriever
initial_results = await hybrid_retriever.search(
    query="Jaké jsou povinnosti dodavatele podle §89?",
    top_k=20
)

# Rerank
ranked_results = await pipeline.rerank(
    query="Jaké jsou povinnosti dodavatele podle §89?",
    initial_results=initial_results
)

# Display
for i, result in enumerate(ranked_results, 1):
    print(f"{i}. {result.legal_reference} (score: {result.scores.ensemble_score:.3f})")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Explanation: {result.reranking_explanation}")
    print(f"   Rank improvement: +{result.rank_improvement}")
    print()
```

### 10.2 With Query Context

```python
# Provide anchor chunks for graph reranking
query_context = {
    "anchor_chunks": ["chunk_law_123", "chunk_law_456"],
    "focus_document": "law_89_2012"
}

ranked_results = await pipeline.rerank(
    query="Najdi konflikty ve smlouvě",
    initial_results=initial_results,
    query_context=query_context
)
```

### 10.3 Custom Weights

```python
# Boost graph reranking for cross-document queries
config = RerankingConfig(
    ensemble_weights={
        "cross_encoder": 0.4,
        "graph": 0.4,
        "precedence": 0.2
    }
)

pipeline = RerankingPipeline(config, graph)
```

---

## 11. Testing

### 11.1 Unit Tests

```python
import pytest
from src.reranking_pipeline import CrossEncoderReranker, RerankingConfig

@pytest.mark.asyncio
async def test_cross_encoder_reranking():
    """Test cross-encoder reranking."""
    config = RerankingConfig()
    reranker = CrossEncoderReranker(config)

    # Mock results
    results = [
        SearchResult(chunk_id="1", content="Relevant text", rank=1),
        SearchResult(chunk_id="2", content="Irrelevant text", rank=2),
        SearchResult(chunk_id="3", content="Somewhat relevant", rank=3)
    ]

    reranked = await reranker.rerank("test query", results)

    # Check that order may change
    assert len(reranked) == 3
    assert all(isinstance(score, float) for _, score in reranked)
    assert all(0 <= score <= 1 for _, score in reranked)

@pytest.mark.asyncio
async def test_graph_aware_proximity():
    """Test graph proximity scoring."""
    # Build mock graph
    graph = LegalKnowledgeGraph()
    graph.add_node("chunk_1", "§89")
    graph.add_node("chunk_2", "§90")
    graph.add_node("chunk_3", "§91")
    graph.add_edge("chunk_1", "chunk_2", "REFERENCES")
    graph.add_edge("chunk_2", "chunk_3", "REFERENCES")

    config = RerankingConfig()
    reranker = GraphAwareReranker(config, graph)

    # chunk_1 is anchor
    anchor_chunks = {"chunk_1"}

    # Proximity: chunk_1 > chunk_2 > chunk_3
    prox_1 = reranker._compute_proximity("chunk_1", anchor_chunks)
    prox_2 = reranker._compute_proximity("chunk_2", anchor_chunks)
    prox_3 = reranker._compute_proximity("chunk_3", anchor_chunks)

    assert prox_1 > prox_2 > prox_3
    assert prox_1 == 1.0  # Anchor itself

def test_legal_precedence():
    """Test precedence scoring."""
    config = RerankingConfig()
    reranker = LegalPrecedenceReranker(config)

    # Statutory > Contract
    law_result = SearchResult(document_type="law_code", chunk_id="1")
    contract_result = SearchResult(document_type="contract", chunk_id="2")

    law_score = reranker._get_precedence_level(law_result)
    contract_score = reranker._get_precedence_level(contract_result)

    assert law_score > contract_score
```

### 11.2 Integration Tests

```python
@pytest.mark.asyncio
async def test_full_reranking_pipeline():
    """Test complete reranking pipeline."""
    config = RerankingConfig.from_yaml("config.yaml")
    graph = LegalKnowledgeGraph.load("test_graph.pkl")
    pipeline = RerankingPipeline(config, graph)

    # Mock initial results (in suboptimal order)
    initial_results = [
        SearchResult(chunk_id="low_quality", content="...", rank=1),
        SearchResult(chunk_id="high_quality", content="...", rank=2),
        SearchResult(chunk_id="medium_quality", content="...", rank=3)
    ]

    # Rerank
    ranked = await pipeline.rerank("test query", initial_results)

    # Check that high_quality moved up
    assert ranked[0].chunk_id == "high_quality"
    assert ranked[0].rank_improvement > 0

    # Check confidence scores
    assert all(0 <= r.confidence <= 1 for r in ranked)
```

### 11.3 Benchmark Tests

```python
def test_reranking_performance():
    """Test reranking speed."""
    import time

    config = RerankingConfig()
    pipeline = RerankingPipeline(config, graph)

    # 20 candidates
    results = [SearchResult(...) for _ in range(20)]

    start = time.time()
    ranked = await pipeline.rerank("query", results)
    elapsed = time.time() - start

    # Should complete in <2 seconds on CPU
    assert elapsed < 2.0

    print(f"Reranking 20→5 took {elapsed:.2f}s")
```

---

## 12. Performance Considerations

### 12.1 Bottlenecks

**Cross-Encoder Inference**:
- Most expensive operation (~1-2s for 20 pairs on CPU)
- Solution: Use GPU if available, or reduce candidates to 10-15

**Graph Operations**:
- Shortest path computations can be slow on large graphs
- Solution: Precompute centrality; limit max_hop_distance

**Parallel Reranking**:
- Run all rerankers concurrently with `asyncio.gather()`

### 12.2 Optimization Strategies

```python
# 1. GPU acceleration for cross-encoder
config = RerankingConfig(cross_encoder_device="cuda")

# 2. Reduce candidates before reranking
initial_results = await hybrid_retriever.search(query, top_k=15)  # Instead of 20

# 3. Disable graph reranking for simple queries
if not is_complex_query(query):
    config.enable_graph_reranking = False

# 4. Use RRF for faster fusion (no score normalization needed)
config.ensemble_method = "rrf"

# 5. Batch cross-encoder inference
config.cross_encoder_batch_size = 32  # Process all at once
```

### 12.3 Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Cross-encoder (20 pairs) | <1s | CPU, batch_size=16 |
| Cross-encoder (20 pairs) | <0.2s | GPU |
| Graph proximity (20 nodes) | <0.1s | With cached centrality |
| Full reranking pipeline | <2s | CPU, all components |
| Full reranking pipeline | <0.5s | GPU |

---

## 13. Future Enhancements

### 13.1 Advanced Techniques

**Learned-to-Rank (LTR)**:
- Train a gradient-boosted model (LightGBM, XGBoost) on features:
  - Cross-encoder score
  - Graph features (proximity, centrality, authority)
  - Precedence scores
  - BM25 score, semantic score
  - Query-document length ratio
- Optimize for ranking metrics (NDCG, MAP)

**Neural Reranking**:
- Fine-tune cross-encoder on legal domain data
- Training data: (query, relevant_chunk) pairs from expert annotations
- Czech legal question-answering dataset

**Explainable Reranking**:
- Highlight which features most influenced ranking
- Generate natural language explanations
- Attribution methods (SHAP, LIME)

### 13.2 Domain-Specific Improvements

**Legal Citation Analysis**:
- Boost chunks that cite or are cited by query-relevant provisions
- Track citation strength (mandatory vs. optional references)

**Clause-Level Compliance Scoring**:
- Train reranker specifically for compliance detection
- Features: obligation/prohibition language, specificity, temporal constraints

**Multi-Document Awareness**:
- Boost contract chunks that reference laws (explicit compliance)
- Penalize contract chunks with no legal grounding (potential gaps)

---

## 14. Debugging and Monitoring

### 14.1 Logging

```python
import logging

logger = logging.getLogger("reranking")
logger.setLevel(logging.DEBUG)

# Log scores from each reranker
logger.debug(f"Cross-encoder scores: {ce_scores}")
logger.debug(f"Graph scores: {graph_scores}")
logger.debug(f"Precedence scores: {prec_scores}")
logger.debug(f"Ensemble scores: {ensemble_scores}")

# Log rank changes
for result in ranked_results:
    if result.rank_improvement > 0:
        logger.info(f"Chunk {result.chunk_id} moved up {result.rank_improvement} positions")
```

### 14.2 Metrics Tracking

```python
from dataclasses import dataclass

@dataclass
class RerankingMetrics:
    """Metrics for reranking quality."""
    avg_rank_improvement: float
    cross_encoder_time: float
    graph_reranking_time: float
    total_time: float

    # Ranking correlation
    kendall_tau: float  # Correlation with original ranking

    # Confidence distribution
    avg_confidence: float
    low_confidence_count: int  # < 0.3

def compute_metrics(ranked_results: List[RankedResult]) -> RerankingMetrics:
    """Compute metrics for monitoring."""
    # Implementation
    pass
```

### 14.3 A/B Testing

```python
# Compare ensemble methods
ensemble_methods = ["weighted_average", "borda_count", "rrf"]

for method in ensemble_methods:
    config.ensemble_method = method
    results = await pipeline.rerank(query, initial_results)

    # Evaluate with expert judgments
    ndcg = evaluate_ndcg(results, expert_relevance_labels)
    print(f"{method}: NDCG@5 = {ndcg:.3f}")
```

---

## 15. Summary

**Key Takeaways**:

1. **Multi-Stage Reranking**: Combine cross-encoder (semantic), graph structure, and legal precedence for robust relevance scoring

2. **Cross-Encoder Power**: Multilingual cross-encoder significantly improves relevance, especially for complex queries

3. **Graph-Aware Boost**: Proximity to query-relevant chunks and centrality in legal structure enhance ranking

4. **Legal Domain Knowledge**: Precedence weighting ensures high-authority provisions are prioritized

5. **Ensemble Robustness**: Fusing multiple rerankers reduces variance and improves overall quality

6. **Performance Trade-offs**: Cross-encoder is expensive; use GPU or reduce candidates for speed

7. **Explainability**: Generate human-readable explanations for why chunks were ranked highly

**Integration with Pipeline**:
```
HybridRetriever (top-20) → RerankingPipeline (top-5) → ComplianceAnalyzer
```

**Next Steps**:
- See [08. Query Processing](08_query_processing.md) for question decomposition
- See [09. Compliance Analyzer](09_compliance_analyzer.md) for using reranked results
- See [10. Knowledge Graph](10_knowledge_graph.md) for graph construction

---

**Page Count**: ~16 pages
**Last Updated**: 2025-10-08
**Status**: Complete ✅
