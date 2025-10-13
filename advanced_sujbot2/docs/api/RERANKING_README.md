# Reranking Module

## Overview

The reranking module implements a multi-stage relevance refinement pipeline for legal compliance analysis. It takes initial retrieval results (typically top-20 candidates from hybrid search) and applies sophisticated scoring to surface the most relevant chunks.

**Based on**: `specs/07_reranking.md`

## Architecture

```
Initial Results (20)
    ↓
┌──────────────────────────────┐
│ CrossEncoderReranker         │  Semantic relevance (multilingual)
│ Model: mmarco-mMiniLMv2      │  Deep query-document understanding
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ GraphAwareReranker           │  Structural features
│ - Proximity to anchors       │  Graph-based scoring
│ - Betweenness centrality     │
│ - PageRank authority         │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ LegalPrecedenceReranker      │  Legal hierarchy
│ - Constitutional > Statutory │  Czech legal system
│ - Temporal precedence        │
│ - Specificity weighting      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ EnsembleFusion               │  Score combination
│ - Weighted average           │  Multiple fusion methods
│ - Borda count                │
│ - Reciprocal rank fusion     │
└──────────────┬───────────────┘
               ↓
    Final Results (5-10)
```

## Components

### 1. CrossEncoderReranker

**Purpose**: Deep semantic relevance scoring using multilingual cross-encoder.

**Model**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Multilingual (Czech language support)
- 12-layer MiniLM architecture
- Trained on MS MARCO multilingual dataset
- Output: Relevance logit (normalized to [0, 1])

**Key Features**:
- Sees both query and document together (vs bi-encoders which encode separately)
- Includes legal context (references, hierarchy path) in input
- Batch processing for efficiency
- Score normalization with sigmoid transformation
- Confidence estimation based on score magnitude

**Usage**:
```python
config = RerankingConfig(
    cross_encoder_model="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    cross_encoder_device="cpu",
    cross_encoder_batch_size=16
)

reranker = CrossEncoderReranker(config)
results = await reranker.rerank(query="Jaké jsou povinnosti dodavatele?", results=initial_results)
```

### 2. GraphAwareReranker

**Purpose**: Leverage knowledge graph structure for scoring.

**Features**:
1. **Proximity Scoring**: Distance to query-relevant chunks (anchor chunks)
   - Uses shortest path in graph
   - Exponential decay with distance
   - Max distance configurable (default: 3 hops)

2. **Centrality Scoring**: Betweenness centrality
   - How often chunk appears on paths between other chunks
   - High centrality = foundational provision

3. **Authority Scoring**: PageRank-like algorithm
   - Chunks referenced by many others get higher scores
   - Authority flows through reference edges

**Graph Requirements**:
- NetworkX-compatible graph object
- Nodes: chunk IDs
- Edges: REFERENCES, PART_OF, etc.

**Usage**:
```python
from knowledge_graph import LegalKnowledgeGraph

graph = LegalKnowledgeGraph.load("indexes/graph.pkl")
config = RerankingConfig(
    enable_graph_reranking=True,
    graph_proximity_weight=0.4,
    graph_centrality_weight=0.3,
    graph_authority_weight=0.3
)

reranker = GraphAwareReranker(config, graph)
results = await reranker.rerank(query, initial_results, query_context={"anchor_chunks": ["chunk_123"]})
```

### 3. LegalPrecedenceReranker

**Purpose**: Apply Czech legal hierarchy and temporal precedence rules.

**Precedence Hierarchy** (highest to lowest):
1. **Constitutional** (1.0): Ústava ČR, Listina základních práv
2. **Statutory** (0.9): Zákony (e.g., Zákon č. 89/2012 Sb.)
3. **Regulatory** (0.7): Vyhlášky, nařízení vlády
4. **Contractual** (0.5): Smlouvy
5. **Guidance** (0.3): Metodiky, doporučení

**Scoring Factors**:
- **Hierarchy** (60%): Base weight from precedence level
- **Temporal** (20%): Newer laws slightly favored (lex posterior)
- **Specificity** (20%): Deeper hierarchy = more specific (lex specialis)

**Usage**:
```python
config = RerankingConfig(
    enable_precedence_weighting=True,
    precedence_weights={
        "constitutional": 1.0,
        "statutory": 0.9,
        "regulatory": 0.7,
        "contractual": 0.5,
        "guidance": 0.3
    },
    temporal_decay_factor=0.95
)

reranker = LegalPrecedenceReranker(config)
results = await reranker.rerank(initial_results)
```

### 4. EnsembleFusion

**Purpose**: Combine scores from multiple rerankers.

**Fusion Methods**:

1. **Weighted Average** (default):
   ```
   final_score = w1·cross_encoder + w2·graph + w3·precedence
   ```
   - Default weights: 0.5, 0.3, 0.2
   - Linear combination of normalized scores

2. **Borda Count** (rank aggregation):
   - Each reranker assigns ranks
   - Points = (N - rank + 1)
   - Sum points across rerankers

3. **Reciprocal Rank Fusion (RRF)**:
   ```
   score = Σ 1/(k + rank_i)  where k=60
   ```
   - Robust to score scale differences
   - No normalization needed

**Confidence Estimation**:
- High mean score + low variance = high confidence
- Agreement between rerankers = higher confidence

**Explainability**:
- Generates human-readable explanations
- Identifies dominant scoring factor
- Highlights graph features (proximity, centrality, authority)

**Usage**:
```python
config = RerankingConfig(
    ensemble_method="weighted_average",
    ensemble_weights={
        "cross_encoder": 0.5,
        "graph": 0.3,
        "precedence": 0.2
    },
    explain_reranking=True
)

fusion = EnsembleFusion(config)
final_results = fusion.fuse(ce_results, graph_results, precedence_results)
```

### 5. RerankingPipeline

**Purpose**: Orchestrate all reranking components.

**Workflow**:
1. Run all rerankers in parallel (async)
2. Fuse scores using ensemble method
3. Filter by confidence threshold
4. Return top-K results

**Usage**:
```python
from reranker import RerankingPipeline, RerankingConfig

# Initialize
config = RerankingConfig.from_yaml(config_dict)
graph = LegalKnowledgeGraph.load("indexes/graph.pkl")
pipeline = RerankingPipeline(config, graph)

# Rerank
initial_results = await hybrid_retriever.search(query, top_k=20)
ranked_results = await pipeline.rerank(query, initial_results)

# Display
for i, result in enumerate(ranked_results, 1):
    print(f"{i}. {result.legal_reference} (score: {result.scores.ensemble_score:.3f})")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Explanation: {result.reranking_explanation}")
    print(f"   Rank improvement: +{result.rank_improvement}")
```

## Data Structures

### SearchResult (Input)
```python
@dataclass
class SearchResult:
    chunk_id: str
    content: str
    legal_reference: str = ""
    document_id: str = ""
    document_type: str = "law_code"
    hierarchy_path: str = ""
    rank: int = 0
    hybrid_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### RankedResult (Output)
```python
@dataclass
class RankedResult:
    chunk_id: str
    content: str
    legal_reference: str
    document_id: str
    document_type: str

    scores: RerankingScores     # All component scores
    final_rank: int
    confidence: float
    reranking_explanation: str

    original_rank: int
    original_hybrid_score: float

    graph_neighbors: List[str]
    reference_path: Optional[List[str]]

    @property
    def rank_improvement(self) -> int:
        return self.original_rank - self.final_rank
```

### RerankingScores
```python
@dataclass
class RerankingScores:
    cross_encoder_score: float      # 0.0 to 1.0
    graph_score: float              # 0.0 to 1.0
    precedence_score: float         # 0.0 to 1.0
    ensemble_score: float           # 0.0 to 1.0

    cross_encoder_confidence: float
    graph_features: Dict[str, float]
    precedence_factors: Dict[str, float]
```

## Configuration

### Example config.yaml
```yaml
reranking:
  # Cross-encoder
  cross_encoder_model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
  cross_encoder_batch_size: 16
  cross_encoder_device: "cpu"

  # Graph-aware
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

  # Ensemble
  ensemble_method: "weighted_average"
  ensemble_weights:
    cross_encoder: 0.5
    graph: 0.3
    precedence: 0.2

  # Output
  final_top_k: 5
  min_confidence_threshold: 0.1
  explain_reranking: true
```

## Performance

### Bottlenecks

1. **Cross-Encoder Inference**: Most expensive (~1-2s for 20 pairs on CPU)
   - Solution: Use GPU (`cross_encoder_device: "cuda"`)
   - Or reduce candidates to 10-15

2. **Graph Operations**: Shortest path can be slow on large graphs
   - Solution: Precompute centrality (done at initialization)
   - Limit `max_hop_distance`

### Optimization Strategies

```python
# 1. GPU acceleration
config = RerankingConfig(cross_encoder_device="cuda")

# 2. Reduce candidates
initial_results = await retriever.search(query, top_k=15)  # Instead of 20

# 3. Disable graph reranking for simple queries
if not is_complex_query(query):
    config.enable_graph_reranking = False

# 4. Use RRF for faster fusion
config.ensemble_method = "rrf"

# 5. Increase batch size
config.cross_encoder_batch_size = 32
```

### Performance Targets

| Operation | CPU | GPU |
|-----------|-----|-----|
| Cross-encoder (20 pairs) | <1s | <0.2s |
| Graph proximity (20 nodes) | <0.1s | - |
| Full pipeline | <2s | <0.5s |

## Tuning for Different Use Cases

### High Precision (Legal/Compliance)
```yaml
reranking:
  ensemble_weights:
    cross_encoder: 0.6  # More semantic
    graph: 0.2
    precedence: 0.2
  final_top_k: 3        # Only top-3
  min_confidence_threshold: 0.5
```

### High Recall (Research)
```yaml
reranking:
  ensemble_weights:
    cross_encoder: 0.4
    graph: 0.4          # More graph structure
    precedence: 0.2
  final_top_k: 10       # More results
  min_confidence_threshold: 0.1
```

### Speed Optimized
```yaml
reranking:
  enable_graph_reranking: false
  cross_encoder_batch_size: 32
  ensemble_method: "rrf"
  final_top_k: 5
```

## Debugging

### Enable Verbose Logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("reranker")
logger.setLevel(logging.DEBUG)
```

### Inspect Scores
```python
for result in ranked_results:
    print(f"Chunk: {result.chunk_id}")
    print(f"  Cross-encoder: {result.scores.cross_encoder_score:.3f}")
    print(f"  Graph: {result.scores.graph_score:.3f}")
    print(f"  Precedence: {result.scores.precedence_score:.3f}")
    print(f"  Ensemble: {result.scores.ensemble_score:.3f}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Explanation: {result.reranking_explanation}")
    print()
```

### Compare Fusion Methods
```python
methods = ["weighted_average", "borda_count", "rrf"]
for method in methods:
    config.ensemble_method = method
    results = await pipeline.rerank(query, initial_results)
    print(f"{method}: Top result = {results[0].chunk_id}, score = {results[0].scores.ensemble_score:.3f}")
```

## Dependencies

```bash
pip install sentence-transformers  # For CrossEncoder
pip install networkx              # For graph operations
pip install numpy                 # For score normalization
```

## Future Enhancements

1. **Learned-to-Rank**: Train gradient-boosted model on all features
2. **Fine-tuned Cross-Encoder**: Domain-specific Czech legal dataset
3. **Multi-Document Awareness**: Boost contract-law alignment
4. **Explainable AI**: SHAP values for score attribution

## References

- Specification: `specs/07_reranking.md`
- Cross-Encoder Paper: [MS MARCO](https://arxiv.org/abs/2004.08476)
- RRF Paper: [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- Configuration: `config.example.yaml`
