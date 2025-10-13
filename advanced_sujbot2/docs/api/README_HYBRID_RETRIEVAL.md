# Hybrid Retrieval System

## Overview

The Triple Hybrid Retrieval System combines three complementary search strategies to achieve optimal precision and recall for legal document retrieval:

1. **Semantic Search** (50% weight) - Using BGE-M3 embeddings and FAISS vector search
2. **Keyword Search** (30% weight) - Using BM25Okapi algorithm
3. **Structural Search** (20% weight) - Using legal hierarchy and reference matching

## Architecture

```
Query → [Semantic + Keyword + Structural] → Score Fusion → Deduplication → Top-K Results
```

### Components

#### 1. SemanticSearcher
- Uses FAISS vector indices for similarity search
- Generates contextualized embeddings with legal hierarchy
- Supports multi-document retrieval

#### 2. KeywordSearcher
- Implements BM25Okapi algorithm for keyword matching
- Tokenizes with Czech stop word removal
- Preserves legal references (§89, Článek 5)

#### 3. StructuralSearcher
- Extracts legal references from queries
- Filters by hierarchy (Part, Chapter)
- Matches content types (obligations, prohibitions, definitions)

#### 4. HybridRetriever
- Combines all three searchers
- Adaptive weighting based on query characteristics
- Score normalization and fusion
- Deduplication by chunk_id

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import asyncio
from src.hybrid_retriever import (
    HybridRetriever,
    SemanticSearcher,
    KeywordSearcher,
    StructuralSearcher,
    RetrievalConfig,
    MultiDocumentVectorStore,
    LegalEmbedder,
    create_hybrid_retriever
)

# Initialize components
vector_store = MultiDocumentVectorStore()
embedder = LegalEmbedder(model_name="BAAI/bge-m3")

# Create retriever with default config
retriever = create_hybrid_retriever(vector_store, embedder)

# Or with custom config
config = RetrievalConfig(
    semantic_weight=0.5,
    keyword_weight=0.3,
    structural_weight=0.2,
    top_k=20,
    adaptive_weights=True
)
retriever = create_hybrid_retriever(vector_store, embedder, config)

# Search
async def search_example():
    results = await retriever.search(
        query="Jaké jsou povinnosti dodavatele podle §89?",
        top_k=10
    )

    for rank, result in enumerate(results, 1):
        print(f"{rank}. [{result.score:.3f}] {result.chunk.legal_reference}")
        print(f"   Semantic: {result.semantic_score:.3f}")
        print(f"   Keyword: {result.keyword_score:.3f}")
        print(f"   Structural: {result.structural_score:.3f}")
        print(f"   Content: {result.chunk.content[:100]}...")

asyncio.run(search_example())
```

## Configuration

Edit `config.yaml` to customize retrieval behavior:

### Default Weights (Balanced)
```yaml
retrieval:
  semantic_weight: 0.5   # Semantic understanding
  keyword_weight: 0.3    # Exact term matching
  structural_weight: 0.2  # Legal structure
```

### High Precision (Legal/Compliance)
```yaml
retrieval:
  semantic_weight: 0.6
  keyword_weight: 0.25
  structural_weight: 0.15
  top_k: 10
  min_score_threshold: 0.3
```

### High Recall (Research/Exploration)
```yaml
retrieval:
  semantic_weight: 0.4
  keyword_weight: 0.4
  structural_weight: 0.2
  top_k: 30
  min_score_threshold: 0.05
  enable_query_expansion: true
```

## Features

### Adaptive Weighting

The system automatically adjusts weights based on query characteristics:

- **Legal references detected** (§89, Článek 5) → Boost structural weight
- **Short queries** (≤3 words) → Boost keyword weight
- **Long analytical queries** (>15 words) → Boost semantic weight

Example:
```python
# Query: "podle §89"
# Weights: semantic=0.4, keyword=0.2, structural=0.4 (boosted)

# Query: "odpovědnost dodavatele"
# Weights: semantic=0.5, keyword=0.4, structural=0.1 (short query)
```

### Query Expansion

Optional synonym-based query expansion:

```python
config = RetrievalConfig(enable_query_expansion=True, max_expansions=3)
retriever = create_hybrid_retriever(vector_store, embedder, config)

# Query: "odpovědnost dodavatele"
# Expanded: ["odpovědnost dodavatele", "liability dodavatele", "ručení dodavatele"]
```

### Caching

Enable caching for repeated queries:

```python
from src.hybrid_retriever import CachedHybridRetriever

cached_retriever = CachedHybridRetriever(
    semantic_searcher,
    keyword_searcher,
    structural_searcher,
    config,
    cache_size=1000
)
```

## Score Breakdown

Each result includes detailed score information:

```python
result = results[0]
breakdown = result.get_score_breakdown()

# {
#     'semantic': 0.85,    # Embedding similarity
#     'keyword': 0.62,     # BM25 score
#     'structural': 0.90,  # Legal reference match
#     'combined': 0.79     # Weighted fusion
# }
```

## Metadata Filtering

Filter results by document metadata:

```python
results = await retriever.search(
    query="povinnosti dodavatele",
    filters={
        'content_type': 'obligation',  # Only obligations
        'part': 'II',                   # Only Part II
        'chapter': 'III'                # Only Chapter III
    },
    top_k=10
)
```

## Document-Specific Search

Search in specific documents:

```python
results = await retriever.search(
    query="záruční doba",
    document_ids=["contract_001", "law_code_089"],
    top_k=5
)
```

## Performance

Target performance metrics:

| Operation | Target | Notes |
|-----------|--------|-------|
| Single query (10k chunks) | <200ms | All three strategies |
| Batch (10 queries) | <500ms | Parallel execution |
| Cache hit | <5ms | In-memory cache |
| Score fusion | <10ms | For 100 candidates |

## Deduplication

The system automatically deduplicates results:

- Results from different strategies for the same chunk are merged
- Only highest-scoring instance is kept
- Score breakdown shows contributions from each strategy

## Error Handling

```python
from src.hybrid_retriever import RetrievalError, InvalidWeightsError

try:
    config = RetrievalConfig(
        semantic_weight=0.5,
        keyword_weight=0.3,
        structural_weight=0.3  # Sum > 1.0
    )
    config.validate()
except InvalidWeightsError as e:
    print(f"Configuration error: {e}")
```

## Testing

Run tests:

```bash
pytest tests/test_hybrid_retrieval.py -v
```

## References

- Specification: `specs/05_hybrid_retrieval.md`
- BM25 Algorithm: Okapi BM25 (Robertson et al., 1994)
- BGE-M3 Model: https://huggingface.co/BAAI/bge-m3
- FAISS: https://github.com/facebookresearch/faiss

## Future Enhancements

Planned improvements:

1. **Learning to Rank** - Train ranking model from labeled data
2. **Multi-modal Search** - Include tables and images
3. **ColBERT Late Interaction** - Token-level matching
4. **Hierarchical Chunking** - Parent-child chunk relationships
5. **Fine-tuned Embeddings** - Domain-specific Czech legal embeddings

## License

See LICENSE file in project root.

## Support

For issues or questions, see the main project README or open an issue on GitHub.
