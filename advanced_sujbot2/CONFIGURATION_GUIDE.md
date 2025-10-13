# SUJBOT2 Configuration Guide

## Overview

SUJBOT2 uses a **hierarchical configuration system** that loads settings from multiple sources with the following priority:

1. **Environment Variables** (highest priority)
2. **config.yaml** file
3. **Default values** (lowest priority)

This allows you to:
- Store default configuration in `config.yaml`
- Override specific values via environment variables
- Keep secrets (API keys) out of version control

## Quick Start

### 1. Basic Setup

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Claude API key
# CLAUDE_API_KEY=your-api-key-here
```

### 2. Loading Configuration in Code

```python
from src.config import load_config
from src.config_factory import create_retrieval_config, create_embedding_config

# Load central config from config.yaml
config = load_config("config.yaml")

# Create component-specific configs using factory functions
retrieval_config = create_retrieval_config(config)
embedding_config = create_embedding_config(config)
reranking_config = create_reranking_config(config)
```

### 3. Accessing Configuration Values

```python
# Use dot notation for nested values
semantic_weight = config.get("retrieval.semantic_weight")  # 0.5
top_k = config.get("retrieval.top_k")  # 20
model = config.get("embeddings.model")  # "joelniklaus/legal-xlm-roberta-base"

# With default values
batch_size = config.get("embeddings.batch_size", 32)
```

## Configuration Structure

### config.yaml Sections

```yaml
# RAG Retrieval (Triple Hybrid Search)
retrieval:
  semantic_weight: 0.5      # Weight for semantic search
  keyword_weight: 0.3       # Weight for keyword (BM25) search
  structural_weight: 0.2    # Weight for structural search
  top_k: 20                 # Number of results before reranking
  bm25:
    k1: 1.5                 # BM25 saturation parameter
    b: 0.75                 # BM25 length normalization

# Embeddings
embeddings:
  model: "joelniklaus/legal-xlm-roberta-base"  # Legal XLM-RoBERTa (768 dim, Czech legal-specialized)
  device: "cpu"             # cpu | cuda | mps | auto
  batch_size: 32
  max_sequence_length: 512  # Legal XLM-RoBERTa max length

# Reranking
reranking:
  cross_encoder_model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
  final_top_k: 5            # Number of final results after reranking
  ensemble_weights:
    cross_encoder: 0.5      # Semantic relevance weight
    graph: 0.3              # Graph structure weight
    precedence: 0.2         # Legal hierarchy weight

# Cross-Document Retrieval
cross_document:
  explicit_weight: 0.5      # Weight for explicit reference matches
  semantic_weight: 0.3      # Weight for semantic similarity
  structural_weight: 0.2    # Weight for structural patterns
  semantic_similarity_threshold: 0.75

# Knowledge Graph
knowledge_graph:
  enable: true
  semantic_link_threshold: 0.75
  max_hops: 3
  graph_boost_factor: 1.2   # Boost for important nodes
```

## Environment Variable Overrides

### Priority System

Environment variables **override** config.yaml values. This is useful for:
- Development vs Production settings
- CI/CD pipelines
- Docker deployments
- Keeping secrets secure

### Naming Convention

ENV variable names follow this pattern:
```
SECTION_SUBSECTION_PARAMETER
```

Examples:
```bash
# Retrieval weights
RETRIEVAL_SEMANTIC_WEIGHT=0.6
RETRIEVAL_KEYWORD_WEIGHT=0.2
RETRIEVAL_STRUCTURAL_WEIGHT=0.2

# Top-K parameters
RETRIEVAL_TOP_K=30
RERANKING_FINAL_TOP_K=10

# Cross-document thresholds
CROSS_DOC_SEMANTIC_THRESHOLD=0.8
CROSS_DOC_EXPLICIT_WEIGHT=0.6

# Knowledge graph
KNOWLEDGE_GRAPH_BOOST_FACTOR=1.5

# Models
EMBEDDING_MODEL=joelniklaus/legal-xlm-roberta-base
RERANKING_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

### Complete ENV Variable List

See `.env.example` for the complete list of configurable parameters.

## Component-Specific Configurations

### Hybrid Retriever

```python
from src.config import load_config
from src.config_factory import create_retrieval_config
from src.hybrid_retriever import create_hybrid_retriever

config = load_config("config.yaml")
retrieval_config = create_retrieval_config(config)

# Use retrieval_config with hybrid retriever
retriever = create_hybrid_retriever(vector_store, embedder, retrieval_config)
```

**Key Parameters:**
- `semantic_weight`, `keyword_weight`, `structural_weight`: Must sum to 1.0
- `top_k`: Number of candidates before reranking
- `bm25_k1`, `bm25_b`: BM25 algorithm parameters
- `reference_boost`: Extra weight when legal references detected

### Reranking Pipeline

```python
from src.config import load_config
from src.config_factory import create_reranking_config
from src.reranker import RerankingPipeline

config = load_config("config.yaml")
reranking_config = create_reranking_config(config)

pipeline = RerankingPipeline(reranking_config, knowledge_graph)
```

**Key Parameters:**
- `cross_encoder_model`: Multilingual cross-encoder for relevance scoring
- `graph_proximity_weight`, `graph_centrality_weight`, `graph_authority_weight`: Graph-based scoring
- `precedence_weights`: Legal hierarchy weights (constitutional > statutory > regulatory)
- `ensemble_weights`: Final score fusion weights (must sum to 1.0)

### Cross-Document Retrieval

```python
from src.config import load_config
from src.config_factory import create_cross_doc_config
from src.cross_doc_retrieval import ComparativeRetriever

config = load_config("config.yaml")
cross_doc_config = create_cross_doc_config(config)

retriever = ComparativeRetriever(vector_store, embedder, reference_map, cross_doc_config)
```

**Key Parameters:**
- `explicit_weight`, `semantic_weight`, `structural_weight`: Strategy weights (must sum to 1.0)
- `semantic_similarity_threshold`: Minimum cosine similarity for matches
- `top_k_per_source`: Number of target matches per source chunk

## Validation

The configuration system includes automatic validation:

```python
from src.config import load_config
from src.config_factory import validate_config

config = load_config("config.yaml")

# Validate entire config
validate_config(config)  # Raises ValueError if invalid
```

**Validation Checks:**
- Weights sum to 1.0 (retrieval, cross-document, reranking ensemble)
- Thresholds are in valid ranges (0-1)
- Required fields are present (e.g., API key)
- Device names are valid (cpu, cuda, mps, auto)

## Testing Configuration

Run the configuration test script to verify everything works:

```bash
python3 test_config.py
```

This tests:
- Loading from config.yaml
- ENV variable overrides
- Factory functions for component configs
- Configuration validation

## Common Use Cases

### Development Environment

```bash
# .env for development
CLAUDE_API_KEY=your-dev-key
LOG_LEVEL=DEBUG
RETRIEVAL_TOP_K=10  # Faster for testing
```

### Production Environment

```bash
# .env for production
CLAUDE_API_KEY=your-prod-key
LOG_LEVEL=INFO
RETRIEVAL_TOP_K=20
RERANKING_FINAL_TOP_K=5
EMBEDDING_DEVICE=cuda  # Use GPU
```

### High Precision Mode (Legal/Compliance)

```bash
# Prioritize precision over recall
RETRIEVAL_SEMANTIC_WEIGHT=0.6
RETRIEVAL_KEYWORD_WEIGHT=0.25
RETRIEVAL_STRUCTURAL_WEIGHT=0.15
RETRIEVAL_MIN_SCORE_THRESHOLD=0.3
RERANKING_MIN_CONFIDENCE=0.2
```

### High Recall Mode (Research/Exploration)

```bash
# Retrieve more candidates
RETRIEVAL_TOP_K=30
RETRIEVAL_MIN_SCORE_THRESHOLD=0.05
RERANKING_FINAL_TOP_K=10
```

## Best Practices

1. **Keep Secrets in .env**: Never commit API keys to git
2. **Use config.yaml for Defaults**: Store sensible defaults in config.yaml
3. **Override with ENV**: Use ENV variables for deployment-specific settings
4. **Validate Early**: Call `validate_config()` at startup
5. **Document Changes**: Update this guide when adding new parameters
6. **Test Locally**: Use `test_config.py` to verify changes

## Troubleshooting

### "Configuration file not found"
- Ensure `config.yaml` exists in the project root
- Or specify path: `load_config("/path/to/config.yaml")`

### "Weights must sum to 1.0"
- Check retrieval, cross-document, or reranking ensemble weights
- Adjust so they sum to exactly 1.0 (allow 0.01 tolerance)

### "API key missing"
- Set `CLAUDE_API_KEY` in `.env` file
- Or set it as environment variable: `export CLAUDE_API_KEY=xxx`

### ENV variables not working
- Check spelling (case-sensitive)
- Use correct format: `SECTION_PARAMETER=value`
- Reload config after changing ENV: `config = load_config("config.yaml")`

## Migration Guide

### From Hardcoded Values

**Before:**
```python
retriever = HybridRetriever(
    semantic_searcher,
    keyword_searcher,
    structural_searcher,
    RetrievalConfig(
        semantic_weight=0.5,  # Hardcoded
        keyword_weight=0.3,
        structural_weight=0.2,
        top_k=20
    )
)
```

**After:**
```python
config = load_config("config.yaml")
retrieval_config = create_retrieval_config(config)
retriever = HybridRetriever(
    semantic_searcher,
    keyword_searcher,
    structural_searcher,
    retrieval_config  # From config
)
```

## Related Files

- `config.yaml`: Main configuration file
- `.env.example`: Example environment variables
- `src/config.py`: Configuration loader
- `src/config_factory.py`: Factory functions for component configs
- `test_config.py`: Configuration test script
