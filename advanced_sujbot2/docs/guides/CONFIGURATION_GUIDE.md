# Configuration Guide - SUJBOT2

This guide explains the complete configuration system for SUJBOT2, including all parameters, environment variables, and usage patterns.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Files](#configuration-files)
3. [Environment Variables](#environment-variables)
4. [Configuration Parameters](#configuration-parameters)
5. [Configuration Priority](#configuration-priority)
6. [Usage Examples](#usage-examples)
7. [Presets](#presets)
8. [Validation](#validation)
9. [Utilities](#utilities)

---

## Quick Start

### 1. Copy Environment Template

```bash
cp .env.example .env
```

### 2. Set Your API Key

Edit `.env` and add your Claude API key:

```bash
CLAUDE_API_KEY=sk-ant-your-actual-key-here
```

### 3. Use Configuration in Code

```python
from src.config import load_config
from src.logging_config import setup_logging_from_config

# Load configuration
config = load_config("config.yaml")

# Setup logging
setup_logging_from_config(config.to_dict())

# Access values
model = config.get("llm.main_model")
alpha = config.get("retrieval.semantic_weight")
```

---

## Configuration Files

### `config.yaml` - Main Configuration

The main configuration file with all system parameters organized into sections:

- **Retrieval**: Triple hybrid search weights and parameters
- **Embeddings**: Model selection and embedding parameters
- **Indexing**: FAISS index configuration
- **Chunking**: Document chunking strategy
- **Reranking**: Cross-encoder reranking settings
- **Cross-Document**: Cross-document retrieval settings
- **Compliance**: Compliance analysis configuration
- **Knowledge Graph**: Graph construction and reasoning
- **LLM**: Claude API configuration
- **API**: API server settings
- **Logging**: Logging configuration
- **Performance**: Performance tuning

### `.env` - Environment Variables

Sensitive configuration and environment-specific overrides:

- API keys (never commit to git!)
- Model selections
- Performance settings
- Feature flags
- Logging levels

### `config.example.yaml` - Configuration Template

Template showing all available options with explanations.

---

## Environment Variables

### Required

| Variable | Description | Default |
|----------|-------------|---------|
| `CLAUDE_API_KEY` | Claude API key from console.anthropic.com | **REQUIRED** |

### Model Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MAIN_AGENT_MODEL` | Main model for answer synthesis | `claude-sonnet-4-5-20250929` |
| `SUBAGENT_MODEL` | Model for question decomposition | `claude-3-5-haiku-20241022` |

### Performance

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_PARALLEL_AGENTS` | Max parallel tasks | `10` |
| `MAX_WORKERS` | Max indexing workers | `8` |

### Feature Flags

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_QUESTION_DECOMPOSITION` | Enable query decomposition | `true` |
| `ENABLE_QUERY_EXPANSION` | Enable query expansion | `true` |
| `ENABLE_RERANKING` | Enable cross-encoder reranking | `true` |

### Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level | `INFO` |
| `VERBOSE_LOGGING` | Enable debug logging | `false` |
| `LOG_FILE` | Log file path | `logs/system.log` |

### Retrieval

| Variable | Description | Default |
|----------|-------------|---------|
| `HYBRID_ALPHA` | Hybrid search weight (0-1) | `0.7` |
| `TOP_K` | Candidates before reranking | `20` |
| `RERANK_TOP_K` | Final results after reranking | `5` |
| `MIN_SCORE` | Minimum relevance threshold | `0.1` |

### Indexing

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_SIZE` | Chunk size in tokens | `512` |
| `CHUNK_OVERLAP` | Overlap as fraction (0-1) | `0.15` |
| `INDEX_DIR` | Index storage directory | `./indexes` |

### Embeddings

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Embedding model name | `BAAI/bge-m3` |
| `EMBEDDING_DEVICE` | Device (cpu/cuda/mps) | `cpu` |
| `EMBEDDING_BATCH_SIZE` | Batch size | `32` |

---

## Configuration Parameters

### Retrieval Configuration

```yaml
retrieval:
  # Triple hybrid weights (must sum to 1.0)
  semantic_weight: 0.5   # Semantic search (embeddings)
  keyword_weight: 0.3    # Keyword search (BM25)
  structural_weight: 0.2  # Structural search (hierarchy/refs)

  # Top-K parameters
  top_k: 20
  candidate_multiplier: 1.5

  # Score normalization
  normalize_scores: true
  normalization_method: min-max

  # BM25 parameters
  bm25:
    k1: 1.5
    b: 0.75

  # Adaptive weighting
  adaptive_weights: true
  reference_boost: 0.2
```

### Embeddings Configuration

```yaml
embeddings:
  model: "BAAI/bge-m3"  # 1024-dim multilingual
  device: "cpu"          # cpu | cuda | mps
  normalize: true        # L2 normalization
  batch_size: 32
  max_sequence_length: 8192
  add_context: true      # Include hierarchy in embeddings
```

### Indexing Configuration

```yaml
indexing:
  index_type: "flat"  # flat | ivf | hnsw
  ivf_nlist: 100      # For IVF only
  hnsw_m: 32          # For HNSW only
  save_indices: true
  index_dir: "data/indices"
```

### Chunking Configuration

```yaml
chunking:
  min_chunk_size: 128
  max_chunk_size: 1024
  target_chunk_size: 512
  chunk_overlap: 0.15
  include_context: true
  aggregate_small_chunks: true
```

### Reranking Configuration

```yaml
reranking:
  enable: true
  model: "cross-encoder/mmarco-mMiniLMv2-L6-H384"
  top_k: 5
  batch_size: 32
  enable_graph_aware: true
  graph_proximity_weight: 0.3
  enable_legal_precedence: true
  precedence_weight: 0.2
```

### Compliance Configuration

```yaml
compliance:
  default_mode: "exhaustive"  # exhaustive | quick
  enable_requirement_extraction: true
  enable_clause_mapping: true
  enable_conflict_detection: true
  enable_gap_analysis: true
  enable_deviation_assessment: true

  severity_thresholds:
    critical: 0.9
    high: 0.7
    medium: 0.5
    low: 0.3

  generate_recommendations: true
  max_recommendations: 10
```

### Knowledge Graph Configuration

```yaml
knowledge_graph:
  enable: true
  build_on_indexing: true
  include_cross_document_links: true
  semantic_link_threshold: 0.75
  max_links_per_node: 20
  enable_multi_hop: true
  max_hops: 3
```

### LLM Configuration

```yaml
llm:
  main_model: "claude-sonnet-4-5-20250929"
  sub_model: "claude-3-5-haiku-20241022"
  timeout: 120
  max_retries: 3
  retry_delay: 1.0
  temperature: 0.1
  max_tokens: 4000
```

---

## Configuration Priority

Configuration values are resolved in this priority order (highest to lowest):

1. **Environment Variables** - Highest priority
2. **Programmatic Config** - Passed to `Config()` constructor
3. **YAML File** - Loaded from `config.yaml`
4. **Default Values** - Built-in defaults

Example:

```python
# Default: chunk_size = 512
# config.yaml: chunk_size = 768
# Programmatic: chunk_size = 1024
# Environment: CHUNK_SIZE=256

# Result: chunk_size = 256 (env var wins)
```

---

## Usage Examples

### Example 1: Basic Usage

```python
from src.config import load_config

config = load_config("config.yaml")
print(config.get("llm.main_model"))
```

### Example 2: Programmatic Overrides

```python
from src.config import load_config

config = load_config(
    config_path="config.yaml",
    config={
        "retrieval": {
            "semantic_weight": 0.8,  # Override
        }
    }
)
```

### Example 3: No Config File (Defaults Only)

```python
from src.config import Config
import os

os.environ["CLAUDE_API_KEY"] = "sk-ant-xxx"
config = Config()  # Use all defaults
```

### Example 4: Save Configuration

```python
config = load_config("config.yaml")
config.set("llm.temperature", 0.2)
config.save("my_config.yaml")
```

### Example 5: Get Full Config as Dict

```python
config = load_config("config.yaml")
config_dict = config.to_dict()
print(config_dict["retrieval"])
```

---

## Presets

Common configuration presets for different use cases:

### High Precision (Legal/Compliance)

```yaml
retrieval:
  semantic_weight: 0.6
  keyword_weight: 0.25
  structural_weight: 0.15
  top_k: 10
  min_score_threshold: 0.3

chunking:
  target_chunk_size: 256
```

**Use when**: Accuracy is critical, false positives must be minimized

### High Recall (Research/Exploration)

```yaml
retrieval:
  semantic_weight: 0.4
  keyword_weight: 0.4
  structural_weight: 0.2
  top_k: 30
  min_score_threshold: 0.05
  enable_query_expansion: true

chunking:
  target_chunk_size: 768
```

**Use when**: Finding all relevant information is more important than precision

### Speed Optimized

```yaml
retrieval:
  semantic_weight: 0.3
  keyword_weight: 0.5
  structural_weight: 0.2
  parallel_retrieval: true
  enable_caching: true

indexing:
  index_type: hnsw
```

**Use when**: Fast response times are critical

---

## Validation

The configuration system automatically validates:

- **API Key**: Must be present
- **Numeric Ranges**: Values within valid ranges (e.g., alpha ∈ [0, 1])
- **Top-K Values**: `rerank_top_k ≤ top_k`
- **Temperature**: Must be in [0, 1]
- **Chunk Sizes**: Positive integers

Example validation error:

```python
from src.exceptions import ConfigurationError

try:
    config = Config(config={
        "retrieval": {
            "hybrid_alpha": 1.5  # Invalid!
        }
    })
except ConfigurationError as e:
    print(f"Validation failed: {e}")
```

---

## Utilities

### Logging Setup

```python
from src.logging_config import setup_logging_from_config, logger

setup_logging_from_config(config.to_dict())
logger.info("System initialized")
```

### Text Processing

```python
from src.utils import (
    normalize_text,
    count_tokens,
    extract_legal_references,
)

text = "§89 odst. 2 stanoví povinnosti"
refs = extract_legal_references(text)  # ['§89 odst. 2']
tokens = count_tokens(text)  # Token count
```

### File Utilities

```python
from src.utils import (
    ensure_dir,
    get_file_hash,
    safe_filename,
)

index_dir = ensure_dir("./indexes")
file_hash = get_file_hash("document.pdf")
safe_name = safe_filename("file: with: colons.pdf")
```

### Performance Utilities

```python
from src.utils import Timer, batch_items

with Timer("Processing"):
    # ... do work ...
    pass

items = [1, 2, 3, 4, 5]
batches = batch_items(items, batch_size=2)  # [[1,2], [3,4], [5]]
```

---

## Best Practices

### 1. Never Commit `.env`

Add to `.gitignore`:

```
.env
*.env.local
```

### 2. Use Environment Variables for Secrets

```bash
# .env
CLAUDE_API_KEY=sk-ant-xxx
```

### 3. Use YAML for Structure

```yaml
# config.yaml
retrieval:
  semantic_weight: 0.5
  keyword_weight: 0.3
```

### 4. Use Programmatic Config for Dynamic Values

```python
config = load_config(
    "config.yaml",
    config={"llm": {"temperature": user_temperature}}
)
```

### 5. Validate Early

```python
try:
    config = load_config("config.yaml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)
```

---

## Troubleshooting

### API Key Error

**Error**: `APIKeyError: Claude API key not found`

**Solution**: Set `CLAUDE_API_KEY` in `.env` or environment

### Invalid Configuration

**Error**: `ConfigurationError: Invalid configuration values`

**Solution**: Check validation errors in exception message

### File Not Found

**Error**: `ConfigurationError: Configuration file not found`

**Solution**: Ensure `config.yaml` exists or pass `config` dict

### Import Error

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Run from project root or install package

---

## Complete Example

```python
#!/usr/bin/env python3
"""
Complete configuration example
"""

import os
from pathlib import Path
from src.config import load_config
from src.logging_config import setup_logging_from_config, logger
from src.utils import ensure_dir

def main():
    # 1. Set API key (in practice, use .env)
    os.environ["CLAUDE_API_KEY"] = "sk-ant-your-key"

    # 2. Load configuration
    config = load_config(
        config_path="config.yaml",
        config={
            "retrieval": {
                "semantic_weight": 0.6,  # High precision
            }
        }
    )

    # 3. Setup logging
    setup_logging_from_config(config.to_dict())
    logger.info("System initialized")

    # 4. Ensure directories exist
    index_dir = ensure_dir(config.get("indexing.index_dir"))
    logger.info(f"Index directory: {index_dir}")

    # 5. Access configuration
    main_model = config.get("llm.main_model")
    logger.info(f"Using model: {main_model}")

    # 6. Save custom config
    config.save("my_config.yaml")
    logger.info("Configuration saved")

if __name__ == "__main__":
    main()
```

---

## References

- **Main Config**: `/config.yaml`
- **Environment Template**: `/.env.example`
- **Config Module**: `/src/config.py`
- **Utilities Module**: `/src/utils.py`
- **Logging Module**: `/src/logging_config.py`
- **Usage Examples**: `/config_usage_example.py`

---

**Last Updated**: 2025-10-08
**Version**: 1.0.0
