# Multi-Agent Benchmark System

Comprehensive benchmark system for evaluating multi-agent RAG performance on standardized QA datasets.

**Status:** ✅ Migrated to multi-agent (2025-11-11)

---

## 🎯 Overview

The benchmark system evaluates multi-agent workflow against ground-truth QA datasets, providing:

- **Standard metrics:** Exact Match, F1 Score, Precision, Recall
- **Performance tracking:** Response time, API cost per query
- **Comparative analysis:** Compare single-agent vs multi-agent results
- **Reproducible evaluation:** Deterministic settings for apples-to-apples comparison

---

## 📂 Directory Structure

```
benchmark_dataset/       # QA datasets
  privacy_qa.json        # PrivacyQA dataset (default)
  privacy_qa/            # Source documents

benchmark_db/            # Indexed vector stores for benchmark
  faiss_layer*.index     # Multi-layer FAISS indexes (FAISS backend)
  bm25_layer*.pkl        # BM25 indexes
  unified_kg.json        # Knowledge graph (optional)

src/benchmark/           # Benchmark system code
  config.py              # Configuration
  dataset.py             # Dataset loading
  metrics.py             # Metric computation
  multi_agent_runner.py  # Multi-agent benchmark runner
  report.py              # Result formatting
```

**Note:** Benchmark results are no longer stored in repository. Run benchmarks locally and results will be generated in `benchmark_results/` (gitignored).

---

## 🚀 Quick Start

### 1. **Index Benchmark Documents**

```bash
# Index documents to benchmark_db
uv run python run_pipeline.py benchmark_dataset/privacy_qa --output-dir benchmark_db
```

### 2. **Run Benchmark**

```bash
# Full evaluation (all queries)
uv run python run_benchmark.py

# Quick test (10 queries)
uv run python run_benchmark.py --max-queries 10

# Debug mode
uv run python run_benchmark.py --debug --max-queries 3
```

### 3. **View Results**

```bash
# Human-readable summary
cat benchmark_results/MULTI-AGENT_*.md

# Detailed JSON
jq . benchmark_results/MULTI-AGENT_*.json
```

---

## 🔧 Configuration Options

### CLI Arguments

```bash
uv run python run_benchmark.py \
  --dataset benchmark_dataset/privacy_qa.json \  # QA dataset
  --vector-store benchmark_db \                  # Indexed store
  --max-queries 20 \                             # Limit queries (optional)
  --model claude-sonnet-4-5 \                    # LLM model
  --temperature 0.0 \                            # Deterministic (default)
  --k 5 \                                        # Retrieval chunks
  --no-reranking \                               # Disable reranking
  --no-caching \                                 # Disable prompt caching
  --debug \                                      # Verbose logging
  --rate-limit-delay 1.0                         # Delay between queries (sec)
```

### Environment Variables

```bash
# Alternative: use .env file
BENCHMARK_DATASET=benchmark_dataset/privacy_qa.json
BENCHMARK_VECTOR_STORE=benchmark_db
BENCHMARK_K=5
BENCHMARK_MAX_QUERIES=10
BENCHMARK_DEBUG=false
BENCHMARK_RERANKING=true
BENCHMARK_AGENT_MODEL=claude-haiku-4-5
BENCHMARK_RATE_LIMIT_DELAY=0.0
```

---

## 📊 Metrics Explained

### Standard Metrics

1. **Exact Match (EM):** Binary match (1.0 or 0.0)
   - Measures if predicted answer **exactly matches** any expected answer
   - Strict metric, case-sensitive

2. **F1 Score:** Token-level F1 (0.0 - 1.0)
   - Harmonic mean of precision and recall
   - Measures token overlap between prediction and ground truth
   - Industry standard for QA evaluation

3. **Precision:** Fraction of predicted tokens that are correct
   - High precision = few false positives

4. **Recall:** Fraction of ground truth tokens found in prediction
   - High recall = few false negatives

### Performance Metrics

- **Retrieval Time (ms):** End-to-end query processing time
- **Cost (USD):** API cost for LLM calls (tracked via cost tracker)
- **RAG Confidence:** Quality score from RAG confidence scorer (if available)

---

## 📝 Output Format

### JSON Output (`MULTI-AGENT_*.json`)

```json
{
  "dataset_name": "PrivacyQA (Multi-Agent)",
  "total_queries": 50,
  "aggregate_metrics": {
    "exact_match": 0.72,
    "f1_score": 0.85,
    "precision": 0.88,
    "recall": 0.83
  },
  "query_results": [
    {
      "query_id": 1,
      "query": "What data does the app collect?",
      "predicted_answer": "...",
      "ground_truth_answers": ["..."],
      "metrics": {
        "exact_match": 1.0,
        "f1_score": 0.92
      },
      "retrieval_time_ms": 1250.5,
      "cost_usd": 0.002341
    }
  ],
  "total_time_seconds": 125.3,
  "total_cost_usd": 0.11703,
  "config": {
    "agent_model": "claude-haiku-4-5",
    "k": 5,
    "enable_reranking": true
  }
}
```

### Markdown Output (`MULTI-AGENT_*.md`)

Human-readable summary with:
- **Executive Summary:** Aggregate metrics, cost, time
- **Per-Query Results:** Table with metrics for each query
- **Configuration:** All benchmark settings

---

## 🔬 Running Benchmarks

### Multi-Agent Benchmark

```bash
# Run full benchmark
uv run python run_benchmark.py

# Quick test
uv run python run_benchmark.py --max-queries 10 --debug
```

### Expected Performance

Based on testing with multi-agent system:
- **F1 score:** ~0.85 on PrivacyQA dataset
- **Exact Match:** ~0.72
- **Cost:** $0.002-0.01 per query (with prompt caching)
- **Latency:** 3-8s per query (depends on complexity)

---

## 📚 Dataset Format

### JSON Structure

```json
{
  "dataset_name": "PrivacyQA",
  "version": "1.0",
  "queries": [
    {
      "query_id": 1,
      "query": "What personal data does the app collect?",
      "expected_answers": [
        "location data, device ID, usage statistics",
        "GPS coordinates, unique device identifier, app usage logs"
      ],
      "source_document": "privacy_policy.pdf",
      "category": "data_collection"
    }
  ]
}
```

### Creating Custom Datasets

1. **Format:** Follow JSON structure above
2. **Expected answers:** Provide multiple acceptable phrasings
3. **Source documents:** Index to vector store before benchmarking
4. **Categories:** Optional, for per-category analysis

---

## 🐛 Troubleshooting

### "Vector store not found"

```bash
# Index documents first
uv run python run_pipeline.py benchmark_dataset/privacy_qa --output-dir benchmark_db
```

### "Dataset not found"

```bash
# Check path
ls -la benchmark_dataset/privacy_qa.json

# Use custom path
uv run python run_benchmark.py --dataset path/to/your/dataset.json
```

### "Multi-agent config not found"

```bash
# Ensure config exists
ls -la config_multi_agent_extension.json

# Or add to config.json
cat config.json | jq '.multi_agent'
```

### Low F1 Scores (<0.5)

- **Check k:** Try `--k 10` for more context
- **Enable reranking:** Remove `--no-reranking`
- **Verify indexing:** Re-index with correct settings
- **Debug mode:** Use `--debug --max-queries 3` to inspect answers

---

## 🎯 Best Practices

### For Reproducible Evaluation

1. **Fixed temperature:** Always use `--temperature 0.0`
2. **Same k:** Use consistent `--k` across runs
3. **Same model:** Don't compare Haiku vs Sonnet results
4. **Full dataset:** Run all queries (avoid `--max-queries` for final results)

### For Development

1. **Quick tests:** `--max-queries 10` for iteration
2. **Debug mode:** `--debug` for per-query inspection
3. **Cost control:** Start with `--no-caching` to verify correctness

### For Production

1. **Enable caching:** Default `--caching` for 90% cost savings
2. **Enable reranking:** Default `--reranking` for +8% accuracy
3. **Rate limiting:** Use `--rate-limit-delay` for API quotas

---

## 📖 Related Documentation

- [`MULTI_AGENT_STATUS.md`](MULTI_AGENT_STATUS.md) - Multi-agent architecture
- [`PIPELINE.md`](PIPELINE.md) - Document indexing pipeline
- [`README.md`](README.md) - General system overview

---

**Last Updated:** 2025-11-11
**Version:** Multi-Agent v2.0
