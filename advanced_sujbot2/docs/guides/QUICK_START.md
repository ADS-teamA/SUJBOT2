# Quick Start Guide - Legal Embedding & Indexing

## Installation

```bash
# Clone repository
cd advanced_sujbot2

# Install dependencies
pip install -r requirements.txt

# or with uv (faster)
uv pip install -r requirements.txt
```

## Basic Usage

### 1. Create Embedder

```python
from src.embeddings import LegalEmbedder, LegalChunk

# Create embedder (auto-detects GPU/CPU)
embedder = LegalEmbedder()

# Create a legal chunk
chunk = LegalChunk(
    chunk_id="chunk_1",
    content="Dodavatel odpovídá za vady díla.",
    legal_reference="§89",
    hierarchy_path="Část II > Hlava III > §89",
    metadata={"content_type": "obligation"}
)

# Generate embedding
embeddings = await embedder.embed_chunks([chunk])
print(f"Shape: {embeddings.shape}")  # (1, 1024)
```

### 2. Create Vector Store

```python
from src.indexing import MultiDocumentVectorStore

# Create vector store
vector_store = MultiDocumentVectorStore(embedder)

# Add document
await vector_store.add_document(
    chunks=[chunk],
    document_id="law_89_2012",
    document_type="law_code"
)
```

### 3. Search

```python
# Semantic search
results = await vector_store.search("odpovědnost za vady", top_k=5)

for result in results:
    print(f"{result.rank}. {result.chunk.legal_reference}: {result.score:.4f}")
    print(f"   {result.chunk.content[:100]}...")

# Reference lookup
chunk = await vector_store.search_by_reference("§89")
print(f"Found: {chunk.get_citation()}")
```

### 4. Save/Load

```python
from src.indexing import IndexPersistence

persistence = IndexPersistence(index_dir="./indexes")

# Save
await persistence.save(
    document_id="law_89_2012",
    index=vector_store.indices["law_89_2012"],
    metadata=vector_store.document_info["law_89_2012"],
    chunks=[chunk],
    reference_map=vector_store.reference_map
)

# Load
index, metadata, chunks, ref_map = await persistence.load("law_89_2012")
```

## Run Example

```bash
python example_usage.py
```

This demonstrates:
- Embedding generation
- Multi-document indexing
- Semantic search
- Reference lookup
- Persistence

## Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

Edit `config.yaml`:

```yaml
# Use GPU
embeddings:
  device: "cuda"  # or "mps" for Apple Silicon

# Use IVF index for large documents
indexing:
  index_type: "ivf"  # flat | ivf | hnsw
```

## Common Tasks

### Add Multiple Documents

```python
# Add law
await vector_store.add_document(law_chunks, "law_89_2012", "law_code")

# Add contract
await vector_store.add_document(contract_chunks, "contract_123", "contract")

# Search across both
results = await vector_store.search("dodavatel", top_k=10)
```

### Filter by Metadata

```python
# Only obligations
results = await vector_store.search(
    "dodavatel",
    filter_metadata={"content_type": "obligation"}
)

# Only in specific document
results = await vector_store.search(
    "dodavatel",
    document_ids=["law_89_2012"]
)
```

### Cache Embeddings

```python
from src.embeddings import EmbeddingCache

cache = EmbeddingCache(max_size=10000)

# Get or compute
embedding = await cache.get_or_compute(chunk.chunk_id, chunk, embedder)

# Stats
stats = cache.get_cache_stats()
print(f"Cache usage: {stats['usage_percent']}%")
```

## Next Steps

1. Read `EMBEDDING_INDEXING_SUMMARY.md` for details
2. Check `example_usage.py` for more examples
3. Run tests to verify installation
4. Integrate with document reader (spec 02)
5. Add hybrid retrieval (spec 05)

## Troubleshooting

**ImportError: No module named 'faiss'**
```bash
pip install faiss-cpu
# or for GPU: pip install faiss-gpu
```

**CUDA out of memory**
```python
# Use CPU instead
config = EmbeddingConfig(device="cpu")
embedder = LegalEmbedder(config)
```

**Slow embedding generation**
```python
# Increase batch size
config = EmbeddingConfig(batch_size=64)

# Or use GPU
config = EmbeddingConfig(device="cuda")
```

## Support

- Documentation: `EMBEDDING_INDEXING_SUMMARY.md`
- Examples: `example_usage.py`
- Tests: `tests/test_*.py`
- Specification: `specs/04_embedding_indexing.md`
