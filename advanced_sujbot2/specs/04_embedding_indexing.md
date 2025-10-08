# Embedding & Indexing Specification - Multi-Document Vector Store

## 1. Purpose

Generate contextualized embeddings for legal chunks and organize them in separate indices for efficient cross-document retrieval.

**Key Innovation**: Multi-index architecture with legal structure awareness and reference-based lookup.

---

## 2. Embedding Strategy

### 2.1 Model Selection

**Chosen Model**: **BAAI/bge-m3**

| Feature | Value | Rationale |
|---------|-------|-----------|
| Dimensions | 1024 | Higher capacity than 384-dim models |
| Languages | 100+ | Excellent Czech support |
| Max sequence | 8192 tokens | Handles long legal texts |
| Performance | SOTA on multilingual | Best-in-class for Czech |
| Hybrid support | Dense + Sparse | Optional BM25 integration |

**Alternatives**:
- `intfloat/multilingual-e5-large` (also 1024 dim, good)
- `sentence-transformers/paraphrase-multilingual-mpnet` (768 dim, legacy)

### 2.2 Contextualized Embeddings

**Problem**: Plain embedding loses hierarchical context.

**Solution**: Inject hierarchy into embedding input:

```python
# Standard approach
embedding = model.encode("Dodavatel odpovídá za vady...")

# Contextualized approach
context = "§89 odst. 2 Odpovědnost za vady | Dodavatel odpovídá za vady..."
embedding = model.encode(context)
```

**Benefits**:
- Embeddings capture legal context
- Better discrimination between similar provisions
- Improved cross-document matching

---

## 3. Multi-Document Vector Store

### 3.1 Architecture

```python
class MultiDocumentVectorStore:
    """
    Separate FAISS indices for each document type
    Enables filtering and cross-document comparison
    """

    def __init__(self, config):
        self.config = config
        self.indices = {}  # {doc_id: FAISSIndex}
        self.metadata_stores = {}  # {doc_id: {chunk_id: chunk_data}}
        self.reference_map = ReferenceMap()  # Cross-document references

    async def add_document(
        self,
        chunks: List[LegalChunk],
        document_id: str,
        document_type: str
    ):
        """Add document to store"""

        # 1. Generate embeddings
        embeddings = await self.embed_chunks(chunks)

        # 2. Create/get index for this document
        if document_id not in self.indices:
            self.indices[document_id] = self._create_faiss_index()

        # 3. Add to index
        self.indices[document_id].add(embeddings)

        # 4. Store metadata
        self.metadata_stores[document_id] = {
            chunk.chunk_id: chunk for chunk in chunks
        }

        # 5. Build reference map
        await self.reference_map.build(chunks)
```

### 3.2 FAISS Index Configuration

```python
def _create_faiss_index(self, vector_size: int = 1024) -> faiss.Index:
    """
    Create FAISS index optimized for legal retrieval

    Options:
    1. IndexFlatIP (current) - exact search, <100k vectors
    2. IndexIVFFlat - approximate, 100k-1M vectors
    3. IndexHNSWFlat - fast approximate, any scale
    """

    if self.config.index_type == 'flat':
        # Exact search with inner product
        index = faiss.IndexFlatIP(vector_size)

    elif self.config.index_type == 'ivf':
        # Approximate search with inverted file
        quantizer = faiss.IndexFlatIP(vector_size)
        index = faiss.IndexIVFFlat(
            quantizer,
            vector_size,
            nlist=100  # Number of clusters
        )
        # Train index
        index.train(training_vectors)

    elif self.config.index_type == 'hnsw':
        # HNSW graph-based search
        index = faiss.IndexHNSWFlat(
            vector_size,
            M=32  # Number of connections
        )

    return index
```

---

## 4. Embedding Generation

```python
class LegalEmbedder:
    """Generate contextualized embeddings for legal chunks"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    async def embed_chunks(
        self,
        chunks: List[LegalChunk],
        add_context: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for chunks with optional context

        Args:
            chunks: List of legal chunks
            add_context: Whether to add hierarchical context

        Returns:
            Normalized embedding matrix (N x 1024)
        """

        # Prepare texts
        if add_context:
            texts = [self._contextualize(chunk) for chunk in chunks]
        else:
            texts = [chunk.content for chunk in chunks]

        # Batch encode
        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            batch_size=32,
            normalize_embeddings=True,  # L2 normalization
            show_progress_bar=True,
            device=self.device
        )

        return embeddings

    def _contextualize(self, chunk: LegalChunk) -> str:
        """
        Add hierarchical context to chunk content

        Format:
        "{legal_reference} | {hierarchy_path} | {content}"
        """

        context_parts = [chunk.legal_reference]

        # Add hierarchy path
        if chunk.hierarchy_path:
            context_parts.append(chunk.hierarchy_path)

        # Add content
        context_parts.append(chunk.content)

        return " | ".join(context_parts)
```

---

## 5. Reference Map

```python
class ReferenceMap:
    """
    Maps legal references across documents
    Enables fast lookup: "§89" → list of chunks
    """

    def __init__(self):
        self.ref_to_chunks = defaultdict(list)  # {ref: [chunk_ids]}
        self.chunk_to_refs = {}  # {chunk_id: [refs]}

    async def build(self, chunks: List[LegalChunk]):
        """Build reference map from chunks"""

        for chunk in chunks:
            # Index by legal reference
            ref = chunk.legal_reference
            self.ref_to_chunks[ref].append(chunk.chunk_id)

            # Index outgoing references
            outgoing = chunk.metadata.get('references_to', [])
            self.chunk_to_refs[chunk.chunk_id] = outgoing

    def get_chunks_by_reference(self, legal_ref: str) -> List[str]:
        """
        Get all chunks matching a legal reference

        Example:
        get_chunks_by_reference("§89") → ["chunk_123", "chunk_456"]
        """
        return self.ref_to_chunks.get(legal_ref, [])

    def get_references_from_chunk(self, chunk_id: str) -> List[str]:
        """Get all references cited by a chunk"""
        return self.chunk_to_refs.get(chunk_id, [])
```

---

## 6. Retrieval Interface

```python
class VectorStoreRetriever:
    """Retrieve chunks from multi-document store"""

    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Semantic search across one or more documents

        Args:
            query: Search query
            document_ids: Filter by document IDs (None = all)
            top_k: Number of results
            filter_metadata: Metadata filters (e.g., {'content_type': 'obligation'})

        Returns:
            List of search results with scores
        """

        # 1. Embed query
        query_embedding = await self.embedder.embed_chunks([
            LegalChunk(content=query, ...)
        ])
        query_vector = query_embedding[0]

        # 2. Search in selected indices
        all_results = []

        indices_to_search = document_ids or list(self.indices.keys())

        for doc_id in indices_to_search:
            index = self.indices[doc_id]

            # FAISS search
            scores, indices = index.search(
                query_vector.reshape(1, -1),
                top_k
            )

            # Map to chunks
            metadata_store = self.metadata_stores[doc_id]
            chunk_ids = list(metadata_store.keys())

            for score, idx in zip(scores[0], indices[0]):
                if idx < len(chunk_ids):
                    chunk_id = chunk_ids[idx]
                    chunk = metadata_store[chunk_id]

                    # Apply metadata filters
                    if filter_metadata and not self._matches_filter(chunk, filter_metadata):
                        continue

                    all_results.append(SearchResult(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        score=float(score),
                        document_id=doc_id
                    ))

        # 3. Merge and sort
        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:top_k]

    def _matches_filter(self, chunk: LegalChunk, filters: Dict) -> bool:
        """Check if chunk matches metadata filters"""
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True

    async def search_by_reference(
        self,
        legal_ref: str,
        document_type: Optional[str] = None
    ) -> Optional[LegalChunk]:
        """
        Direct lookup by legal reference

        Example:
        search_by_reference("§89 odst. 2", document_type="law_code")
        """

        chunk_ids = self.reference_map.get_chunks_by_reference(legal_ref)

        if not chunk_ids:
            return None

        # Filter by document type if specified
        if document_type:
            chunk_ids = [
                cid for cid in chunk_ids
                if self._get_chunk(cid).document_type == document_type
            ]

        # Return first match (or most relevant)
        return self._get_chunk(chunk_ids[0]) if chunk_ids else None
```

---

## 7. Index Persistence

```python
class IndexPersistence:
    """Save and load indices from disk"""

    def __init__(self, index_dir: Path = Path("./indexes")):
        self.index_dir = index_dir

    async def save(
        self,
        document_id: str,
        index: faiss.Index,
        metadata: Dict,
        chunks: List[LegalChunk]
    ):
        """Save index and metadata to disk"""

        doc_dir = self.index_dir / document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(index, str(doc_dir / "faiss.index"))

        # Save metadata
        with open(doc_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save chunks
        with open(doc_dir / "chunks.pkl", 'wb') as f:
            pickle.dump(chunks, f)

        # Save reference map
        with open(doc_dir / "reference_map.json", 'w') as f:
            json.dump(self._serialize_reference_map(), f)

    async def load(self, document_id: str) -> Tuple[faiss.Index, Dict, List]:
        """Load index and metadata from disk"""

        doc_dir = self.index_dir / document_id

        # Load FAISS index
        index = faiss.read_index(str(doc_dir / "faiss.index"))

        # Load metadata
        with open(doc_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        # Load chunks
        with open(doc_dir / "chunks.pkl", 'rb') as f:
            chunks = pickle.load(f)

        return index, metadata, chunks
```

---

## 8. Configuration

```yaml
# config.yaml
embedding:
  model: BAAI/bge-m3
  device: cuda  # cuda | cpu | mps
  batch_size: 32
  max_sequence_length: 8192
  normalize: true

  # Contextualization
  add_hierarchical_context: true
  context_format: "{legal_ref} | {hierarchy} | {content}"

vector_store:
  index_type: flat  # flat | ivf | hnsw
  vector_size: 1024

  # For IVF
  ivf_nlist: 100  # Number of clusters

  # For HNSW
  hnsw_m: 32  # Number of connections
  hnsw_ef_construction: 200
  hnsw_ef_search: 50

  # Performance
  enable_gpu: false  # Use GPU for FAISS (if available)

persistence:
  index_dir: ./indexes
  auto_save: true
  compression: false  # Compress indices
```

---

## 9. Performance Optimization

### 9.1 Batch Embedding

```python
async def embed_chunks_batched(
    chunks: List[LegalChunk],
    batch_size: int = 32
) -> np.ndarray:
    """Embed chunks in batches for better GPU utilization"""

    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = await self.embedder.embed_chunks(batch)
        all_embeddings.append(embeddings)

    return np.vstack(all_embeddings)
```

### 9.2 Caching

```python
from functools import lru_cache

class EmbeddingCache:
    """Cache embeddings for frequently accessed chunks"""

    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size

    async def get_or_compute(
        self,
        chunk_id: str,
        chunk: LegalChunk
    ) -> np.ndarray:
        """Get from cache or compute"""

        if chunk_id in self.cache:
            return self.cache[chunk_id]

        # Compute
        embedding = await self.embedder.embed_chunks([chunk])

        # Store (with LRU eviction)
        if len(self.cache) >= self.max_size:
            # Remove oldest
            oldest = next(iter(self.cache))
            del self.cache[oldest]

        self.cache[chunk_id] = embedding[0]

        return embedding[0]
```

---

## 10. Testing

```python
def test_embedding_generation():
    """Test embedding generation"""
    embedder = LegalEmbedder()

    chunk = LegalChunk(
        content="Dodavatel odpovídá za vady",
        legal_reference="§89",
        hierarchy_path="Část II > §89"
    )

    embeddings = embedder.embed_chunks([chunk])

    assert embeddings.shape == (1, 1024)
    assert np.allclose(np.linalg.norm(embeddings[0]), 1.0)  # Normalized


def test_multi_index_search():
    """Test searching across multiple indices"""
    store = MultiDocumentVectorStore()

    # Add contract
    store.add_document(contract_chunks, "contract_1", "contract")

    # Add law
    store.add_document(law_chunks, "law_89_2012", "law_code")

    # Search in both
    results = store.search("odpovědnost za vady", top_k=5)

    assert len(results) <= 5
    assert any(r.document_id == "contract_1" for r in results)
    assert any(r.document_id == "law_89_2012" for r in results)


def test_reference_lookup():
    """Test direct reference lookup"""
    store = MultiDocumentVectorStore()

    chunk = store.search_by_reference("§89 odst. 2")

    assert chunk is not None
    assert chunk.legal_reference == "§89 odst. 2"
```

---

## 11. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Embedding generation | <1s per 100 chunks | GPU: <0.3s |
| Index creation | <5s per 10k chunks | |
| Search (10k chunks) | <50ms | Flat index |
| Search (100k chunks) | <100ms | IVF index |
| Search (1M chunks) | <200ms | HNSW index |
| Memory per 10k chunks | ~50 MB | Flat index |

---

## 12. Error Handling

```python
class EmbeddingError(Exception):
    """Base exception for embedding errors"""
    pass

class ModelLoadError(EmbeddingError):
    """Failed to load embedding model"""
    pass

class IndexCorruptedError(EmbeddingError):
    """FAISS index is corrupted"""
    pass

# Usage:
try:
    embeddings = await embedder.embed_chunks(chunks)
except ModelLoadError:
    logger.error("Failed to load embedding model")
    # Fallback to simpler model
except torch.cuda.OutOfMemoryError:
    logger.warning("GPU OOM, falling back to CPU")
    embedder.device = 'cpu'
```
