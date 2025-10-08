# Hybrid Retrieval Specification - Triple Hybrid Search

## 1. Purpose

Combine three complementary retrieval strategies (semantic, keyword, structural) to achieve optimal precision and recall for legal document search.

**Key Innovation**: Triple hybrid approach that leverages:
- **Semantic search** for meaning and context
- **Keyword search** for exact terms and technical vocabulary
- **Structural search** for legal hierarchy and references

---

## 2. Design Rationale

### 2.1 Why Hybrid?

**Problem with Single-Strategy Retrieval**:

| Strategy | Strengths | Weaknesses |
|----------|-----------|------------|
| **Semantic only** | Captures meaning, synonyms | Misses exact terms, poor for IDs |
| **Keyword only** | Exact matches, fast | Misses paraphrasing, context |
| **Structural only** | Precise references | Requires known structure |

**Solution**: Combine all three with weighted fusion.

### 2.2 Triple Hybrid Architecture

```
Query: "Jaké jsou povinnosti dodavatele ohledně odpovědnosti za vady?"

┌─────────────────────────────────────────────────────────┐
│                    PARALLEL RETRIEVAL                    │
└─────────────────────────────────────────────────────────┘

    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Semantic    │     │  Keyword     │     │  Structural  │
    │  (BGE-M3)    │     │  (BM25)      │     │  (Hierarchy) │
    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ Embedding    │     │ Tokenize     │     │ Parse refs   │
    │ query        │     │ query        │     │ & filters    │
    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ FAISS        │     │ BM25 scorer  │     │ Metadata     │
    │ search       │     │              │     │ filter       │
    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ Top-30       │     │ Top-30       │     │ Top-30       │
    │ semantic     │     │ keyword      │     │ structural   │
    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
           │                    │                    │
           └────────────────────┴────────────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │  SCORE FUSION      │
                    │  α·sem + β·key +   │
                    │  γ·struct          │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Deduplication     │
                    │  & Ranking         │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Top-20 Results    │
                    └────────────────────┘
```

---

## 3. Data Structures

### 3.1 Search Result

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

@dataclass
class SearchResult:
    """Single search result with metadata"""

    # Identity
    chunk_id: str
    chunk: LegalChunk
    document_id: str

    # Scores
    score: float  # Combined score
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    structural_score: Optional[float] = None

    # Ranking
    rank: int = 0

    # Metadata
    retrieval_method: str = "hybrid"  # semantic | keyword | structural | hybrid
    timestamp: datetime = field(default_factory=datetime.now)

    def get_score_breakdown(self) -> Dict[str, float]:
        """Get breakdown of component scores"""
        return {
            'semantic': self.semantic_score or 0.0,
            'keyword': self.keyword_score or 0.0,
            'structural': self.structural_score or 0.0,
            'combined': self.score
        }

@dataclass
class SearchQuery:
    """Parsed search query with components"""

    # Original query
    original_query: str

    # Parsed components
    keywords: List[str]
    entities: List[str]  # Legal references, dates, etc.
    filters: Dict[str, Any]  # Metadata filters

    # Query type
    query_type: str  # factual | analytical | compliance | etc.

    # Retrieval hints
    preferred_strategy: Optional[str] = None  # Override default weights
    boost_fields: Optional[List[str]] = None  # Which metadata to boost

@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval"""

    # Weights (must sum to 1.0)
    semantic_weight: float = 0.5  # α
    keyword_weight: float = 0.3   # β
    structural_weight: float = 0.2  # γ

    # Top-K parameters
    top_k: int = 20  # Final results
    candidate_multiplier: float = 1.5  # Retrieve top_k * multiplier per strategy

    # Score normalization
    normalize_scores: bool = True
    normalization_method: str = "min-max"  # min-max | z-score

    # Filters
    enable_metadata_filtering: bool = True
    enable_score_threshold: bool = True
    min_score_threshold: float = 0.1

    # Performance
    enable_caching: bool = True
    parallel_retrieval: bool = True

    def validate(self):
        """Validate configuration"""
        total_weight = self.semantic_weight + self.keyword_weight + self.structural_weight
        assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"
```

---

## 4. Semantic Search Component

### 4.1 Implementation

```python
# File: src/retrieval/semantic_search.py

import numpy as np
import faiss
from typing import List, Optional
import asyncio

class SemanticSearcher:
    """Semantic search using embeddings and FAISS"""

    def __init__(
        self,
        embedder: LegalEmbedder,
        vector_store: MultiDocumentVectorStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Semantic search using embeddings

        Args:
            query: Search query
            document_ids: Filter by document IDs (None = all)
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of search results sorted by semantic similarity
        """

        # 1. Embed query
        query_embedding = await self._embed_query(query)

        # 2. Select indices to search
        indices_to_search = document_ids or list(self.vector_store.indices.keys())

        # 3. Search in parallel across all indices
        all_results = []

        if len(indices_to_search) > 1:
            # Parallel search
            tasks = [
                self._search_single_index(
                    query_embedding,
                    doc_id,
                    top_k,
                    filters
                )
                for doc_id in indices_to_search
            ]
            results_per_index = await asyncio.gather(*tasks)

            # Flatten
            for results in results_per_index:
                all_results.extend(results)
        else:
            # Single index
            all_results = await self._search_single_index(
                query_embedding,
                indices_to_search[0],
                top_k,
                filters
            )

        # 4. Sort by score and take top-K
        all_results.sort(key=lambda x: x.semantic_score, reverse=True)

        return all_results[:top_k]

    async def _embed_query(self, query: str) -> np.ndarray:
        """Embed query with same model as documents"""

        # Create temporary chunk for consistent embedding
        temp_chunk = LegalChunk(
            chunk_id="query",
            content=query,
            document_type="query",
            hierarchy_path="",
            legal_reference="",
            structural_level="query",
            metadata={}
        )

        embeddings = await self.embedder.embed_chunks([temp_chunk])
        return embeddings[0]

    async def _search_single_index(
        self,
        query_embedding: np.ndarray,
        document_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Search in a single FAISS index"""

        # Get index
        index = self.vector_store.indices[document_id]
        metadata_store = self.vector_store.metadata_stores[document_id]

        # FAISS search
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = index.search(query_vector, top_k * 2)  # Over-retrieve for filtering

        # Map to chunks
        results = []
        chunk_ids = list(metadata_store.keys())

        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(chunk_ids):
                continue

            chunk_id = chunk_ids[idx]
            chunk = metadata_store[chunk_id]

            # Apply metadata filters
            if filters and not self._matches_filters(chunk, filters):
                continue

            result = SearchResult(
                chunk_id=chunk_id,
                chunk=chunk,
                document_id=document_id,
                score=float(score),
                semantic_score=float(score),
                retrieval_method="semantic"
            )
            results.append(result)

            if len(results) >= top_k:
                break

        return results

    def _matches_filters(
        self,
        chunk: LegalChunk,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if chunk matches metadata filters"""

        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)

            # Support multiple value types
            if isinstance(value, list):
                if chunk_value not in value:
                    return False
            elif isinstance(value, dict):
                # Range filter: {'min': x, 'max': y}
                if 'min' in value and chunk_value < value['min']:
                    return False
                if 'max' in value and chunk_value > value['max']:
                    return False
            else:
                if chunk_value != value:
                    return False

        return True
```

---

## 5. Keyword Search Component

### 5.1 BM25 Implementation

```python
# File: src/retrieval/keyword_search.py

from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple, Optional
import re

class KeywordSearcher:
    """Keyword search using BM25Okapi algorithm"""

    def __init__(
        self,
        vector_store: MultiDocumentVectorStore,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 searcher

        Args:
            vector_store: Multi-document vector store
            k1: BM25 saturation parameter (1.2-2.0)
            b: BM25 length normalization (0.75 typical)
        """
        self.vector_store = vector_store
        self.k1 = k1
        self.b = b

        # BM25 indices per document
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        self.tokenized_docs: Dict[str, List[List[str]]] = {}
        self.chunk_ids: Dict[str, List[str]] = {}

        # Build indices
        self._build_indices()

    def _build_indices(self):
        """Build BM25 indices for all documents"""

        for doc_id, metadata_store in self.vector_store.metadata_stores.items():
            # Get all chunks
            chunks = list(metadata_store.values())

            # Tokenize
            tokenized_docs = [
                self._tokenize(chunk.content)
                for chunk in chunks
            ]

            # Build BM25 index
            self.bm25_indices[doc_id] = BM25Okapi(
                tokenized_docs,
                k1=self.k1,
                b=self.b
            )

            self.tokenized_docs[doc_id] = tokenized_docs
            self.chunk_ids[doc_id] = [chunk.chunk_id for chunk in chunks]

            logger.info(f"Built BM25 index for {doc_id}: {len(chunks)} chunks")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25

        Strategy:
        - Lowercase
        - Split on whitespace and punctuation
        - Keep legal references intact (§89)
        - Remove stop words
        """

        # Preserve legal references
        text = text.lower()

        # Replace § with placeholder to keep it
        text = text.replace('§', '_PARA_')

        # Split on whitespace and punctuation (but keep some)
        tokens = re.findall(r'\b\w+\b', text)

        # Restore § symbol
        tokens = [t.replace('_para_', '§') for t in tokens]

        # Remove stop words (Czech)
        stop_words = {
            'a', 'i', 'k', 'o', 's', 'u', 'v', 'z',
            'do', 'je', 'na', 'od', 'po', 'se', 'si', 've',
            'aby', 'ale', 'ani', 'být', 'kde', 'kdo', 'než', 'pod',
            'pro', 'při', 'tak', 'také', 'ten', 'než', 'nebo'
        }

        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

        return tokens

    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        BM25 keyword search

        Args:
            query: Search query
            document_ids: Filter by document IDs
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of search results sorted by BM25 score
        """

        # 1. Tokenize query
        query_tokens = self._tokenize(query)

        # 2. Search in selected indices
        indices_to_search = document_ids or list(self.bm25_indices.keys())

        all_results = []

        for doc_id in indices_to_search:
            results = await self._search_single_index(
                query_tokens,
                doc_id,
                top_k * 2,  # Over-retrieve for filtering
                filters
            )
            all_results.extend(results)

        # 3. Sort by score and take top-K
        all_results.sort(key=lambda x: x.keyword_score, reverse=True)

        return all_results[:top_k]

    async def _search_single_index(
        self,
        query_tokens: List[str],
        document_id: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Search in a single BM25 index"""

        # Get BM25 scores
        bm25_index = self.bm25_indices[document_id]
        scores = bm25_index.get_scores(query_tokens)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Map to chunks
        metadata_store = self.vector_store.metadata_stores[document_id]
        chunk_ids = self.chunk_ids[document_id]

        results = []

        for idx in top_indices:
            score = scores[idx]

            if score <= 0:
                continue

            chunk_id = chunk_ids[idx]
            chunk = metadata_store[chunk_id]

            # Apply filters
            if filters and not self._matches_filters(chunk, filters):
                continue

            result = SearchResult(
                chunk_id=chunk_id,
                chunk=chunk,
                document_id=document_id,
                score=float(score),
                keyword_score=float(score),
                retrieval_method="keyword"
            )
            results.append(result)

        return results

    def _matches_filters(
        self,
        chunk: LegalChunk,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if chunk matches metadata filters"""
        # Same as semantic search
        for key, value in filters.items():
            if chunk.metadata.get(key) != value:
                return False
        return True
```

### 5.2 BM25 Algorithm Details

**BM25 Formula**:

```
BM25(D, Q) = Σ IDF(qi) · (f(qi, D) · (k1 + 1)) / (f(qi, D) + k1 · (1 - b + b · |D|/avgdl))

where:
- D = document
- Q = query
- qi = term i in query
- f(qi, D) = frequency of qi in D
- |D| = length of document D
- avgdl = average document length
- k1 = 1.5 (saturation parameter)
- b = 0.75 (length normalization)
- IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5))
  - N = total number of documents
  - n(qi) = number of documents containing qi
```

**Example Calculation**:

```python
# Query: "odpovědnost dodavatele"
# Tokenized: ["odpovědnost", "dodavatele"]

# Document A: "Dodavatel odpovídá za vady. Odpovědnost dodavatele..."
# Tokenized: ["dodavatel", "odpovídá", "vady", "odpovědnost", "dodavatele", ...]
# f("odpovědnost", A) = 1
# f("dodavatele", A) = 1
# |A| = 15 tokens

# Document B: "Smlouva definuje práva objednatele..."
# f("odpovědnost", B) = 0
# f("dodavatele", B) = 0
# |B| = 20 tokens

# Corpus stats:
# N = 100 documents
# n("odpovědnost") = 30 documents contain this term
# n("dodavatele") = 50 documents contain this term
# avgdl = 18 tokens

# IDF("odpovědnost") = log((100 - 30 + 0.5) / (30 + 0.5)) ≈ 0.83
# IDF("dodavatele") = log((100 - 50 + 0.5) / (50 + 0.5)) ≈ 0.00

# For Document A:
# Term "odpovědnost":
#   score = 0.83 · (1 · 2.5) / (1 + 1.5 · (1 - 0.75 + 0.75 · 15/18))
#        = 0.83 · 2.5 / (1 + 1.5 · 0.875)
#        = 0.83 · 2.5 / 2.3125
#        ≈ 0.90

# Term "dodavatele":
#   score = 0.00 · ... ≈ 0.00

# BM25(A) = 0.90 + 0.00 = 0.90

# For Document B:
# BM25(B) = 0.00 (no query terms present)

# Result: Document A ranks higher
```

---

## 6. Structural Search Component

```python
# File: src/retrieval/structural_search.py

from typing import List, Dict, Any, Optional, Set
import re

class StructuralSearcher:
    """Structural search based on legal hierarchy and metadata"""

    def __init__(self, vector_store: MultiDocumentVectorStore):
        self.vector_store = vector_store
        self.reference_extractor = LegalReferenceExtractor()

    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 30,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Structural search based on:
        1. Legal references in query (§89 → find §89)
        2. Hierarchy filters (Část II, Hlava III)
        3. Content type filters (obligation, prohibition)
        4. Structural patterns

        Args:
            query: Search query
            document_ids: Filter by document IDs
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of search results sorted by structural relevance
        """

        # 1. Extract structural components from query
        structural_hints = self._extract_structural_hints(query)

        # 2. Build structural filters
        structural_filters = self._build_filters(structural_hints, filters)

        # 3. Search in selected documents
        indices_to_search = document_ids or list(self.vector_store.metadata_stores.keys())

        all_results = []

        for doc_id in indices_to_search:
            results = self._search_by_structure(
                doc_id,
                structural_filters,
                structural_hints
            )
            all_results.extend(results)

        # 4. Score and rank
        all_results = self._score_structural_matches(all_results, structural_hints)
        all_results.sort(key=lambda x: x.structural_score, reverse=True)

        return all_results[:top_k]

    def _extract_structural_hints(self, query: str) -> Dict[str, Any]:
        """
        Extract structural hints from query

        Examples:
        - "podle §89" → {references: ["§89"]}
        - "v Části II" → {part: "II"}
        - "povinnosti dodavatele" → {content_type: "obligation"}
        """

        hints = {
            'references': [],
            'part': None,
            'chapter': None,
            'content_types': []
        }

        # Extract legal references
        ref_patterns = [
            r'§\s*(\d+)(?:\s+odst\.\s*(\d+))?(?:\s+písm\.\s*([a-z]))?',
            r'[Čč]l(?:ánek|\.)\s*(\d+)(?:\.\s*(\d+))?'
        ]

        for pattern in ref_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                hints['references'].append(match.group(0))

        # Extract hierarchy hints
        if match := re.search(r'[Čč]ást\s+([IVX]+)', query):
            hints['part'] = match.group(1)

        if match := re.search(r'[Hh]lava\s+([IVX]+)', query):
            hints['chapter'] = match.group(1)

        # Detect content types
        content_indicators = {
            'obligation': ['povinnost', 'musí', 'je povinen'],
            'prohibition': ['zákaz', 'nesmí', 'zakázáno'],
            'definition': ['definice', 'se rozumí', 'znamená']
        }

        query_lower = query.lower()
        for content_type, indicators in content_indicators.items():
            if any(ind in query_lower for ind in indicators):
                hints['content_types'].append(content_type)

        return hints

    def _build_filters(
        self,
        structural_hints: Dict[str, Any],
        user_filters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine structural hints with user filters"""

        filters = user_filters or {}

        # Add structural hints as filters
        if structural_hints['part']:
            filters['part'] = structural_hints['part']

        if structural_hints['chapter']:
            filters['chapter'] = structural_hints['chapter']

        if structural_hints['content_types']:
            filters['content_type'] = structural_hints['content_types']

        return filters

    def _search_by_structure(
        self,
        document_id: str,
        filters: Dict[str, Any],
        structural_hints: Dict[str, Any]
    ) -> List[SearchResult]:
        """Search in a single document by structure"""

        metadata_store = self.vector_store.metadata_stores[document_id]
        results = []

        # Get all chunks
        for chunk_id, chunk in metadata_store.items():

            # Check if matches filters
            if not self._matches_structural_filters(chunk, filters):
                continue

            # Check if matches references
            if structural_hints['references']:
                if not self._matches_references(chunk, structural_hints['references']):
                    continue

            # Create result with placeholder score (will be computed later)
            result = SearchResult(
                chunk_id=chunk_id,
                chunk=chunk,
                document_id=document_id,
                score=0.0,
                structural_score=0.0,
                retrieval_method="structural"
            )
            results.append(result)

        return results

    def _matches_structural_filters(
        self,
        chunk: LegalChunk,
        filters: Dict[str, Any]
    ) -> bool:
        """Check if chunk matches structural filters"""

        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)

            if isinstance(value, list):
                # Multiple allowed values
                if chunk_value not in value:
                    return False
            else:
                # Single value
                if chunk_value != value:
                    return False

        return True

    def _matches_references(
        self,
        chunk: LegalChunk,
        query_references: List[str]
    ) -> bool:
        """
        Check if chunk matches any of the query references

        Matches:
        - Direct match: query "§89", chunk legal_ref "§89"
        - Subsection match: query "§89", chunk "§89 odst. 2"
        - Referenced: query "§89", chunk contains reference to §89
        """

        # Direct match
        chunk_ref = chunk.legal_reference
        for query_ref in query_references:
            # Normalize references for comparison
            normalized_query = self._normalize_reference(query_ref)
            normalized_chunk = self._normalize_reference(chunk_ref)

            # Exact match or parent match
            if normalized_chunk.startswith(normalized_query):
                return True

        # Check if chunk references any of the query refs
        chunk_refs = chunk.metadata.get('references_to', [])
        for chunk_ref in chunk_refs:
            normalized_chunk_ref = self._normalize_reference(chunk_ref)
            for query_ref in query_references:
                normalized_query = self._normalize_reference(query_ref)
                if normalized_chunk_ref.startswith(normalized_query):
                    return True

        return False

    def _normalize_reference(self, ref: str) -> str:
        """Normalize legal reference for comparison"""
        # Remove spaces, lowercase
        normalized = ref.lower().replace(' ', '')
        return normalized

    def _score_structural_matches(
        self,
        results: List[SearchResult],
        structural_hints: Dict[str, Any]
    ) -> List[SearchResult]:
        """
        Score structural matches based on:
        - Reference match quality (exact vs partial)
        - Hierarchy depth match
        - Content type match
        """

        for result in results:
            score = 0.0
            chunk = result.chunk

            # Reference match score
            if structural_hints['references']:
                ref_score = self._score_reference_match(
                    chunk,
                    structural_hints['references']
                )
                score += ref_score * 0.5

            # Hierarchy match score
            if structural_hints['part'] or structural_hints['chapter']:
                hier_score = self._score_hierarchy_match(
                    chunk,
                    structural_hints
                )
                score += hier_score * 0.3

            # Content type match score
            if structural_hints['content_types']:
                content_score = self._score_content_type_match(
                    chunk,
                    structural_hints['content_types']
                )
                score += content_score * 0.2

            result.structural_score = score
            result.score = score

        return results

    def _score_reference_match(
        self,
        chunk: LegalChunk,
        query_references: List[str]
    ) -> float:
        """
        Score reference match quality

        Scores:
        - 1.0: Exact match (query "§89", chunk "§89")
        - 0.8: Parent match (query "§89", chunk "§89 odst. 2")
        - 0.6: Referenced (chunk references §89)
        - 0.0: No match
        """

        chunk_ref = chunk.legal_reference.lower().replace(' ', '')

        for query_ref in query_references:
            query_ref_norm = query_ref.lower().replace(' ', '')

            # Exact match
            if chunk_ref == query_ref_norm:
                return 1.0

            # Parent match (chunk is more specific)
            if chunk_ref.startswith(query_ref_norm):
                return 0.8

            # Child match (query is more specific)
            if query_ref_norm.startswith(chunk_ref):
                return 0.7

        # Check references
        chunk_refs = chunk.metadata.get('references_to', [])
        for ref in chunk_refs:
            ref_norm = ref.lower().replace(' ', '')
            for query_ref in query_references:
                query_ref_norm = query_ref.lower().replace(' ', '')
                if ref_norm.startswith(query_ref_norm):
                    return 0.6

        return 0.0

    def _score_hierarchy_match(
        self,
        chunk: LegalChunk,
        structural_hints: Dict[str, Any]
    ) -> float:
        """Score hierarchy match"""

        score = 0.0
        matches = 0
        total = 0

        if structural_hints['part']:
            total += 1
            if chunk.metadata.get('part') == structural_hints['part']:
                matches += 1

        if structural_hints['chapter']:
            total += 1
            if chunk.metadata.get('chapter') == structural_hints['chapter']:
                matches += 1

        if total > 0:
            score = matches / total

        return score

    def _score_content_type_match(
        self,
        chunk: LegalChunk,
        content_types: List[str]
    ) -> float:
        """Score content type match"""

        chunk_type = chunk.metadata.get('content_type', 'general')

        if chunk_type in content_types:
            return 1.0

        return 0.0
```

---

## 7. Score Fusion & Deduplication

```python
# File: src/retrieval/hybrid_retriever.py

class HybridRetriever:
    """Combines semantic, keyword, and structural search"""

    def __init__(
        self,
        semantic_searcher: SemanticSearcher,
        keyword_searcher: KeywordSearcher,
        structural_searcher: StructuralSearcher,
        config: RetrievalConfig
    ):
        self.semantic = semantic_searcher
        self.keyword = keyword_searcher
        self.structural = structural_searcher
        self.config = config

        # Validate config
        self.config.validate()

    async def search(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        query_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Triple hybrid search with score fusion

        Args:
            query: Search query
            document_ids: Filter by document IDs
            top_k: Number of final results (default: config.top_k)
            filters: Metadata filters
            query_type: Query type (optional, for adaptive weighting)

        Returns:
            List of search results sorted by combined score
        """

        top_k = top_k or self.config.top_k
        candidate_k = int(top_k * self.config.candidate_multiplier)

        # 1. Adaptive weighting based on query type
        weights = self._get_adaptive_weights(query, query_type)

        # 2. Parallel retrieval
        if self.config.parallel_retrieval:
            semantic_results, keyword_results, structural_results = await asyncio.gather(
                self.semantic.search(query, document_ids, candidate_k, filters),
                self.keyword.search(query, document_ids, candidate_k, filters),
                self.structural.search(query, document_ids, candidate_k, filters)
            )
        else:
            # Sequential (for debugging)
            semantic_results = await self.semantic.search(query, document_ids, candidate_k, filters)
            keyword_results = await self.keyword.search(query, document_ids, candidate_k, filters)
            structural_results = await self.structural.search(query, document_ids, candidate_k, filters)

        # 3. Normalize scores
        if self.config.normalize_scores:
            semantic_results = self._normalize_scores(semantic_results, 'semantic_score')
            keyword_results = self._normalize_scores(keyword_results, 'keyword_score')
            structural_results = self._normalize_scores(structural_results, 'structural_score')

        # 4. Combine results with score fusion
        combined_results = self._fuse_scores(
            semantic_results,
            keyword_results,
            structural_results,
            weights
        )

        # 5. Filter by minimum score threshold
        if self.config.enable_score_threshold:
            combined_results = [
                r for r in combined_results
                if r.score >= self.config.min_score_threshold
            ]

        # 6. Sort and take top-K
        combined_results.sort(key=lambda x: x.score, reverse=True)

        return combined_results[:top_k]

    def _get_adaptive_weights(
        self,
        query: str,
        query_type: Optional[str]
    ) -> Dict[str, float]:
        """
        Adaptive weighting based on query characteristics

        Strategy:
        - Legal reference query → boost structural
        - Technical terminology → boost keyword
        - Conceptual query → boost semantic
        """

        # Default weights from config
        weights = {
            'semantic': self.config.semantic_weight,
            'keyword': self.config.keyword_weight,
            'structural': self.config.structural_weight
        }

        # Adaptive adjustments
        query_lower = query.lower()

        # Has legal references? Boost structural
        if re.search(r'§\s*\d+', query) or re.search(r'[Čč]l(?:ánek|\.)\s*\d+', query):
            weights['structural'] += 0.2
            weights['semantic'] -= 0.1
            weights['keyword'] -= 0.1

        # Short query with exact terms? Boost keyword
        if len(query.split()) <= 3:
            weights['keyword'] += 0.1
            weights['semantic'] -= 0.1

        # Long analytical query? Boost semantic
        if len(query.split()) > 15:
            weights['semantic'] += 0.1
            weights['keyword'] -= 0.1

        # Normalize to sum to 1.0
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def _normalize_scores(
        self,
        results: List[SearchResult],
        score_field: str
    ) -> List[SearchResult]:
        """
        Normalize scores to [0, 1] range

        Method: Min-max normalization
        normalized_score = (score - min) / (max - min)
        """

        if not results:
            return results

        # Get scores
        scores = [getattr(r, score_field) for r in results]

        if not scores:
            return results

        # Min-max normalization
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All same score
            for r in results:
                setattr(r, score_field, 1.0)
        else:
            for r in results:
                score = getattr(r, score_field)
                normalized = (score - min_score) / (max_score - min_score)
                setattr(r, score_field, normalized)

        return results

    def _fuse_scores(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        structural_results: List[SearchResult],
        weights: Dict[str, float]
    ) -> List[SearchResult]:
        """
        Fuse scores from three retrieval strategies

        Formula:
        combined_score = α·semantic + β·keyword + γ·structural

        Where α + β + γ = 1.0
        """

        # Build score dictionaries
        semantic_scores = {r.chunk_id: r.semantic_score for r in semantic_results if r.semantic_score is not None}
        keyword_scores = {r.chunk_id: r.keyword_score for r in keyword_results if r.keyword_score is not None}
        structural_scores = {r.chunk_id: r.structural_score for r in structural_results if r.structural_score is not None}

        # Get union of all chunk IDs
        all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys()) | set(structural_scores.keys())

        # Build chunk map
        chunk_map = {}
        for results in [semantic_results, keyword_results, structural_results]:
            for r in results:
                if r.chunk_id not in chunk_map:
                    chunk_map[r.chunk_id] = r

        # Combine scores
        combined_results = []

        for chunk_id in all_chunk_ids:
            sem_score = semantic_scores.get(chunk_id, 0.0)
            key_score = keyword_scores.get(chunk_id, 0.0)
            struct_score = structural_scores.get(chunk_id, 0.0)

            # Weighted combination
            combined_score = (
                weights['semantic'] * sem_score +
                weights['keyword'] * key_score +
                weights['structural'] * struct_score
            )

            # Get result object
            result = chunk_map[chunk_id]

            # Update scores
            result.semantic_score = sem_score if sem_score > 0 else None
            result.keyword_score = key_score if key_score > 0 else None
            result.structural_score = struct_score if struct_score > 0 else None
            result.score = combined_score
            result.retrieval_method = "hybrid"

            combined_results.append(result)

        return combined_results
```

---

## 8. Configuration

```yaml
# config.yaml
retrieval:
  # Triple hybrid weights (must sum to 1.0)
  semantic_weight: 0.5   # α - embeddings
  keyword_weight: 0.3    # β - BM25
  structural_weight: 0.2  # γ - hierarchy/refs

  # Top-K parameters
  top_k: 20
  candidate_multiplier: 1.5  # Retrieve 30 candidates per strategy

  # Score normalization
  normalize_scores: true
  normalization_method: min-max  # min-max | z-score

  # Filtering
  enable_metadata_filtering: true
  enable_score_threshold: true
  min_score_threshold: 0.1

  # Performance
  enable_caching: true
  parallel_retrieval: true

  # BM25 parameters
  bm25:
    k1: 1.5  # Saturation parameter
    b: 0.75  # Length normalization

  # Adaptive weighting
  adaptive_weights: true
  reference_boost: 0.2  # Boost structural when refs detected

  # Query expansion (optional)
  enable_query_expansion: false
  max_expansions: 3
```

---

## 9. Usage Examples

### 9.1 Basic Search

```python
# Initialize
retriever = HybridRetriever(
    semantic_searcher,
    keyword_searcher,
    structural_searcher,
    config
)

# Simple search
results = await retriever.search(
    query="Jaké jsou povinnosti dodavatele?",
    top_k=10
)

for rank, result in enumerate(results, 1):
    print(f"{rank}. [{result.score:.3f}] {result.chunk.legal_reference}")
    print(f"   Semantic: {result.semantic_score:.3f}")
    print(f"   Keyword: {result.keyword_score:.3f}")
    print(f"   Structural: {result.structural_score:.3f}")
```

### 9.2 Filtered Search

```python
# Search with filters
results = await retriever.search(
    query="odpovědnost za vady",
    document_ids=["contract_001"],  # Only in contract
    filters={
        'content_type': 'obligation',  # Only obligations
        'part': 'II'  # Only Part II
    },
    top_k=5
)
```

### 9.3 Reference-Based Search

```python
# Query with legal reference
results = await retriever.search(
    query="podle §89 občanského zákoníku",
    top_k=5
)

# Structural search will boost chunks matching §89
```

---

## 10. Testing

```python
# tests/test_hybrid_retrieval.py

import pytest

@pytest.fixture
def hybrid_retriever():
    # Setup retriever with mock data
    return HybridRetriever(...)

def test_semantic_only_search(hybrid_retriever):
    """Test pure semantic search"""
    config = RetrievalConfig(
        semantic_weight=1.0,
        keyword_weight=0.0,
        structural_weight=0.0
    )

    results = await hybrid_retriever.search(
        "odpovědnost dodavatele",
        top_k=5
    )

    assert len(results) <= 5
    assert all(r.semantic_score is not None for r in results)

def test_keyword_only_search(hybrid_retriever):
    """Test pure keyword search"""
    config = RetrievalConfig(
        semantic_weight=0.0,
        keyword_weight=1.0,
        structural_weight=0.0
    )

    results = await hybrid_retriever.search(
        "§89 odst. 2",
        top_k=5
    )

    assert len(results) <= 5

def test_triple_hybrid_fusion(hybrid_retriever):
    """Test score fusion from all three strategies"""
    results = await hybrid_retriever.search(
        "povinnosti dodavatele podle §89",
        top_k=10
    )

    # Should have results from multiple strategies
    has_semantic = any(r.semantic_score is not None for r in results)
    has_keyword = any(r.keyword_score is not None for r in results)
    has_structural = any(r.structural_score is not None for r in results)

    assert has_semantic or has_keyword or has_structural

def test_score_normalization(hybrid_retriever):
    """Test score normalization"""
    results = await hybrid_retriever.search("test query", top_k=10)

    # All scores should be in [0, 1]
    for r in results:
        assert 0.0 <= r.score <= 1.0

def test_deduplication(hybrid_retriever):
    """Test that results are deduplicated"""
    results = await hybrid_retriever.search("test query", top_k=20)

    # No duplicate chunk IDs
    chunk_ids = [r.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))

def test_metadata_filtering(hybrid_retriever):
    """Test metadata filtering"""
    results = await hybrid_retriever.search(
        "test query",
        filters={'content_type': 'obligation'},
        top_k=10
    )

    # All results should match filter
    assert all(
        r.chunk.metadata.get('content_type') == 'obligation'
        for r in results
    )

def test_adaptive_weighting(hybrid_retriever):
    """Test adaptive weight adjustment"""
    # Query with legal reference should boost structural
    weights_with_ref = hybrid_retriever._get_adaptive_weights("podle §89", None)
    weights_without_ref = hybrid_retriever._get_adaptive_weights("odpovědnost", None)

    assert weights_with_ref['structural'] > weights_without_ref['structural']
```

---

## 11. Performance Optimization

### 11.1 Caching

```python
from functools import lru_cache
from hashlib import md5

class CachedHybridRetriever(HybridRetriever):
    """Hybrid retriever with result caching"""

    def __init__(self, *args, cache_size: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self._cache = {}

    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        # Generate cache key
        cache_key = self._generate_cache_key(query, kwargs)

        # Check cache
        if cache_key in self._cache:
            logger.debug(f"Cache hit for query: {query}")
            return self._cache[cache_key]

        # Perform search
        results = await super().search(query, **kwargs)

        # Store in cache (with LRU eviction)
        self._cache[cache_key] = results
        if len(self._cache) > self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return results

    def _generate_cache_key(self, query: str, params: Dict) -> str:
        """Generate unique cache key from query and parameters"""
        key_str = f"{query}_{json.dumps(params, sort_keys=True)}"
        return md5(key_str.encode()).hexdigest()
```

### 11.2 Batch Retrieval

```python
async def batch_search(
    retriever: HybridRetriever,
    queries: List[str],
    **kwargs
) -> List[List[SearchResult]]:
    """Search multiple queries in parallel"""

    tasks = [
        retriever.search(query, **kwargs)
        for query in queries
    ]

    results = await asyncio.gather(*tasks)
    return results
```

---

## 12. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Single query (10k chunks) | <200ms | All three strategies |
| Batch (10 queries) | <500ms | Parallel execution |
| Cache hit | <5ms | In-memory cache |
| Score fusion | <10ms | For 100 candidates |

---

## 13. Future Enhancements

### 13.1 Query Expansion

```python
class QueryExpander:
    """Expand query with synonyms and related terms"""

    async def expand(self, query: str) -> List[str]:
        """
        Generate query variations

        Example:
        "odpovědnost" → ["odpovědnost", "liability", "ručení"]
        """
        pass
```

### 13.2 Learning to Rank

```python
class LearnedRanker:
    """Machine learning model for result ranking"""

    def train(self, queries: List[str], relevance_labels: List[List[int]]):
        """Train ranking model from labeled data"""
        pass

    def rerank(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using learned model"""
        pass
```

---

## 14. Monitoring & Debugging

```python
# Enable detailed logging
import logging
logging.getLogger('retrieval').setLevel(logging.DEBUG)

# Retrieve with debug info
results = await retriever.search(
    query="test",
    top_k=5
)

# Inspect score breakdown
for r in results:
    breakdown = r.get_score_breakdown()
    print(f"Chunk {r.chunk_id}:")
    print(f"  Semantic: {breakdown['semantic']:.3f}")
    print(f"  Keyword:  {breakdown['keyword']:.3f}")
    print(f"  Structural: {breakdown['structural']:.3f}")
    print(f"  Combined: {breakdown['combined']:.3f}")
```

---

## 15. Error Handling

```python
class RetrievalError(Exception):
    """Base exception for retrieval errors"""
    pass

class InvalidWeightsError(RetrievalError):
    """Weights don't sum to 1.0"""
    pass

class EmptyResultsError(RetrievalError):
    """No results found"""
    pass

# Usage
try:
    results = await retriever.search(query)
except InvalidWeightsError:
    logger.error("Fix retrieval weights in config")
except EmptyResultsError:
    logger.warning(f"No results for query: {query}")
    results = []
```

---

**Specification Complete**: 05_hybrid_retrieval.md
**Pages**: ~20
**Status**: Ready for implementation
