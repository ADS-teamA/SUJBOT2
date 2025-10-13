"""
Hybrid Retrieval System - Triple Hybrid Search

Combines semantic, keyword, and structural search for optimal legal document retrieval.
Based on specification: specs/05_hybrid_retrieval.md

Author: SUJBOT2 Team
Date: 2025-10-08
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Set, Tuple
from hashlib import md5
import json

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LegalChunk:
    """Legal chunk data structure (from 03_chunking_strategy.md)"""

    chunk_id: str
    content: str
    document_type: str  # 'law_code' | 'contract' | 'regulation'
    hierarchy_path: str
    legal_reference: str  # e.g., "§89", "Článek 5"
    structural_level: str  # 'paragraph' | 'subsection' | 'article' | etc.
    metadata: Dict[str, Any]


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

    # BM25 parameters
    bm25_k1: float = 1.5  # Saturation parameter
    bm25_b: float = 0.75  # Length normalization

    # Adaptive weighting
    adaptive_weights: bool = True
    reference_boost: float = 0.2  # Boost structural when refs detected

    # Query expansion
    enable_query_expansion: bool = False
    max_expansions: int = 3

    def validate(self):
        """Validate configuration"""
        total_weight = self.semantic_weight + self.keyword_weight + self.structural_weight
        assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"


# ============================================================================
# Import Real Components (previously mock dependencies)
# ============================================================================

# Import real implementations from other modules
try:
    from .indexing import MultiDocumentVectorStore
    from .embeddings import LegalEmbedder
except ImportError:
    # Support both package and direct imports
    from indexing import MultiDocumentVectorStore
    from embeddings import LegalEmbedder


class LegalReferenceExtractor:
    """Mock legal reference extractor"""

    def extract(self, text: str) -> List[str]:
        """Extract legal references from text"""
        references = []

        # Extract § references
        pattern = r'§\s*\d+(?:\s+odst\.\s*\d+)?(?:\s+písm\.\s*[a-z])?'
        references.extend(re.findall(pattern, text))

        # Extract článek references
        pattern = r'[Čč]l(?:ánek|\.)\s*\d+(?:\.\s*\d+)?'
        references.extend(re.findall(pattern, text))

        return references


# ============================================================================
# Semantic Search Component
# ============================================================================

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
            if indices_to_search:
                all_results = await self._search_single_index(
                    query_embedding,
                    indices_to_search[0],
                    top_k,
                    filters
                )

        # 4. Sort by score and take top-K
        all_results.sort(key=lambda x: x.semantic_score or 0, reverse=True)

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
        if document_id not in self.vector_store.indices:
            logger.warning(f"Document {document_id} not found in vector store")
            return []

        index = self.vector_store.indices[document_id]
        metadata_store = self.vector_store.metadata_stores[document_id]

        # FAISS search
        query_vector = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = index.search(query_vector, min(top_k * 2, index.ntotal))  # Over-retrieve for filtering

        # Map to chunks
        results = []
        chunk_ids = list(metadata_store.keys())

        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(chunk_ids) or idx < 0:
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
                if 'min' in value and chunk_value is not None and chunk_value < value['min']:
                    return False
                if 'max' in value and chunk_value is not None and chunk_value > value['max']:
                    return False
            else:
                if chunk_value != value:
                    return False

        return True


# ============================================================================
# Keyword Search Component (BM25)
# ============================================================================

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
            if tokenized_docs:  # Only build if there are docs
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
            if doc_id not in self.bm25_indices:
                continue

            results = await self._search_single_index(
                query_tokens,
                doc_id,
                top_k * 2,  # Over-retrieve for filtering
                filters
            )
            all_results.extend(results)

        # 3. Sort by score and take top-K
        all_results.sort(key=lambda x: x.keyword_score or 0, reverse=True)

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
        for key, value in filters.items():
            chunk_value = chunk.metadata.get(key)
            if isinstance(value, list):
                if chunk_value not in value:
                    return False
            else:
                if chunk_value != value:
                    return False
        return True


# ============================================================================
# Structural Search Component
# ============================================================================

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
        all_results.sort(key=lambda x: x.structural_score or 0, reverse=True)

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

        filters = user_filters.copy() if user_filters else {}

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

        if document_id not in self.vector_store.metadata_stores:
            return []

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

            # Default score if no hints
            if score == 0.0:
                score = 0.5  # Default structural relevance

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


# ============================================================================
# Hybrid Retriever (Main Component)
# ============================================================================

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

        logger.info(f"Hybrid search with weights: {weights}")

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

        logger.info(f"Retrieved: {len(semantic_results)} semantic, {len(keyword_results)} keyword, {len(structural_results)} structural")

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

        # 7. Set ranks
        for rank, result in enumerate(combined_results[:top_k], 1):
            result.rank = rank

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

        if not self.config.adaptive_weights:
            return weights

        # Adaptive adjustments
        query_lower = query.lower()

        # Has legal references? Boost structural
        if re.search(r'§\s*\d+', query) or re.search(r'[Čč]l(?:ánek|\.)\s*\d+', query):
            weights['structural'] += self.config.reference_boost
            weights['semantic'] -= self.config.reference_boost / 2
            weights['keyword'] -= self.config.reference_boost / 2

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
        if total > 0:
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
        scores = [getattr(r, score_field) for r in results if getattr(r, score_field) is not None]

        if not scores:
            return results

        # Min-max normalization
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All same score
            for r in results:
                if getattr(r, score_field) is not None:
                    setattr(r, score_field, 1.0)
        else:
            for r in results:
                score = getattr(r, score_field)
                if score is not None:
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

        # Get union of all chunk IDs (deduplication)
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
            sem_score = semantic_scores.get(chunk_id, None)
            key_score = keyword_scores.get(chunk_id, None)
            struct_score = structural_scores.get(chunk_id, None)

            # Count active strategies and normalize weights
            # This ensures chunks found by only one strategy don't lose score
            active_weights = {}
            if sem_score is not None:
                active_weights['semantic'] = weights['semantic']
            if key_score is not None:
                active_weights['keyword'] = weights['keyword']
            if struct_score is not None:
                active_weights['structural'] = weights['structural']

            # Normalize weights by active strategies only
            if active_weights:
                total_active_weight = sum(active_weights.values())
                normalized_weights = {k: v / total_active_weight for k, v in active_weights.items()}

                # Weighted combination with normalized weights
                combined_score = 0.0
                if sem_score is not None:
                    combined_score += normalized_weights.get('semantic', 0.0) * sem_score
                if key_score is not None:
                    combined_score += normalized_weights.get('keyword', 0.0) * key_score
                if struct_score is not None:
                    combined_score += normalized_weights.get('structural', 0.0) * struct_score
            else:
                combined_score = 0.0

            # Get result object
            result = chunk_map[chunk_id]

            # Update scores (already None if strategy didn't find the chunk)
            result.semantic_score = sem_score
            result.keyword_score = key_score
            result.structural_score = struct_score
            result.score = combined_score
            result.retrieval_method = "hybrid"

            combined_results.append(result)

        return combined_results


# ============================================================================
# Query Expansion (Optional Enhancement)
# ============================================================================

class QueryExpander:
    """Expand query with synonyms and related terms"""

    def __init__(self):
        # Czech legal synonyms dictionary
        self.synonyms = {
            'odpovědnost': ['liability', 'ručení', 'zodpovědnost'],
            'smlouva': ['contract', 'dohoda', 'ujednání'],
            'dodavatel': ['supplier', 'poskytovatel', 'subdodavatel'],
            'objednatel': ['customer', 'zadavatel', 'klient'],
            'vada': ['defect', 'nedostatek', 'závada'],
            'záruční': ['warranty', 'garancí'],
            'povinnost': ['obligation', 'závazek'],
            'právo': ['right', 'nárok'],
        }

    async def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Generate query variations

        Args:
            query: Original query
            max_expansions: Maximum number of expansions

        Returns:
            List of query variations (including original)
        """

        variations = [query]  # Always include original
        query_lower = query.lower()

        # Find synonyms in query
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Create variations with synonyms
                for synonym in synonyms[:max_expansions]:
                    variation = query_lower.replace(term, synonym)
                    variations.append(variation)

                    if len(variations) >= max_expansions + 1:
                        break

        return variations[:max_expansions + 1]


# ============================================================================
# Caching (Optional Performance Enhancement)
# ============================================================================

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
        # Remove non-hashable items
        hashable_params = {
            k: v for k, v in params.items()
            if not isinstance(v, (list, dict)) or k in ['document_ids']
        }
        key_str = f"{query}_{json.dumps(hashable_params, sort_keys=True)}"
        return md5(key_str.encode()).hexdigest()


# ============================================================================
# Factory Functions
# ============================================================================

def create_hybrid_retriever(
    vector_store: MultiDocumentVectorStore,
    embedder: LegalEmbedder,
    config: Optional[RetrievalConfig] = None
) -> HybridRetriever:
    """
    Factory function to create a configured HybridRetriever

    Args:
        vector_store: Multi-document vector store
        embedder: Legal embedder for semantic search
        config: Retrieval configuration (uses defaults if None)

    Returns:
        Configured HybridRetriever instance
    """

    if config is None:
        config = RetrievalConfig()

    # Create component searchers
    semantic_searcher = SemanticSearcher(embedder, vector_store)
    keyword_searcher = KeywordSearcher(
        vector_store,
        k1=config.bm25_k1,
        b=config.bm25_b
    )
    structural_searcher = StructuralSearcher(vector_store)

    # Create hybrid retriever
    retriever = HybridRetriever(
        semantic_searcher,
        keyword_searcher,
        structural_searcher,
        config
    )

    return retriever


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create mock components
    vector_store = MultiDocumentVectorStore()
    embedder = LegalEmbedder()

    # Create retriever
    config = RetrievalConfig(
        semantic_weight=0.5,
        keyword_weight=0.3,
        structural_weight=0.2,
        top_k=10
    )

    retriever = create_hybrid_retriever(vector_store, embedder, config)

    print("Hybrid Retriever created successfully!")
    print(f"Configuration: {config}")
