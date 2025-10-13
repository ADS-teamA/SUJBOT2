"""
Cross-Document Retrieval System
Enables retrieval and comparison across multiple documents (contract ↔ law)
to identify relationships, references, and potential conflicts.

This implements a three-tier matching strategy:
1. Explicit reference matching - Direct citations (contract mentions "§89")
2. Implicit semantic matching - Similar provisions without explicit references
3. Structural pattern matching - Corresponding sections (warranties, obligations)
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class MatchType(Enum):
    """Type of cross-document match"""
    EXPLICIT_REFERENCE = "explicit_reference"  # Direct citation
    SEMANTIC_SIMILAR = "semantic_similar"      # Semantically similar
    STRUCTURAL_PATTERN = "structural_pattern"  # Same position in structure
    TOPIC_RELATED = "topic_related"            # Related topic


class RelationType(Enum):
    """Relationship between documents"""
    COMPLIES = "complies"           # Contract complies with law
    CONFLICTS = "conflicts"         # Direct conflict
    DEVIATES = "deviates"          # Differs but might be acceptable
    MISSING = "missing"            # Required provision missing
    REFERENCES = "references"       # One references the other
    IMPLEMENTS = "implements"       # Contract implements law requirement


@dataclass
class LegalChunk:
    """A chunk of legal document optimized for retrieval"""

    # Identity
    chunk_id: str
    chunk_index: Optional[int] = None

    # Content
    content: str = ""
    title: Optional[str] = None

    # Document context
    document_id: str = ""
    document_type: str = ""  # 'law_code' | 'contract' | 'regulation'

    # Legal structure
    hierarchy_path: str = ""  # "Část II > Hlava III > §89"
    legal_reference: str = ""  # "§89" or "Článek 5.2"
    structural_level: str = ""  # 'paragraph' | 'article' | 'subsection'

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_citation(self) -> str:
        """Get properly formatted citation"""
        if self.document_type == 'law_code':
            law_name = self.metadata.get('law_citation', '')
            return f"{law_name}, {self.legal_reference}"
        else:
            return f"{self.hierarchy_path}"


@dataclass
class DocumentPair:
    """Pair of related chunks from different documents"""

    # Source (e.g., contract clause)
    source_chunk: LegalChunk
    source_document_id: str

    # Target (e.g., law provision)
    target_chunk: LegalChunk
    target_document_id: str

    # Relationship
    match_type: MatchType
    relation_type: Optional[RelationType] = None

    # Scores
    overall_score: float = 0.0
    explicit_score: float = 0.0
    semantic_score: float = 0.0
    structural_score: float = 0.0

    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    confidence: float = 0.0
    explanation: str = ""


@dataclass
class CrossDocumentResults:
    """Results of cross-document retrieval"""

    # Query context
    query: str
    source_document_type: str  # 'contract' | 'law_code'
    target_document_type: str

    # Matched pairs
    pairs: List[DocumentPair]

    # Matrices
    similarity_matrix: Optional[np.ndarray] = None  # NxM matrix

    # Statistics
    total_pairs: int = 0
    explicit_matches: int = 0
    semantic_matches: int = 0
    structural_matches: int = 0

    # Metadata
    retrieval_time_ms: float = 0.0

    def get_top_pairs(self, k: int = 10) -> List[DocumentPair]:
        """Get top-K pairs by overall score"""
        sorted_pairs = sorted(self.pairs, key=lambda p: p.overall_score, reverse=True)
        return sorted_pairs[:k]

    def get_pairs_by_type(self, match_type: MatchType) -> List[DocumentPair]:
        """Filter pairs by match type"""
        return [p for p in self.pairs if p.match_type == match_type]

    def get_conflicts(self) -> List[DocumentPair]:
        """Get pairs with CONFLICTS relation"""
        return [p for p in self.pairs if p.relation_type == RelationType.CONFLICTS]


# ============================================================================
# Reference Map (for explicit reference lookups)
# ============================================================================


class ReferenceMap:
    """
    Maintains mapping from legal references to chunks
    Example: "§89 odst. 2" -> [chunk_id_1, chunk_id_2, ...]
    """

    def __init__(self):
        self.reference_to_chunks: Dict[str, List[str]] = {}
        self.chunk_to_references: Dict[str, List[str]] = {}
        self.chunks_cache: Dict[str, LegalChunk] = {}

    def add_chunk(self, chunk: LegalChunk):
        """Add chunk to reference map"""
        if not chunk.legal_reference:
            return

        # Normalize reference
        normalized_ref = self._normalize_reference(chunk.legal_reference)

        # Add to mapping
        if normalized_ref not in self.reference_to_chunks:
            self.reference_to_chunks[normalized_ref] = []

        if chunk.chunk_id not in self.reference_to_chunks[normalized_ref]:
            self.reference_to_chunks[normalized_ref].append(chunk.chunk_id)

        # Cache chunk
        self.chunks_cache[chunk.chunk_id] = chunk

        # Reverse mapping
        if chunk.chunk_id not in self.chunk_to_references:
            self.chunk_to_references[chunk.chunk_id] = []
        if normalized_ref not in self.chunk_to_references[chunk.chunk_id]:
            self.chunk_to_references[chunk.chunk_id].append(normalized_ref)

    def get_chunks_by_reference(self, reference: str) -> List[str]:
        """Get chunk IDs that match a reference"""
        normalized = self._normalize_reference(reference)
        return self.reference_to_chunks.get(normalized, [])

    def get_chunk(self, chunk_id: str) -> Optional[LegalChunk]:
        """Get cached chunk by ID"""
        return self.chunks_cache.get(chunk_id)

    def _normalize_reference(self, reference: str) -> str:
        """Normalize legal reference for matching"""
        # Remove extra whitespace
        normalized = ' '.join(reference.split())
        # Standardize paragraph symbol
        normalized = normalized.replace('§', '§').strip()
        return normalized

    def save(self, path: Path):
        """Save reference map to disk"""
        data = {
            'reference_to_chunks': self.reference_to_chunks,
            'chunk_to_references': self.chunk_to_references,
            'chunks_cache': {
                cid: {
                    'chunk_id': c.chunk_id,
                    'content': c.content,
                    'title': getattr(c, 'title', None),
                    'document_id': c.document_id,
                    'document_type': c.document_type,
                    'hierarchy_path': c.hierarchy_path,
                    'legal_reference': c.legal_reference,
                    'structural_level': c.structural_level,
                    'metadata': c.metadata,
                    'chunk_index': getattr(c, 'chunk_index', None)
                }
                for cid, c in self.chunks_cache.items()
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ReferenceMap':
        """Load reference map from disk"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ref_map = cls()
        ref_map.reference_to_chunks = data['reference_to_chunks']
        ref_map.chunk_to_references = data['chunk_to_references']

        # Reconstruct chunks
        for cid, chunk_data in data['chunks_cache'].items():
            ref_map.chunks_cache[cid] = LegalChunk(**chunk_data)

        return ref_map


# ============================================================================
# Explicit Reference Matcher
# ============================================================================


class ExplicitReferenceMatcher:
    """Match documents based on explicit legal references"""

    def __init__(self, reference_map: ReferenceMap):
        self.reference_map = reference_map

        # Czech legal reference patterns
        self.patterns = {
            'paragraph': r'§\s*(\d+[a-z]?)(?:\s+odst\.\s*(\d+))?(?:\s+písm\.\s*([a-z]))?',
            'paragraph_alt': r'paragrafu?\s+(\d+)',
            'article': r'[Čč]l(?:ánek|\.)\s*(\d+)(?:\.\s*(\d+))?',
            'law_citation': r'[Zz]ákon(?:a|u)?\s+č\.\s*(\d+)/(\d+)\s*Sb\.',
            'part': r'[Čč]ást(?:i)?\s+([IVX]+)',
            'chapter': r'[Hh]lav[aěy]\s+([IVX]+)',
            'contextual': r'(?:podle|dle|v\s+souladu\s+s|na\s+základě)\s+(§\s*\d+|[Čč]l(?:ánek|\.)\s*\d+)'
        }

    async def find_explicit_matches(
        self,
        source_chunk: LegalChunk,
        target_document_id: str
    ) -> List[DocumentPair]:
        """
        Find explicit references from source chunk to target document

        Example:
        Contract chunk contains "podle §89 odst. 2"
        → Find §89 odst. 2 in law document

        Args:
            source_chunk: Source chunk (e.g., contract clause)
            target_document_id: Target document to search (e.g., law)

        Returns:
            List of document pairs with explicit references
        """

        # 1. Extract all references from source chunk
        references = self._extract_references(source_chunk.content)

        if not references:
            return []

        # 2. For each reference, find target chunk
        pairs = []

        for ref in references:
            # Lookup in reference map
            target_chunk_ids = self.reference_map.get_chunks_by_reference(
                ref['normalized_ref']
            )

            # Filter by target document
            target_chunk_ids = [
                chunk_id for chunk_id in target_chunk_ids
                if self._get_document_id(chunk_id) == target_document_id
            ]

            # Create pairs
            for chunk_id in target_chunk_ids:
                target_chunk = self.reference_map.get_chunk(chunk_id)

                if target_chunk is None:
                    continue

                pair = DocumentPair(
                    source_chunk=source_chunk,
                    source_document_id=source_chunk.document_id,
                    target_chunk=target_chunk,
                    target_document_id=target_document_id,
                    match_type=MatchType.EXPLICIT_REFERENCE,
                    relation_type=RelationType.REFERENCES,
                    explicit_score=1.0,  # Perfect match
                    overall_score=1.0,
                    evidence={
                        'reference_text': ref['original_text'],
                        'reference_type': ref['type'],
                        'match_method': 'direct_lookup',
                        'context': ref.get('context', '')
                    },
                    confidence=1.0,
                    explanation=f"Explicit reference: {ref['original_text']}"
                )
                pairs.append(pair)

        return pairs

    def _extract_references(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all legal references from text

        Returns:
        [
            {
                'original_text': '§89 odst. 2',
                'normalized_ref': '§89 odst. 2',
                'type': 'paragraph',
                'components': {'paragraph': 89, 'subsection': 2},
                'start': 15,
                'end': 27
            },
            ...
        ]
        """

        references = []

        for ref_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                ref = self._parse_reference_match(match, ref_type, text)
                if ref:
                    references.append(ref)

        # Deduplicate overlapping matches
        references = self._deduplicate_references(references)

        return references

    def _parse_reference_match(
        self,
        match: re.Match,
        ref_type: str,
        text: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a single reference match"""

        start, end = match.span()

        # Extract context (surrounding text)
        context_start = max(0, start - 30)
        context_end = min(len(text), end + 30)
        context = text[context_start:context_end]

        if ref_type == 'paragraph':
            para, subsec, letter = match.groups()
            normalized_ref = f"§{para}"
            if subsec:
                normalized_ref += f" odst. {subsec}"
            if letter:
                normalized_ref += f" písm. {letter}"

            return {
                'original_text': match.group(0),
                'normalized_ref': normalized_ref,
                'type': ref_type,
                'components': {
                    'paragraph': int(re.sub(r'[a-z]', '', para)),
                    'subsection': int(subsec) if subsec else None,
                    'letter': letter
                },
                'start': start,
                'end': end,
                'context': context
            }

        elif ref_type == 'article':
            article, point = match.groups()
            normalized_ref = f"Článek {article}"
            if point:
                normalized_ref += f".{point}"

            return {
                'original_text': match.group(0),
                'normalized_ref': normalized_ref,
                'type': ref_type,
                'components': {
                    'article': int(article),
                    'point': point
                },
                'start': start,
                'end': end,
                'context': context
            }

        elif ref_type == 'law_citation':
            number, year = match.groups()
            normalized_ref = f"Zákon č. {number}/{year} Sb."

            return {
                'original_text': match.group(0),
                'normalized_ref': normalized_ref,
                'type': ref_type,
                'components': {
                    'law_number': number,
                    'year': year
                },
                'start': start,
                'end': end,
                'context': context
            }

        # For other types, return basic info
        return {
            'original_text': match.group(0),
            'normalized_ref': match.group(0),
            'type': ref_type,
            'components': {},
            'start': start,
            'end': end,
            'context': context
        }

    def _deduplicate_references(
        self,
        references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove overlapping references

        Example:
        - "podle §89" (contextual)
        - "§89" (paragraph)
        → Keep only the more specific one
        """

        if not references:
            return []

        # Sort by start position
        references.sort(key=lambda r: r['start'])

        deduplicated = []
        prev_ref = None

        for ref in references:
            if prev_ref is None:
                deduplicated.append(ref)
                prev_ref = ref
                continue

            # Check overlap
            if ref['start'] < prev_ref['end']:
                # Overlapping - keep the more specific one
                if len(ref['normalized_ref']) > len(prev_ref['normalized_ref']):
                    # New reference is more specific
                    deduplicated[-1] = ref
                    prev_ref = ref
            else:
                # No overlap
                deduplicated.append(ref)
                prev_ref = ref

        return deduplicated

    def _get_document_id(self, chunk_id: str) -> str:
        """Extract document ID from chunk"""
        chunk = self.reference_map.get_chunk(chunk_id)
        return chunk.document_id if chunk else ""


# ============================================================================
# Semantic Matcher
# ============================================================================


class SemanticMatcher:
    """Find semantically similar provisions across documents"""

    def __init__(
        self,
        embedder: Any,  # Embedding model
        vector_store: Any  # Multi-document vector store
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self._embedding_cache: Dict[str, np.ndarray] = {}

    async def find_semantic_matches(
        self,
        source_chunk: LegalChunk,
        target_document_id: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[DocumentPair]:
        """
        Find semantically similar chunks in target document

        Strategy:
        1. Use source chunk's content as query
        2. Search target document's vector index
        3. Filter by minimum similarity
        4. Create document pairs

        Args:
            source_chunk: Source chunk to match
            target_document_id: Target document to search
            top_k: Number of matches to return
            min_similarity: Minimum cosine similarity (0-1)

        Returns:
            List of document pairs sorted by similarity
        """

        # 1. Get source chunk embedding
        source_embedding = await self._get_chunk_embedding(source_chunk)

        # 2. Search in target index (assume vector_store has document-specific search)
        # This is a simplified implementation - actual implementation would depend
        # on the specific vector store structure

        try:
            # Get target chunks (placeholder - actual implementation needs vector store API)
            results = await self._search_vector_store(
                source_embedding,
                target_document_id,
                top_k * 2
            )
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

        # 3. Create pairs
        pairs = []

        for result in results:
            score = result.get('score', 0.0)

            # Filter by minimum similarity
            if score < min_similarity:
                continue

            target_chunk = result.get('chunk')
            if not target_chunk:
                continue

            pair = DocumentPair(
                source_chunk=source_chunk,
                source_document_id=source_chunk.document_id,
                target_chunk=target_chunk,
                target_document_id=target_document_id,
                match_type=MatchType.SEMANTIC_SIMILAR,
                semantic_score=float(score),
                overall_score=float(score),
                evidence={
                    'cosine_similarity': float(score),
                    'match_method': 'vector_similarity'
                },
                confidence=float(score),
                explanation=f"Semantic similarity: {score:.3f}"
            )
            pairs.append(pair)

            if len(pairs) >= top_k:
                break

        return pairs

    async def _get_chunk_embedding(self, chunk: LegalChunk) -> np.ndarray:
        """Get embedding for a chunk (from cache or compute)"""

        # Check cache
        if chunk.chunk_id in self._embedding_cache:
            return self._embedding_cache[chunk.chunk_id]

        # Compute embedding
        if hasattr(self.embedder, 'encode'):
            # SentenceTransformer-like API
            embedding = await asyncio.to_thread(
                self.embedder.encode,
                [chunk.content],
                show_progress_bar=False
            )
            embedding = embedding[0]
        else:
            # Fallback - assume embedder has embed method
            embeddings = await self.embedder.embed_chunks([chunk])
            embedding = embeddings[0] if embeddings else np.zeros(384)

        # Cache
        self._embedding_cache[chunk.chunk_id] = embedding

        return embedding

    async def _search_vector_store(
        self,
        query_embedding: np.ndarray,
        target_document_id: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search vector store for similar documents
        This is a placeholder - actual implementation depends on vector store structure
        """
        # Placeholder implementation
        # Real implementation would interact with FAISS/Qdrant/etc.
        return []


# ============================================================================
# Structural Matcher
# ============================================================================


class StructuralMatcher:
    """Match chunks based on structural patterns and positions"""

    def __init__(self, vector_store: Any):
        self.vector_store = vector_store

        # Structural pattern mappings
        self.pattern_mappings = {
            # Contract section → Law section
            'warranties': ['záruky', 'záruka', 'warranty', 'záruční'],
            'liability': ['odpovědnost', 'liability', 'odpovídá'],
            'payment': ['platba', 'úhrada', 'payment', 'platební'],
            'termination': ['ukončení', 'vypovězení', 'termination'],
            'penalties': ['sankce', 'pokuta', 'penalty', 'sankční'],
            'obligations': ['povinnost', 'obligation', 'povinnosti'],
            'rights': ['právo', 'práva', 'rights'],
            'delivery': ['dodání', 'dodávka', 'delivery'],
            'acceptance': ['přejímka', 'převzetí', 'acceptance'],
        }

    async def find_structural_matches(
        self,
        source_chunk: LegalChunk,
        target_document_id: str,
        top_k: int = 5
    ) -> List[DocumentPair]:
        """
        Find structurally corresponding chunks

        Strategies:
        1. Topic matching: "Warranties" section → law provisions about warranties
        2. Position matching: Article N → similar position in law
        3. Content type matching: obligations → obligations

        Args:
            source_chunk: Source chunk
            target_document_id: Target document
            top_k: Number of matches

        Returns:
            List of document pairs
        """

        matches = []

        # Strategy 1: Topic-based matching
        topic_matches = await self._match_by_topic(
            source_chunk,
            target_document_id
        )
        matches.extend(topic_matches)

        # Strategy 2: Content type matching
        content_type_matches = await self._match_by_content_type(
            source_chunk,
            target_document_id
        )
        matches.extend(content_type_matches)

        # Strategy 3: Hierarchical position matching
        position_matches = await self._match_by_position(
            source_chunk,
            target_document_id
        )
        matches.extend(position_matches)

        # Deduplicate and score
        matches = self._deduplicate_structural_matches(matches)
        matches.sort(key=lambda p: p.structural_score, reverse=True)

        return matches[:top_k]

    async def _match_by_topic(
        self,
        source_chunk: LegalChunk,
        target_document_id: str
    ) -> List[DocumentPair]:
        """Match based on topic/subject matter"""

        # Extract topic from source
        source_topic = self._extract_topic(source_chunk)

        if not source_topic:
            return []

        # Placeholder - would need to iterate through target chunks
        # Real implementation needs access to all chunks in target document
        pairs = []

        return pairs

    async def _match_by_content_type(
        self,
        source_chunk: LegalChunk,
        target_document_id: str
    ) -> List[DocumentPair]:
        """Match based on content type (obligation, prohibition, etc.)"""

        source_type = source_chunk.metadata.get('content_type', 'general')

        if source_type == 'general':
            return []

        # Placeholder - would filter target chunks by content type
        pairs = []

        return pairs

    async def _match_by_position(
        self,
        source_chunk: LegalChunk,
        target_document_id: str
    ) -> List[DocumentPair]:
        """
        Match based on relative position in document

        Heuristic: Early sections often correspond
        (e.g., Článek 1 often relates to § at beginning)
        """

        # Get source position
        source_index = source_chunk.metadata.get('chunk_index', 0)
        source_total = source_chunk.metadata.get('total_chunks', 1)

        if source_total == 0:
            return []

        source_position = source_index / source_total  # 0.0 - 1.0

        # Placeholder - would need to iterate through target chunks
        pairs = []

        return pairs

    def _extract_topic(self, chunk: LegalChunk) -> Optional[str]:
        """
        Extract topic/subject from chunk

        Strategies:
        1. Use article/paragraph title
        2. Detect keywords in content
        """

        # Try title first
        title = getattr(chunk, 'title', None)
        if title:
            title_lower = title.lower()
            for topic, keywords in self.pattern_mappings.items():
                if any(kw in title_lower for kw in keywords):
                    return topic

        # Try content
        content_lower = chunk.content.lower()
        for topic, keywords in self.pattern_mappings.items():
            # Count keyword occurrences
            count = sum(content_lower.count(kw) for kw in keywords)
            if count >= 2:  # At least 2 mentions
                return topic

        return None

    def _topics_match(self, topic1: Optional[str], topic2: Optional[str]) -> bool:
        """Check if two topics match"""
        if not topic1 or not topic2:
            return False

        # Exact match
        if topic1 == topic2:
            return True

        # Check if they share keywords
        keywords1 = set(self.pattern_mappings.get(topic1, []))
        keywords2 = set(self.pattern_mappings.get(topic2, []))

        # Any overlap in keywords?
        return bool(keywords1 & keywords2)

    def _deduplicate_structural_matches(
        self,
        pairs: List[DocumentPair]
    ) -> List[DocumentPair]:
        """Remove duplicate pairs, keeping highest score"""

        seen = {}

        for pair in pairs:
            key = (pair.source_chunk.chunk_id, pair.target_chunk.chunk_id)

            if key not in seen:
                seen[key] = pair
            else:
                # Keep higher score
                if pair.structural_score > seen[key].structural_score:
                    seen[key] = pair

        return list(seen.values())


# ============================================================================
# Comparative Retriever - Main Orchestrator
# ============================================================================


class ComparativeRetriever:
    """
    Main class for cross-document retrieval

    Orchestrates three matching strategies:
    1. Explicit reference matching
    2. Semantic matching
    3. Structural pattern matching
    """

    def __init__(
        self,
        vector_store: Any,
        embedder: Any,
        reference_map: ReferenceMap,
        config: Optional[Dict] = None
    ):
        self.vector_store = vector_store
        self.config = config or {}

        # Initialize matchers
        self.explicit_matcher = ExplicitReferenceMatcher(reference_map)
        self.semantic_matcher = SemanticMatcher(embedder, vector_store)
        self.structural_matcher = StructuralMatcher(vector_store)

        # Weights for combining scores
        self.explicit_weight = self.config.get('explicit_weight', 0.5)
        self.semantic_weight = self.config.get('semantic_weight', 0.3)
        self.structural_weight = self.config.get('structural_weight', 0.2)

    async def find_related_provisions(
        self,
        source_chunk: LegalChunk,
        target_document_id: str,
        top_k: int = 10
    ) -> CrossDocumentResults:
        """
        Find all related provisions in target document

        Combines all three matching strategies

        Args:
            source_chunk: Source chunk (e.g., contract clause)
            target_document_id: Target document (e.g., law)
            top_k: Number of top matches to return

        Returns:
            CrossDocumentResults with all matches
        """

        start_time = time.time()

        # 1. Run all matchers in parallel
        explicit_pairs, semantic_pairs, structural_pairs = await asyncio.gather(
            self.explicit_matcher.find_explicit_matches(source_chunk, target_document_id),
            self.semantic_matcher.find_semantic_matches(source_chunk, target_document_id, top_k=top_k),
            self.structural_matcher.find_structural_matches(source_chunk, target_document_id, top_k=top_k)
        )

        # 2. Combine and deduplicate
        all_pairs = self._merge_pairs(
            explicit_pairs,
            semantic_pairs,
            structural_pairs
        )

        # 3. Score and rank
        all_pairs = self._compute_combined_scores(all_pairs)
        all_pairs.sort(key=lambda p: p.overall_score, reverse=True)

        # 4. Take top-K
        top_pairs = all_pairs[:top_k]

        # 5. Compute similarity matrix (optional)
        similarity_matrix = None
        if self.config.get('compute_similarity_matrix', False):
            similarity_matrix = await self._compute_similarity_matrix(
                source_chunk,
                [p.target_chunk for p in top_pairs]
            )

        # 6. Create results
        retrieval_time_ms = (time.time() - start_time) * 1000

        # Get document types safely
        source_doc_type = source_chunk.document_type
        target_doc_type = "unknown"
        if top_pairs:
            target_doc_type = top_pairs[0].target_chunk.document_type

        results = CrossDocumentResults(
            query=source_chunk.content[:100] + "...",  # Preview
            source_document_type=source_doc_type,
            target_document_type=target_doc_type,
            pairs=top_pairs,
            similarity_matrix=similarity_matrix,
            total_pairs=len(top_pairs),
            explicit_matches=len(explicit_pairs),
            semantic_matches=len(semantic_pairs),
            structural_matches=len(structural_pairs),
            retrieval_time_ms=retrieval_time_ms
        )

        return results

    def _merge_pairs(
        self,
        explicit_pairs: List[DocumentPair],
        semantic_pairs: List[DocumentPair],
        structural_pairs: List[DocumentPair]
    ) -> List[DocumentPair]:
        """
        Merge pairs from different strategies, handling duplicates

        If same (source, target) pair appears in multiple strategies,
        merge their scores
        """

        pair_map = {}

        # Add explicit pairs
        for pair in explicit_pairs:
            key = (pair.source_chunk.chunk_id, pair.target_chunk.chunk_id)
            pair_map[key] = pair

        # Add semantic pairs
        for pair in semantic_pairs:
            key = (pair.source_chunk.chunk_id, pair.target_chunk.chunk_id)
            if key in pair_map:
                # Merge scores
                existing = pair_map[key]
                existing.semantic_score = pair.semantic_score
            else:
                pair_map[key] = pair

        # Add structural pairs
        for pair in structural_pairs:
            key = (pair.source_chunk.chunk_id, pair.target_chunk.chunk_id)
            if key in pair_map:
                # Merge scores
                existing = pair_map[key]
                existing.structural_score = pair.structural_score
            else:
                pair_map[key] = pair

        return list(pair_map.values())

    def _compute_combined_scores(
        self,
        pairs: List[DocumentPair]
    ) -> List[DocumentPair]:
        """
        Compute combined scores from component scores

        Formula:
        combined = α·explicit + β·semantic + γ·structural
        """

        for pair in pairs:
            explicit = pair.explicit_score or 0.0
            semantic = pair.semantic_score or 0.0
            structural = pair.structural_score or 0.0

            combined = (
                self.explicit_weight * explicit +
                self.semantic_weight * semantic +
                self.structural_weight * structural
            )

            pair.overall_score = combined

            # Update confidence (max of component confidences)
            pair.confidence = max(
                explicit,
                semantic,
                structural
            )

        return pairs

    async def _compute_similarity_matrix(
        self,
        source_chunk: LegalChunk,
        target_chunks: List[LegalChunk]
    ) -> np.ndarray:
        """
        Compute similarity matrix between source and multiple targets

        Returns:
        NxM matrix where:
        - N = 1 (single source)
        - M = len(target_chunks)
        - matrix[0, j] = similarity(source, target_j)
        """

        # Get embeddings
        source_embedding = await self.semantic_matcher._get_chunk_embedding(source_chunk)

        target_embeddings = []
        for target in target_chunks:
            emb = await self.semantic_matcher._get_chunk_embedding(target)
            target_embeddings.append(emb)

        target_embeddings = np.array(target_embeddings)

        # Compute cosine similarities
        similarities = np.dot(target_embeddings, source_embedding)

        # Reshape to matrix
        similarity_matrix = similarities.reshape(1, -1)

        return similarity_matrix

    async def batch_find_related_provisions(
        self,
        source_chunks: List[LegalChunk],
        target_document_id: str,
        top_k: int = 10
    ) -> List[CrossDocumentResults]:
        """
        Batch version - find related provisions for multiple source chunks

        More efficient than calling find_related_provisions repeatedly
        """

        tasks = [
            self.find_related_provisions(chunk, target_document_id, top_k)
            for chunk in source_chunks
        ]

        results = await asyncio.gather(*tasks)

        return results


# ============================================================================
# Utility Functions
# ============================================================================


def create_comparative_retriever(
    vector_store: Any,
    embedder: Any,
    reference_map: ReferenceMap,
    config_path: Optional[str] = None
) -> ComparativeRetriever:
    """
    Factory function to create ComparativeRetriever with configuration

    Args:
        vector_store: Vector store instance
        embedder: Embedding model
        reference_map: Reference map for explicit matching
        config_path: Path to config file (optional)

    Returns:
        Configured ComparativeRetriever instance
    """

    config = {}

    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config.get('cross_document_retrieval', {})

    return ComparativeRetriever(
        vector_store=vector_store,
        embedder=embedder,
        reference_map=reference_map,
        config=config
    )


# ============================================================================
# Example Usage
# ============================================================================


async def example_usage():
    """Example usage of cross-document retrieval"""

    # 1. Setup (placeholder)
    vector_store = None  # Initialize your vector store
    embedder = None  # Initialize your embedder
    reference_map = ReferenceMap()

    # 2. Create retriever
    retriever = ComparativeRetriever(
        vector_store,
        embedder,
        reference_map,
        config={
            'explicit_weight': 0.5,
            'semantic_weight': 0.3,
            'structural_weight': 0.2
        }
    )

    # 3. Create sample source chunk
    contract_clause = LegalChunk(
        chunk_id="contract_article_5_2",
        content="Článek 5.2 - Záruční doba je 12 měsíců od předání díla. "
                "Dodavatel odpovídá za vady díla podle §2113 občanského zákoníku.",
        document_id="contract_001",
        document_type="contract",
        hierarchy_path="Část III > Článek 5 > Bod 5.2",
        legal_reference="Článek 5.2",
        structural_level="article",
        metadata={'chunk_index': 25, 'total_chunks': 100}
    )

    # 4. Find related provisions
    results = await retriever.find_related_provisions(
        contract_clause,
        target_document_id="law_89_2012",
        top_k=5
    )

    # 5. Inspect results
    print(f"Found {results.total_pairs} related provisions")
    print(f"- Explicit matches: {results.explicit_matches}")
    print(f"- Semantic matches: {results.semantic_matches}")
    print(f"- Structural matches: {results.structural_matches}")
    print(f"Retrieval time: {results.retrieval_time_ms:.2f}ms")

    # 6. Top matches
    for i, pair in enumerate(results.get_top_pairs(k=3), 1):
        print(f"\n{i}. Match: {pair.target_chunk.legal_reference}")
        print(f"   Score: {pair.overall_score:.3f}")
        print(f"   Type: {pair.match_type.value}")
        print(f"   Explanation: {pair.explanation}")
        print(f"   Content: {pair.target_chunk.content[:100]}...")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
