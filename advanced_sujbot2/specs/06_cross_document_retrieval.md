# Cross-Document Retrieval Specification - Comparative Retrieval

## 1. Purpose

Enable retrieval and comparison across multiple documents (contract ↔ law) to identify relationships, references, and potential conflicts.

**Key Innovation**: Three-tier matching strategy:
1. **Explicit reference matching** - Direct citations (smlouva mentions "§89")
2. **Implicit semantic matching** - Similar provisions without explicit references
3. **Structural pattern matching** - Corresponding sections (warranties, obligations)

---

## 2. Use Cases

### 2.1 Compliance Checking

**Scenario**: Check if contract clause complies with law

```
Contract: "Článek 5.2 - Záruční doba je 12 měsíců"
         ↓
Find related provisions in law
         ↓
Law: "§2113 - Minimální záruční doba je 24 měsíců"
         ↓
Identify: CONFLICT (12 < 24)
```

### 2.2 Gap Analysis

**Scenario**: Find missing mandatory provisions

```
Law: "§2079 - Smlouva musí obsahovat identifikaci stran"
    ↓
Search contract for corresponding section
    ↓
NOT FOUND → Gap identified
```

### 2.3 Reference Tracking

**Scenario**: Trace references between documents

```
Contract: "dle §89 odst. 2 občanského zákoníku"
         ↓
Direct lookup: §89 odst. 2
         ↓
Law: "§89 odst. 2 - Dodavatel odpovídá..."
```

---

## 3. Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│           CROSS-DOCUMENT RETRIEVAL SYSTEM                │
└─────────────────────────────────────────────────────────┘

    ┌──────────────────┐              ┌──────────────────┐
    │  Contract Index  │              │  Law Index       │
    │  (FAISS + meta)  │              │  (FAISS + meta)  │
    └────────┬─────────┘              └────────┬─────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │   ComparativeRetriever        │
            └───────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Explicit Ref   │  │ Semantic       │  │ Structural     │
│ Matcher        │  │ Matcher        │  │ Matcher        │
└────────┬───────┘  └────────┬───────┘  └────────┬───────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                            │
                            ▼
            ┌───────────────────────────────┐
            │  CrossDocumentResults         │
            │  - Explicit matches           │
            │  - Semantic matches           │
            │  - Structural patterns        │
            │  - Similarity matrix          │
            └───────────────────────────────┘
```

---

## 4. Data Structures

### 4.1 Cross-Document Result Types

```python
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

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
```

---

## 5. Explicit Reference Matching

### 5.1 Reference Extractor

```python
# File: src/retrieval/reference_matcher.py

import re
from typing import List, Dict, Optional, Set

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
            target_chunks = self.reference_map.get_chunks_by_reference(
                ref['normalized_ref']
            )

            # Filter by target document
            target_chunks = [
                chunk_id for chunk_id in target_chunks
                if self._get_document_id(chunk_id) == target_document_id
            ]

            # Create pairs
            for chunk_id in target_chunks:
                target_chunk = self._get_chunk(chunk_id)

                pair = DocumentPair(
                    source_chunk=source_chunk,
                    source_document_id=source_chunk.metadata['document_id'],
                    target_chunk=target_chunk,
                    target_document_id=target_document_id,
                    match_type=MatchType.EXPLICIT_REFERENCE,
                    relation_type=RelationType.REFERENCES,
                    explicit_score=1.0,  # Perfect match
                    overall_score=1.0,
                    evidence={
                        'reference_text': ref['original_text'],
                        'reference_type': ref['type'],
                        'match_method': 'direct_lookup'
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

        return None

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
```

---

## 6. Semantic Matching

### 6.1 Implicit Semantic Matcher

```python
# File: src/retrieval/semantic_matcher.py

class SemanticMatcher:
    """Find semantically similar provisions across documents"""

    def __init__(
        self,
        embedder: LegalEmbedder,
        vector_store: MultiDocumentVectorStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store

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
        # (Already computed during indexing, retrieve from store)
        source_embedding = await self._get_chunk_embedding(source_chunk)

        # 2. Search in target index
        target_index = self.vector_store.indices[target_document_id]
        target_metadata = self.vector_store.metadata_stores[target_document_id]

        # FAISS search
        query_vector = source_embedding.reshape(1, -1).astype('float32')
        scores, indices = target_index.search(query_vector, top_k * 2)

        # 3. Create pairs
        pairs = []
        chunk_ids = list(target_metadata.keys())

        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(chunk_ids):
                continue

            # Filter by minimum similarity
            if score < min_similarity:
                continue

            chunk_id = chunk_ids[idx]
            target_chunk = target_metadata[chunk_id]

            pair = DocumentPair(
                source_chunk=source_chunk,
                source_document_id=source_chunk.metadata['document_id'],
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

        # Try to get from vector store
        doc_id = chunk.metadata['document_id']
        index = self.vector_store.indices[doc_id]
        metadata_store = self.vector_store.metadata_stores[doc_id]

        # Find chunk index
        chunk_ids = list(metadata_store.keys())
        if chunk.chunk_id in chunk_ids:
            chunk_idx = chunk_ids.index(chunk.chunk_id)

            # Reconstruct vector from FAISS (not ideal but works)
            # Better: maintain separate embedding cache
            embeddings = await self.embedder.embed_chunks([chunk])
            return embeddings[0]

        # Fallback: compute fresh
        embeddings = await self.embedder.embed_chunks([chunk])
        return embeddings[0]
```

---

## 7. Structural Pattern Matching

### 7.1 Structural Matcher

```python
# File: src/retrieval/structural_matcher.py

class StructuralMatcher:
    """Match chunks based on structural patterns and positions"""

    def __init__(self, vector_store: MultiDocumentVectorStore):
        self.vector_store = vector_store

        # Structural pattern mappings
        self.pattern_mappings = {
            # Contract section → Law section
            'warranties': ['záruky', 'záruka', 'warranty'],
            'liability': ['odpovědnost', 'liability'],
            'payment': ['platba', 'úhrada', 'payment'],
            'termination': ['ukončení', 'vypovězení', 'termination'],
            'penalties': ['sankce', 'pokuta', 'penalty'],
            'obligations': ['povinnost', 'obligation'],
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

        # Find matching topic in target
        target_metadata = self.vector_store.metadata_stores[target_document_id]

        pairs = []

        for chunk_id, target_chunk in target_metadata.items():
            target_topic = self._extract_topic(target_chunk)

            if self._topics_match(source_topic, target_topic):
                pair = DocumentPair(
                    source_chunk=source_chunk,
                    source_document_id=source_chunk.metadata['document_id'],
                    target_chunk=target_chunk,
                    target_document_id=target_document_id,
                    match_type=MatchType.STRUCTURAL_PATTERN,
                    structural_score=0.8,
                    overall_score=0.8,
                    evidence={
                        'source_topic': source_topic,
                        'target_topic': target_topic,
                        'match_method': 'topic_matching'
                    },
                    confidence=0.7,
                    explanation=f"Structural topic match: {source_topic} ↔ {target_topic}"
                )
                pairs.append(pair)

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

        # Find chunks with same content type in target
        target_metadata = self.vector_store.metadata_stores[target_document_id]

        pairs = []

        for chunk_id, target_chunk in target_metadata.items():
            target_type = target_chunk.metadata.get('content_type', 'general')

            if source_type == target_type:
                pair = DocumentPair(
                    source_chunk=source_chunk,
                    source_document_id=source_chunk.metadata['document_id'],
                    target_chunk=target_chunk,
                    target_document_id=target_document_id,
                    match_type=MatchType.STRUCTURAL_PATTERN,
                    structural_score=0.6,
                    overall_score=0.6,
                    evidence={
                        'content_type': source_type,
                        'match_method': 'content_type_matching'
                    },
                    confidence=0.6,
                    explanation=f"Same content type: {source_type}"
                )
                pairs.append(pair)

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

        # Find chunks at similar position in target
        target_metadata = self.vector_store.metadata_stores[target_document_id]

        pairs = []

        for chunk_id, target_chunk in target_metadata.items():
            target_index = target_chunk.metadata.get('chunk_index', 0)
            target_total = target_chunk.metadata.get('total_chunks', 1)

            if target_total == 0:
                continue

            target_position = target_index / target_total

            # Position similarity
            position_diff = abs(source_position - target_position)

            # Only match if positions are close (within 10%)
            if position_diff < 0.1:
                score = 1.0 - position_diff * 10  # 0.1 diff → 0.0 score

                pair = DocumentPair(
                    source_chunk=source_chunk,
                    source_document_id=source_chunk.metadata['document_id'],
                    target_chunk=target_chunk,
                    target_document_id=target_document_id,
                    match_type=MatchType.STRUCTURAL_PATTERN,
                    structural_score=score * 0.4,  # Low confidence heuristic
                    overall_score=score * 0.4,
                    evidence={
                        'source_position': source_position,
                        'target_position': target_position,
                        'position_diff': position_diff,
                        'match_method': 'position_matching'
                    },
                    confidence=0.4,
                    explanation=f"Similar document position: {source_position:.2f} ≈ {target_position:.2f}"
                )
                pairs.append(pair)

        return pairs

    def _extract_topic(self, chunk: LegalChunk) -> Optional[str]:
        """
        Extract topic/subject from chunk

        Strategies:
        1. Use article/paragraph title
        2. Detect keywords in content
        """

        # Try title first
        title = chunk.title
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
```

---

## 8. Comparative Retriever - Main Class

```python
# File: src/retrieval/comparative_retriever.py

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
        vector_store: MultiDocumentVectorStore,
        embedder: LegalEmbedder,
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

        results = CrossDocumentResults(
            query=source_chunk.content[:100] + "...",  # Preview
            source_document_type=source_chunk.document_type,
            target_document_type=self.vector_store.metadata_stores[target_document_id][next(iter(self.vector_store.metadata_stores[target_document_id]))].document_type,
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
```

---

## 9. Configuration

```yaml
# config.yaml
cross_document_retrieval:
  # Strategy weights
  explicit_weight: 0.5   # α - explicit references
  semantic_weight: 0.3   # β - semantic similarity
  structural_weight: 0.2  # γ - structural patterns

  # Semantic matching
  semantic:
    top_k: 5
    min_similarity: 0.5  # Minimum cosine similarity

  # Structural matching
  structural:
    top_k: 5
    enable_topic_matching: true
    enable_content_type_matching: true
    enable_position_matching: false  # Low confidence heuristic

  # Performance
  compute_similarity_matrix: false  # Expensive
  enable_caching: true
  parallel_matching: true

  # Reference extraction
  reference_extraction:
    patterns:
      - paragraph
      - article
      - law_citation
      - contextual
    deduplicate_overlaps: true
```

---

## 10. Usage Examples

### 10.1 Basic Usage

```python
# Initialize
retriever = ComparativeRetriever(
    vector_store,
    embedder,
    reference_map,
    config
)

# Find related provisions
contract_clause = get_chunk("contract_article_5_2")
results = await retriever.find_related_provisions(
    contract_clause,
    target_document_id="law_89_2012",
    top_k=5
)

# Inspect results
print(f"Found {results.total_pairs} related provisions")
print(f"- Explicit matches: {results.explicit_matches}")
print(f"- Semantic matches: {results.semantic_matches}")
print(f"- Structural matches: {results.structural_matches}")

# Top matches
for pair in results.get_top_pairs(k=3):
    print(f"\nMatch: {pair.target_chunk.legal_reference}")
    print(f"  Score: {pair.overall_score:.3f}")
    print(f"  Type: {pair.match_type.value}")
    print(f"  Explanation: {pair.explanation}")
```

### 10.2 Batch Processing

```python
# Process entire contract
contract_chunks = get_all_chunks("contract_001")

all_results = await retriever.batch_find_related_provisions(
    contract_chunks,
    target_document_id="law_89_2012",
    top_k=5
)

# Analyze results
for chunk, results in zip(contract_chunks, all_results):
    if results.explicit_matches > 0:
        print(f"{chunk.legal_reference} has explicit law references")
```

### 10.3 Find Conflicts

```python
# Get related provisions
results = await retriever.find_related_provisions(
    contract_clause,
    "law_89_2012"
)

# Check for conflicts (requires compliance analysis)
# This is a preview - full logic in compliance_analyzer.py

for pair in results.get_top_pairs(k=5):
    # Compare provisions (simplified)
    if await check_conflict(pair.source_chunk, pair.target_chunk):
        print(f"CONFLICT: {pair.source_chunk.legal_reference} vs {pair.target_chunk.legal_reference}")
```

---

## 11. Testing

```python
# tests/test_cross_document_retrieval.py

def test_explicit_reference_extraction():
    """Test extraction of legal references"""
    matcher = ExplicitReferenceMatcher(reference_map)

    text = "Podle §89 odst. 2 občanského zákoníku musí dodavatel..."

    refs = matcher._extract_references(text)

    assert len(refs) == 1
    assert refs[0]['normalized_ref'] == "§89 odst. 2"
    assert refs[0]['components']['paragraph'] == 89
    assert refs[0]['components']['subsection'] == 2

def test_explicit_matching():
    """Test explicit reference matching"""
    contract_chunk = create_mock_chunk(
        content="Podle §89 odst. 2...",
        document_type="contract"
    )

    pairs = await matcher.find_explicit_matches(
        contract_chunk,
        "law_89_2012"
    )

    assert len(pairs) > 0
    assert pairs[0].match_type == MatchType.EXPLICIT_REFERENCE
    assert pairs[0].explicit_score == 1.0

def test_semantic_matching():
    """Test semantic similarity matching"""
    contract_chunk = create_mock_chunk(
        content="Dodavatel odpovídá za vady díla",
        document_type="contract"
    )

    pairs = await semantic_matcher.find_semantic_matches(
        contract_chunk,
        "law_89_2012",
        top_k=5
    )

    assert len(pairs) <= 5
    assert all(p.semantic_score >= 0.5 for p in pairs)

def test_combined_retrieval():
    """Test combined cross-document retrieval"""
    retriever = ComparativeRetriever(...)

    contract_chunk = get_chunk("contract_article_5")

    results = await retriever.find_related_provisions(
        contract_chunk,
        "law_89_2012",
        top_k=10
    )

    assert results.total_pairs <= 10
    assert results.explicit_matches >= 0
    assert results.semantic_matches >= 0

def test_score_combination():
    """Test score combination from multiple strategies"""
    # Create mock pairs with different scores
    pairs = [
        DocumentPair(
            ...,
            explicit_score=1.0,
            semantic_score=0.7,
            structural_score=0.5
        )
    ]

    retriever = ComparativeRetriever(...)
    pairs = retriever._compute_combined_scores(pairs)

    # Check combined score
    expected = 0.5 * 1.0 + 0.3 * 0.7 + 0.2 * 0.5
    assert abs(pairs[0].overall_score - expected) < 0.01
```

---

## 12. Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Single clause comparison | <500ms | All three strategies |
| Batch (100 clauses) | <5s | Parallel processing |
| Explicit ref extraction | <10ms | Regex-based |
| Semantic matching | <200ms | FAISS search |
| Structural matching | <100ms | Metadata filtering |

---

## 13. Future Enhancements

### 13.1 ML-Based Conflict Detection

```python
class ConflictClassifier:
    """ML model to classify provision pairs as conflicting"""

    def predict_conflict(self, pair: DocumentPair) -> float:
        """Return probability of conflict (0-1)"""
        pass
```

### 13.2 Cross-Language Matching

```python
# Match Slovak law with Czech contract
results = await retriever.find_related_provisions(
    slovak_contract_chunk,
    czech_law_document,
    enable_translation=True
)
```

### 13.3 Temporal Matching

```python
# Find how law changed over time
results = await retriever.find_related_provisions(
    chunk,
    target_documents=[
        "law_89_2012_v1",  # Original version
        "law_89_2012_v2",  # After amendment
    ]
)
```

---

**Specification Complete**: 06_cross_document_retrieval.md
**Pages**: ~18
**Status**: Ready for implementation
