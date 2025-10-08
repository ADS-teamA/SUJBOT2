# Chunking Strategy Specification - Hierarchical Legal Chunking

## 1. Purpose

Transform parsed legal documents into semantically meaningful, retrieval-optimized chunks that preserve legal structure and hierarchy.

**Key Innovation**: Chunk by legal boundaries (§, articles) rather than arbitrary token counts, while maintaining hierarchy metadata for precise citation.

---

## 2. Design Principles

### 2.1 Legal Structure First

**Traditional RAG chunking**:
```
Chunk 1: "...konec §88. §89 (1) Dodavatel odpovídá..."
Chunk 2: "...za vady. (2) Záruční doba je..."
```
❌ Problems:
- Chunks split mid-paragraph
- Lost structural context
- Cannot cite: "§89 odst. 2"

**Legal chunking**:
```
Chunk 1: "§89 (1) Dodavatel odpovídá za vady díla..."
Chunk 2: "§89 (2) Záruční doba je 24 měsíců..."
```
✅ Benefits:
- Aligned with legal structure
- Precise citation
- Preserved hierarchy

### 2.2 Adaptive Chunking

Different strategies for different element sizes:

| Element Size | Strategy | Rationale |
|-------------|----------|-----------|
| < min_size | Aggregate with neighbors | Too small, loses context |
| min_size - max_size | One chunk | Optimal size |
| > max_size | Split into subsections | Too large, split semantically |

---

## 3. Chunking Strategies

### 3.1 Law Code Chunking

```python
class LawCodeChunker:
    """Chunk law by paragraphs and subsections"""

    def __init__(self, config: ChunkingConfig):
        self.min_chunk_size = config.min_chunk_size  # 128 tokens
        self.max_chunk_size = config.max_chunk_size  # 1024 tokens
        self.include_context = config.include_context  # Add parent context

    async def chunk_law(self, law_doc: LegalDocument) -> List[LegalChunk]:
        """
        Primary strategy: One paragraph = one chunk

        Fallbacks:
        - Small paragraphs → aggregate
        - Large paragraphs → split by subsections
        """
        chunks = []

        for paragraph in law_doc.structure.sections:
            para_text = paragraph.get_full_text()
            para_tokens = self._count_tokens(para_text)

            # CASE A: Small paragraph → aggregate with neighbors
            if para_tokens < self.min_chunk_size:
                chunk = await self._aggregate_small_paragraph(
                    paragraph,
                    law_doc.structure
                )
                chunks.append(chunk)

            # CASE B: Normal paragraph → one chunk
            elif para_tokens <= self.max_chunk_size:
                chunk = self._create_paragraph_chunk(paragraph)
                chunks.append(chunk)

            # CASE C: Large paragraph → split by subsections
            else:
                subsection_chunks = await self._split_by_subsections(paragraph)
                chunks.extend(subsection_chunks)

        return chunks

    def _create_paragraph_chunk(self, paragraph: Paragraph) -> LegalChunk:
        """Create chunk from a single paragraph"""

        # Option: Add hierarchical context
        if self.include_context:
            content = self._add_hierarchical_context(paragraph)
        else:
            content = paragraph.content

        return LegalChunk(
            chunk_id=f"chunk_{paragraph.element_id}",
            content=content,
            document_type='law_code',

            # Legal metadata
            hierarchy_path=self._get_hierarchy_path(paragraph),
            legal_reference=paragraph.legal_reference,  # "§89"
            structural_level='paragraph',

            # Hierarchical IDs
            metadata={
                'part': paragraph.chapter.part.number if paragraph.chapter and paragraph.chapter.part else None,
                'chapter': paragraph.chapter.number if paragraph.chapter else None,
                'paragraph': paragraph.number,
                'subsection': None,

                # References
                'references_to': [ref.target_reference for ref in paragraph.outgoing_refs],
                'referenced_by': [ref.source_element_id for ref in paragraph.incoming_refs],

                # Content type
                'content_type': self._classify_content_type(paragraph),

                # Position
                'start_char': paragraph.start_char,
                'end_char': paragraph.end_char,
                'token_count': self._count_tokens(content)
            }
        )

    async def _aggregate_small_paragraph(
        self,
        paragraph: Paragraph,
        structure: DocumentStructure
    ) -> LegalChunk:
        """Aggregate small paragraph with neighbors"""

        # Find next paragraphs until we reach min_size
        aggregated_paras = [paragraph]
        total_tokens = self._count_tokens(paragraph.content)

        # Get siblings (paragraphs in same chapter)
        siblings = structure.get_siblings(paragraph)
        para_index = siblings.index(paragraph)

        # Try to aggregate with next paragraphs
        for next_para in siblings[para_index + 1:]:
            next_tokens = self._count_tokens(next_para.content)

            if total_tokens + next_tokens <= self.max_chunk_size:
                aggregated_paras.append(next_para)
                total_tokens += next_tokens

                if total_tokens >= self.min_chunk_size:
                    break
            else:
                break

        # Create aggregated chunk
        combined_content = "\n\n".join(p.content for p in aggregated_paras)

        return LegalChunk(
            chunk_id=f"chunk_aggregated_{paragraph.element_id}",
            content=combined_content,
            document_type='law_code',
            hierarchy_path=self._get_hierarchy_path(paragraph),
            legal_reference=f"§{paragraph.number}-{aggregated_paras[-1].number}",
            structural_level='paragraph_group',
            metadata={
                'aggregated_paragraphs': [p.number for p in aggregated_paras],
                'is_aggregated': True,
                **self._extract_metadata(paragraph)
            }
        )

    async def _split_by_subsections(
        self,
        paragraph: Paragraph
    ) -> List[LegalChunk]:
        """Split large paragraph into subsection chunks"""

        chunks = []

        for subsection in paragraph.subsections:
            chunk = LegalChunk(
                chunk_id=f"chunk_{subsection.element_id}",
                content=subsection.content,
                document_type='law_code',
                hierarchy_path=self._get_hierarchy_path(subsection),
                legal_reference=subsection.legal_reference,  # "§89 odst. 2"
                structural_level='subsection',
                metadata={
                    'part': paragraph.chapter.part.number if paragraph.chapter and paragraph.chapter.part else None,
                    'chapter': paragraph.chapter.number if paragraph.chapter else None,
                    'paragraph': paragraph.number,
                    'subsection': subsection.number,
                    'token_count': self._count_tokens(subsection.content)
                }
            )
            chunks.append(chunk)

        return chunks

    def _add_hierarchical_context(self, element: StructuralElement) -> str:
        """
        Add context from parent elements

        Example:
        "Zákon č. 89/2012 Sb., Část II Závazkové právo,
         Hlava III Kupní smlouva, §89 Odpovědnost za vady:
         (1) Dodavatel odpovídá..."
        """
        context_parts = []

        # Walk up the hierarchy
        current = element
        while current:
            if isinstance(current, Part):
                context_parts.insert(0, f"Část {current.number} {current.title}")
            elif isinstance(current, Chapter):
                context_parts.insert(0, f"Hlava {current.number} {current.title}")
            elif isinstance(current, Paragraph):
                context_parts.insert(0, f"§{current.number} {current.title}")

            current = current.parent

        # Add law citation if available
        if element.metadata.get('law_citation'):
            context_parts.insert(0, element.metadata['law_citation'])

        # Combine context + content
        context_str = " | ".join(context_parts)
        return f"{context_str}\n\n{element.content}"
```

### 3.2 Contract Chunking

```python
class ContractChunker:
    """Chunk contracts by articles and points"""

    async def chunk_contract(self, contract_doc: LegalDocument) -> List[LegalChunk]:
        """
        Primary strategy: One article = one chunk

        Fallbacks:
        - Small articles → aggregate
        - Large articles → split by points
        """
        chunks = []

        for article in contract_doc.structure.sections:
            article_text = article.get_full_text()
            article_tokens = self._count_tokens(article_text)

            # CASE A: Small article
            if article_tokens < self.min_chunk_size:
                chunk = await self._aggregate_small_article(article, contract_doc.structure)
                chunks.append(chunk)

            # CASE B: Normal article
            elif article_tokens <= self.max_chunk_size:
                chunk = self._create_article_chunk(article)
                chunks.append(chunk)

            # CASE C: Large article → split by points
            else:
                point_chunks = await self._split_by_points(article)
                chunks.extend(point_chunks)

        return chunks

    def _create_article_chunk(self, article: Article) -> LegalChunk:
        """Create chunk from a single article"""

        return LegalChunk(
            chunk_id=f"chunk_{article.element_id}",
            content=article.content,
            document_type='contract',

            hierarchy_path=f"Článek {article.number}",
            legal_reference=f"Článek {article.number}",
            structural_level='article',

            metadata={
                'article': article.number,
                'article_title': article.title,
                'point': None,

                # Contract-specific
                'parties_mentioned': self._extract_parties(article.content),
                'contains_obligation': self._has_obligation(article.content),
                'contains_penalty': self._has_penalty(article.content),

                'token_count': self._count_tokens(article.content)
            }
        )

    async def _split_by_points(self, article: Article) -> List[LegalChunk]:
        """Split article into point chunks"""

        chunks = []

        for point in article.points:
            chunk = LegalChunk(
                chunk_id=f"chunk_{point.element_id}",
                content=point.content,
                document_type='contract',

                hierarchy_path=f"Článek {article.number}.{point.number}",
                legal_reference=f"Článek {article.number}.{point.number}",
                structural_level='article_point',

                metadata={
                    'article': article.number,
                    'article_title': article.title,
                    'point': point.number,
                    'token_count': self._count_tokens(point.content)
                }
            )
            chunks.append(chunk)

        return chunks
```

### 3.3 Hybrid Semantic Chunking

For documents without clear structure:

```python
class HybridSemanticChunker:
    """Combines structural and semantic chunking"""

    async def chunk_hybrid(self, legal_doc: LegalDocument) -> List[LegalChunk]:
        """
        1. Get structural boundaries (§, articles)
        2. For each boundary, check size
        3. If too large, apply semantic splitting
        4. Always preserve structural metadata
        """

        structural_chunks = self._get_structural_boundaries(legal_doc)

        final_chunks = []

        for struct_chunk in structural_chunks:
            tokens = struct_chunk.token_count

            if tokens <= self.max_chunk_size:
                # Keep as is
                final_chunks.append(struct_chunk)
            else:
                # Apply semantic splitting while preserving structure
                semantic_splits = await self._semantic_split(struct_chunk)
                final_chunks.extend(semantic_splits)

        return final_chunks

    async def _semantic_split(self, large_chunk: LegalChunk) -> List[LegalChunk]:
        """
        Split large chunk semantically while preserving metadata

        Uses sentence embeddings to find natural boundaries
        """
        sentences = self._split_into_sentences(large_chunk.content)

        # Generate embeddings for each sentence
        embeddings = await self.embedder.encode(sentences)

        # Find semantic boundaries (low cosine similarity between consecutive sentences)
        boundaries = self._find_semantic_boundaries(embeddings)

        # Create chunks at boundaries
        sub_chunks = []
        start_idx = 0

        for boundary_idx in boundaries:
            sub_content = " ".join(sentences[start_idx:boundary_idx])

            sub_chunk = LegalChunk(
                chunk_id=f"{large_chunk.chunk_id}_sub_{len(sub_chunks)}",
                content=sub_content,
                document_type=large_chunk.document_type,
                hierarchy_path=large_chunk.hierarchy_path,
                legal_reference=large_chunk.legal_reference,
                structural_level=f"{large_chunk.structural_level}_split",
                metadata={
                    **large_chunk.metadata,
                    'is_semantic_split': True,
                    'split_index': len(sub_chunks),
                    'parent_chunk': large_chunk.chunk_id
                }
            )
            sub_chunks.append(sub_chunk)
            start_idx = boundary_idx

        return sub_chunks
```

---

## 4. Data Structures

```python
@dataclass
class LegalChunk:
    """A chunk of legal document optimized for retrieval"""

    # Identity
    chunk_id: str
    chunk_index: Optional[int] = None

    # Content
    content: str

    # Document context
    document_id: str
    document_type: str  # 'law_code' | 'contract' | 'regulation'

    # Legal structure
    hierarchy_path: str  # "Část II > Hlava III > §89"
    legal_reference: str  # "§89" or "Článek 5.2"
    structural_level: str  # 'paragraph' | 'article' | 'subsection' | etc.

    # Metadata
    metadata: Dict[str, Any]

    def get_citation(self) -> str:
        """Get properly formatted citation"""
        if self.document_type == 'law_code':
            law_name = self.metadata.get('law_citation', '')
            return f"{law_name}, {self.legal_reference}"
        else:
            return f"{self.hierarchy_path}"

@dataclass
class ChunkingConfig:
    """Configuration for chunking strategy"""

    # Size constraints
    min_chunk_size: int = 128  # tokens
    chunk_size: int = 512  # target size
    max_chunk_size: int = 1024  # tokens

    # Strategy
    strategy: str = 'hierarchical_legal'  # 'hierarchical_legal' | 'semantic' | 'hybrid'

    # Law-specific
    law_chunk_by: str = 'paragraph'  # 'paragraph' | 'subsection'
    law_include_context: bool = True  # Add hierarchical context

    # Contract-specific
    contract_chunk_by: str = 'article'  # 'article' | 'point'
    contract_include_context: bool = False

    # Semantic chunking (for hybrid)
    semantic_similarity_threshold: float = 0.7
    respect_sentence_boundaries: bool = True

    # Overlap
    chunk_overlap: float = 0.0  # No overlap for legal chunks (structure defines boundaries)
```

---

## 5. Chunking Pipeline

```python
# File: src/legal_chunker.py

class LegalChunkingPipeline:
    """Orchestrates chunking process"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.law_chunker = LawCodeChunker(config)
        self.contract_chunker = ContractChunker(config)
        self.hybrid_chunker = HybridSemanticChunker(config)

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def chunk_document(
        self,
        legal_doc: LegalDocument
    ) -> List[LegalChunk]:
        """Main entry point for chunking"""

        # Select chunker based on document type
        if legal_doc.document_type == 'law_code':
            chunks = await self.law_chunker.chunk_law(legal_doc)

        elif legal_doc.document_type == 'contract':
            chunks = await self.contract_chunker.chunk_contract(legal_doc)

        else:  # regulation or unknown
            chunks = await self.hybrid_chunker.chunk_hybrid(legal_doc)

        # Post-processing
        chunks = self._add_chunk_indices(chunks)
        chunks = self._validate_chunks(chunks)

        logger.info(f"Chunked {legal_doc.document_id}: {len(chunks)} chunks")

        return chunks

    def _add_chunk_indices(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Add sequential indices to chunks"""
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        return chunks

    def _validate_chunks(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Validate chunk quality"""
        validated = []

        for chunk in chunks:
            tokens = self._count_tokens(chunk.content)

            # Filter out too small chunks (unless structural boundary)
            if tokens < 10 and not chunk.metadata.get('is_structural_boundary'):
                logger.warning(f"Skipping tiny chunk: {chunk.chunk_id}")
                continue

            # Warn about very large chunks
            if tokens > self.config.max_chunk_size * 1.5:
                logger.warning(
                    f"Large chunk detected: {chunk.chunk_id} ({tokens} tokens)"
                )

            validated.append(chunk)

        return validated

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.tokenizer.encode(text))
```

---

## 6. Content Classification

```python
class ContentClassifier:
    """Classify legal content type"""

    OBLIGATION_PATTERNS = [
        r'\bmusí\b',
        r'\bje povinen\b',
        r'\bje nutné\b',
        r'\bje třeba\b',
        r'\bje povinna\b',
        r'\bmá povinnost\b'
    ]

    PROHIBITION_PATTERNS = [
        r'\bnesmí\b',
        r'\bje zakázáno\b',
        r'\bje nepřípustné\b',
        r'\bzakazuje se\b'
    ]

    DEFINITION_PATTERNS = [
        r'\bse rozumí\b',
        r'\bje\b.*\bjako\b',
        r'\bmá se za to\b',
        r'\bznačí\b',
        r'\boznačuje\b'
    ]

    def classify_content_type(self, chunk: LegalChunk) -> str:
        """
        Classify content as:
        - obligation
        - prohibition
        - definition
        - procedure
        - general
        """
        content = chunk.content.lower()

        if any(re.search(p, content) for p in self.OBLIGATION_PATTERNS):
            return 'obligation'

        elif any(re.search(p, content) for p in self.PROHIBITION_PATTERNS):
            return 'prohibition'

        elif any(re.search(p, content) for p in self.DEFINITION_PATTERNS):
            return 'definition'

        elif 'postup' in content or 'řízení' in content:
            return 'procedure'

        else:
            return 'general'
```

---

## 7. Configuration Example

```yaml
# config.yaml
chunking:
  # General settings
  min_chunk_size: 128
  chunk_size: 512
  max_chunk_size: 1024

  # Strategy
  strategy: hierarchical_legal

  # Law code
  law_code:
    chunk_by: paragraph  # paragraph | subsection
    include_context: true  # Add hierarchical context
    aggregate_small: true  # Aggregate paragraphs < min_size
    split_large: true  # Split paragraphs > max_size

  # Contract
  contract:
    chunk_by: article  # article | point
    include_context: false
    aggregate_small: true
    split_large: true

  # Content classification
  classification:
    enabled: true
    detect_obligations: true
    detect_prohibitions: true
    detect_definitions: true

  # Validation
  validation:
    min_acceptable_size: 10  # tokens
    max_acceptable_size: 1536  # 1.5x max_chunk_size
    skip_empty: true
```

---

## 8. Testing

```python
# tests/test_legal_chunker.py

def test_law_paragraph_chunking():
    """Test basic paragraph chunking"""
    law = create_mock_law()
    chunker = LawCodeChunker(ChunkingConfig())

    chunks = chunker.chunk_law(law)

    # Each paragraph should be a chunk
    assert len(chunks) == len(law.structure.sections)

    # Check metadata
    assert chunks[0].structural_level == 'paragraph'
    assert chunks[0].legal_reference.startswith('§')


def test_small_paragraph_aggregation():
    """Test aggregation of small paragraphs"""
    # Create law with very small paragraphs
    law = create_law_with_small_paragraphs()
    chunker = LawCodeChunker(ChunkingConfig(min_chunk_size=256))

    chunks = chunker.chunk_law(law)

    # Should aggregate
    assert len(chunks) < len(law.structure.sections)
    assert any(c.metadata.get('is_aggregated') for c in chunks)


def test_large_paragraph_splitting():
    """Test splitting of large paragraphs"""
    # Create law with very large paragraph
    law = create_law_with_large_paragraph()
    chunker = LawCodeChunker(ChunkingConfig(max_chunk_size=512))

    chunks = chunker.chunk_law(law)

    # Should split into subsections
    subsection_chunks = [c for c in chunks if c.structural_level == 'subsection']
    assert len(subsection_chunks) > 0


def test_hierarchical_context():
    """Test context addition"""
    law = create_mock_law()
    chunker = LawCodeChunker(ChunkingConfig(law_include_context=True))

    chunks = chunker.chunk_law(law)

    # Content should include hierarchy
    assert 'Část' in chunks[0].content or 'Hlava' in chunks[0].content
```

---

## 9. Performance Targets

| Document | Chunks | Chunking Time | Memory |
|----------|--------|---------------|--------|
| 100 pages law | ~300 | <2s | <50 MB |
| 1,000 pages law | ~3,000 | <10s | <200 MB |
| 100 pages contract | ~200 | <1s | <30 MB |

---

## 10. Output Example

```json
{
  "chunk_id": "chunk_paragraph_89",
  "content": "Zákon č. 89/2012 Sb., Část II Závazkové právo, Hlava III Kupní smlouva, §89 Odpovědnost za vady\n\n(1) Dodavatel odpovídá za vady díla...",
  "document_id": "law_89_2012",
  "document_type": "law_code",
  "hierarchy_path": "Část II > Hlava III > §89",
  "legal_reference": "§89",
  "structural_level": "paragraph",
  "metadata": {
    "part": "II",
    "chapter": "III",
    "paragraph": 89,
    "subsection": null,
    "content_type": "obligation",
    "references_to": ["§88", "§90"],
    "referenced_by": ["§100", "§105"],
    "token_count": 256,
    "chunk_index": 45,
    "total_chunks": 1523
  }
}
```
