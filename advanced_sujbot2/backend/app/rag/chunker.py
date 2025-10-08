"""
Hierarchical Legal Chunking Strategy

Transform parsed legal documents into semantically meaningful, retrieval-optimized chunks
that preserve legal structure and hierarchy.

Key Innovation: Chunk by legal boundaries (§, articles) rather than arbitrary token counts,
while maintaining hierarchy metadata for precise citation.
"""

import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import tiktoken

# Set up logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LegalChunk:
    """A chunk of legal document optimized for retrieval"""

    # Identity
    chunk_id: str
    chunk_index: Optional[int] = None

    # Content
    content: str = ""

    # Document context
    document_id: str = ""
    document_type: str = ""  # 'law_code' | 'contract' | 'regulation'

    # Legal structure
    hierarchy_path: str = ""  # "Část II > Hlava III > §89"
    legal_reference: str = ""  # "§89" or "Článek 5.2"
    structural_level: str = ""  # 'paragraph' | 'article' | 'subsection' | etc.

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
class ChunkingConfig:
    """Configuration for chunking strategy"""

    # Size constraints (in tokens)
    min_chunk_size: int = 128
    chunk_size: int = 512  # target size
    max_chunk_size: int = 1024

    # Strategy
    strategy: str = 'hierarchical_legal'  # 'hierarchical_legal' | 'semantic' | 'hybrid'

    # Law-specific
    law_chunk_by: str = 'paragraph'  # 'paragraph' | 'subsection'
    law_include_context: bool = True  # Add hierarchical context
    law_aggregate_small: bool = True  # Aggregate paragraphs < min_size
    law_split_large: bool = True  # Split paragraphs > max_size

    # Contract-specific
    contract_chunk_by: str = 'article'  # 'article' | 'point'
    contract_include_context: bool = False
    contract_aggregate_small: bool = True
    contract_split_large: bool = True

    # Semantic chunking (for hybrid)
    semantic_similarity_threshold: float = 0.7
    respect_sentence_boundaries: bool = True

    # Overlap
    chunk_overlap: float = 0.0  # No overlap for legal chunks (structure defines boundaries)

    # Validation
    min_acceptable_size: int = 10  # tokens
    max_acceptable_size: int = 1536  # 1.5x max_chunk_size
    skip_empty: bool = True

    # Content classification
    classification_enabled: bool = True
    detect_obligations: bool = True
    detect_prohibitions: bool = True
    detect_definitions: bool = True


# ============================================================================
# Content Classifier
# ============================================================================

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

    def classify_content_type(self, content: str) -> str:
        """
        Classify content as:
        - obligation: Contains legal obligations
        - prohibition: Contains prohibitions
        - definition: Contains definitions
        - procedure: Contains procedural steps
        - general: General content

        Args:
            content: Text content to classify

        Returns:
            Content type as string
        """
        content_lower = content.lower()

        if any(re.search(p, content_lower) for p in self.OBLIGATION_PATTERNS):
            return 'obligation'

        elif any(re.search(p, content_lower) for p in self.PROHIBITION_PATTERNS):
            return 'prohibition'

        elif any(re.search(p, content_lower) for p in self.DEFINITION_PATTERNS):
            return 'definition'

        elif 'postup' in content_lower or 'řízení' in content_lower:
            return 'procedure'

        else:
            return 'general'


# ============================================================================
# Base Chunker
# ============================================================================

class HierarchicalLegalChunker(ABC):
    """Base class for hierarchical legal chunking"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.classifier = ContentClassifier()

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.tokenizer.encode(text))

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences respecting legal conventions"""
        # Simple sentence splitting (can be enhanced with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    def _get_hierarchy_path(self, element: 'StructuralElement') -> str:
        """
        Build hierarchy path for an element

        Args:
            element: Structural element

        Returns:
            Hierarchy path string like "Část II > Hlava III > §89"
        """
        parts = []
        current = element

        # Walk up the hierarchy
        while current:
            if hasattr(current, 'element_type'):
                if current.element_type == 'part' and hasattr(current, 'number'):
                    parts.insert(0, f"Část {current.number}")
                elif current.element_type == 'chapter' and hasattr(current, 'number'):
                    parts.insert(0, f"Hlava {current.number}")
                elif current.element_type == 'paragraph' and hasattr(current, 'number'):
                    parts.insert(0, f"§{current.number}")
                elif current.element_type == 'article' and hasattr(current, 'number'):
                    parts.insert(0, f"Článek {current.number}")

            # Move to parent
            current = getattr(current, 'parent', None)

        return " > ".join(parts) if parts else ""

    def _add_hierarchical_context(self, element: 'StructuralElement') -> str:
        """
        Add context from parent elements

        Example:
        "Zákon č. 89/2012 Sb., Část II Závazkové právo,
         Hlava III Kupní smlouva, §89 Odpovědnost za vady:
         (1) Dodavatel odpovídá..."

        Args:
            element: Structural element

        Returns:
            Content with hierarchical context prepended
        """
        context_parts = []

        # Walk up the hierarchy
        current = element
        while current and hasattr(current, 'element_type'):
            if current.element_type == 'part' and hasattr(current, 'number'):
                title = getattr(current, 'title', '')
                context_parts.insert(0, f"Část {current.number} {title}".strip())
            elif current.element_type == 'chapter' and hasattr(current, 'number'):
                title = getattr(current, 'title', '')
                context_parts.insert(0, f"Hlava {current.number} {title}".strip())
            elif current.element_type == 'paragraph' and hasattr(current, 'number'):
                title = getattr(current, 'title', '')
                context_parts.insert(0, f"§{current.number} {title}".strip())
            elif current.element_type == 'article' and hasattr(current, 'number'):
                title = getattr(current, 'title', '')
                context_parts.insert(0, f"Článek {current.number} {title}".strip())

            current = getattr(current, 'parent', None)

        # Add law citation if available
        if hasattr(element, 'metadata') and element.metadata.get('law_citation'):
            context_parts.insert(0, element.metadata['law_citation'])

        # Combine context + content
        context_str = " | ".join(context_parts)
        content = getattr(element, 'content', '')

        if context_str:
            return f"{context_str}\n\n{content}"
        return content

    @abstractmethod
    async def chunk(self, legal_doc: 'LegalDocument') -> List[LegalChunk]:
        """Chunk a legal document"""
        pass


# ============================================================================
# Law Code Chunker
# ============================================================================

class LawCodeChunker(HierarchicalLegalChunker):
    """Chunk law by paragraphs and subsections"""

    async def chunk(self, law_doc: 'LegalDocument') -> List[LegalChunk]:
        """
        Primary strategy: One paragraph = one chunk

        Fallbacks:
        - Small paragraphs → aggregate
        - Large paragraphs → split by subsections

        Args:
            law_doc: Legal document to chunk

        Returns:
            List of legal chunks
        """
        chunks = []

        sections = law_doc.structure.sections if hasattr(law_doc.structure, 'sections') else []

        for paragraph in sections:
            # Get paragraph content
            para_text = self._get_element_content(paragraph)
            para_tokens = self._count_tokens(para_text)

            # CASE A: Small paragraph → aggregate with neighbors
            if para_tokens < self.config.min_chunk_size and self.config.law_aggregate_small:
                chunk = await self._aggregate_small_paragraph(paragraph, law_doc.structure)
                if chunk:
                    chunks.append(chunk)

            # CASE B: Normal paragraph → one chunk
            elif para_tokens <= self.config.max_chunk_size:
                chunk = self._create_paragraph_chunk(paragraph, law_doc)
                chunks.append(chunk)

            # CASE C: Large paragraph → split by subsections
            elif self.config.law_split_large:
                subsection_chunks = await self._split_by_subsections(paragraph, law_doc)
                chunks.extend(subsection_chunks)
            else:
                # Keep as single large chunk
                chunk = self._create_paragraph_chunk(paragraph, law_doc)
                chunks.append(chunk)

        return chunks

    def _get_element_content(self, element: Any) -> str:
        """Get full text content of an element"""
        if hasattr(element, 'get_full_text'):
            return element.get_full_text()
        return getattr(element, 'content', '')

    def _create_paragraph_chunk(self, paragraph: Any, law_doc: 'LegalDocument') -> LegalChunk:
        """Create chunk from a single paragraph"""

        # Option: Add hierarchical context
        if self.config.law_include_context:
            content = self._add_hierarchical_context(paragraph)
        else:
            content = getattr(paragraph, 'content', '')

        # Extract references
        references_to = []
        referenced_by = []
        if hasattr(paragraph, 'outgoing_refs'):
            references_to = [ref.target_reference for ref in paragraph.outgoing_refs]
        if hasattr(paragraph, 'incoming_refs'):
            referenced_by = [ref.source_element_id for ref in paragraph.incoming_refs]

        # Get parent references
        chapter = getattr(paragraph, 'chapter', None)
        part = None
        if chapter and hasattr(chapter, 'part'):
            part = chapter.part

        metadata = {
            'part': getattr(part, 'number', None) if part else None,
            'chapter': getattr(chapter, 'number', None) if chapter else None,
            'paragraph': getattr(paragraph, 'number', None),
            'subsection': None,
            'references_to': references_to,
            'referenced_by': referenced_by,
            'content_type': self.classifier.classify_content_type(content),
            'start_char': getattr(paragraph, 'start_char', 0),
            'end_char': getattr(paragraph, 'end_char', 0),
            'token_count': self._count_tokens(content)
        }

        return LegalChunk(
            chunk_id=f"chunk_{getattr(paragraph, 'element_id', 'unknown')}",
            content=content,
            document_id=getattr(law_doc, 'document_id', ''),
            document_type='law_code',
            hierarchy_path=self._get_hierarchy_path(paragraph),
            legal_reference=getattr(paragraph, 'legal_reference', ''),
            structural_level='paragraph',
            metadata=metadata
        )

    async def _aggregate_small_paragraph(
        self,
        paragraph: Any,
        structure: 'DocumentStructure'
    ) -> Optional[LegalChunk]:
        """Aggregate small paragraph with neighbors"""

        # Find next paragraphs until we reach min_size
        aggregated_paras = [paragraph]
        total_tokens = self._count_tokens(self._get_element_content(paragraph))

        # Get siblings (paragraphs in same chapter)
        siblings = self._get_siblings(paragraph, structure)
        if not siblings:
            # If no siblings, return single paragraph chunk anyway
            return self._create_paragraph_chunk(paragraph, type('Doc', (), {
                'document_id': '',
                'structure': structure
            })())

        try:
            para_index = siblings.index(paragraph)
        except ValueError:
            return None

        # Try to aggregate with next paragraphs
        for next_para in siblings[para_index + 1:]:
            next_tokens = self._count_tokens(self._get_element_content(next_para))

            if total_tokens + next_tokens <= self.config.max_chunk_size:
                aggregated_paras.append(next_para)
                total_tokens += next_tokens

                if total_tokens >= self.config.min_chunk_size:
                    break
            else:
                break

        # Create aggregated chunk
        combined_content = "\n\n".join(
            self._get_element_content(p) for p in aggregated_paras
        )

        first_para = aggregated_paras[0]
        last_para = aggregated_paras[-1]

        return LegalChunk(
            chunk_id=f"chunk_aggregated_{getattr(first_para, 'element_id', 'unknown')}",
            content=combined_content,
            document_id='',
            document_type='law_code',
            hierarchy_path=self._get_hierarchy_path(first_para),
            legal_reference=f"§{getattr(first_para, 'number', '?')}-{getattr(last_para, 'number', '?')}",
            structural_level='paragraph_group',
            metadata={
                'aggregated_paragraphs': [getattr(p, 'number', None) for p in aggregated_paras],
                'is_aggregated': True,
                'token_count': total_tokens,
                'content_type': self.classifier.classify_content_type(combined_content)
            }
        )

    def _get_siblings(self, element: Any, structure: 'DocumentStructure') -> List[Any]:
        """Get sibling elements (elements with same parent)"""
        parent = getattr(element, 'parent', None)
        if not parent:
            # Top-level elements - return all sections
            return getattr(structure, 'sections', [])

        # Return children of parent
        if hasattr(parent, 'sections'):
            return parent.sections
        elif hasattr(parent, 'children'):
            return parent.children
        return []

    async def _split_by_subsections(
        self,
        paragraph: Any,
        law_doc: 'LegalDocument'
    ) -> List[LegalChunk]:
        """Split large paragraph into subsection chunks"""

        chunks = []
        subsections = getattr(paragraph, 'subsections', [])

        if not subsections:
            # No subsections, return as single chunk
            return [self._create_paragraph_chunk(paragraph, law_doc)]

        for subsection in subsections:
            content = getattr(subsection, 'content', '')

            chapter = getattr(paragraph, 'chapter', None)
            part = None
            if chapter and hasattr(chapter, 'part'):
                part = chapter.part

            chunk = LegalChunk(
                chunk_id=f"chunk_{getattr(subsection, 'element_id', 'unknown')}",
                content=content,
                document_id=getattr(law_doc, 'document_id', ''),
                document_type='law_code',
                hierarchy_path=self._get_hierarchy_path(subsection),
                legal_reference=getattr(subsection, 'legal_reference', ''),
                structural_level='subsection',
                metadata={
                    'part': getattr(part, 'number', None) if part else None,
                    'chapter': getattr(chapter, 'number', None) if chapter else None,
                    'paragraph': getattr(paragraph, 'number', None),
                    'subsection': getattr(subsection, 'number', None),
                    'token_count': self._count_tokens(content),
                    'content_type': self.classifier.classify_content_type(content)
                }
            )
            chunks.append(chunk)

        return chunks


# ============================================================================
# Contract Chunker
# ============================================================================

class ContractChunker(HierarchicalLegalChunker):
    """Chunk contracts by articles and points"""

    async def chunk(self, contract_doc: 'LegalDocument') -> List[LegalChunk]:
        """
        Primary strategy: One article = one chunk

        Fallbacks:
        - Small articles → aggregate
        - Large articles → split by points

        Args:
            contract_doc: Contract document to chunk

        Returns:
            List of legal chunks
        """
        chunks = []

        sections = contract_doc.structure.sections if hasattr(contract_doc.structure, 'sections') else []

        for article in sections:
            article_text = self._get_element_content(article)
            article_tokens = self._count_tokens(article_text)

            # CASE A: Small article
            if article_tokens < self.config.min_chunk_size and self.config.contract_aggregate_small:
                chunk = await self._aggregate_small_article(article, contract_doc.structure)
                if chunk:
                    chunks.append(chunk)

            # CASE B: Normal article
            elif article_tokens <= self.config.max_chunk_size:
                chunk = self._create_article_chunk(article, contract_doc)
                chunks.append(chunk)

            # CASE C: Large article → split by points
            elif self.config.contract_split_large:
                point_chunks = await self._split_by_points(article, contract_doc)
                chunks.extend(point_chunks)
            else:
                # Keep as single large chunk
                chunk = self._create_article_chunk(article, contract_doc)
                chunks.append(chunk)

        return chunks

    def _get_element_content(self, element: Any) -> str:
        """Get full text content of an element"""
        if hasattr(element, 'get_full_text'):
            return element.get_full_text()
        return getattr(element, 'content', '')

    def _create_article_chunk(self, article: Any, contract_doc: 'LegalDocument') -> LegalChunk:
        """Create chunk from a single article"""

        content = getattr(article, 'content', '')

        return LegalChunk(
            chunk_id=f"chunk_{getattr(article, 'element_id', 'unknown')}",
            content=content,
            document_id=getattr(contract_doc, 'document_id', ''),
            document_type='contract',
            hierarchy_path=f"Článek {getattr(article, 'number', '?')}",
            legal_reference=f"Článek {getattr(article, 'number', '?')}",
            structural_level='article',
            metadata={
                'article': getattr(article, 'number', None),
                'article_title': getattr(article, 'title', ''),
                'point': None,
                'parties_mentioned': self._extract_parties(content),
                'contains_obligation': self._has_obligation(content),
                'contains_penalty': self._has_penalty(content),
                'token_count': self._count_tokens(content),
                'content_type': self.classifier.classify_content_type(content)
            }
        )

    def _extract_parties(self, content: str) -> List[str]:
        """Extract mentioned parties from content"""
        parties = []
        # Simple heuristic - look for common party references
        party_patterns = [
            r'dodavatel',
            r'objednatel',
            r'kupující',
            r'prodávající',
            r'nájemce',
            r'pronajímatel'
        ]

        content_lower = content.lower()
        for pattern in party_patterns:
            if re.search(pattern, content_lower):
                parties.append(pattern)

        return parties

    def _has_obligation(self, content: str) -> bool:
        """Check if content contains obligations"""
        return self.classifier.classify_content_type(content) == 'obligation'

    def _has_penalty(self, content: str) -> bool:
        """Check if content contains penalties"""
        penalty_patterns = [
            r'pokuta',
            r'penále',
            r'sankce',
            r'úrok z prodlení',
            r'náhrada škody'
        ]
        content_lower = content.lower()
        return any(re.search(p, content_lower) for p in penalty_patterns)

    async def _aggregate_small_article(
        self,
        article: Any,
        structure: 'DocumentStructure'
    ) -> Optional[LegalChunk]:
        """Aggregate small article with neighbors"""

        aggregated_articles = [article]
        total_tokens = self._count_tokens(self._get_element_content(article))

        # Get all articles
        all_articles = getattr(structure, 'sections', [])
        if not all_articles:
            return self._create_article_chunk(article, type('Doc', (), {
                'document_id': '',
                'structure': structure
            })())

        try:
            article_index = all_articles.index(article)
        except ValueError:
            return None

        # Try to aggregate with next articles
        for next_article in all_articles[article_index + 1:]:
            next_tokens = self._count_tokens(self._get_element_content(next_article))

            if total_tokens + next_tokens <= self.config.max_chunk_size:
                aggregated_articles.append(next_article)
                total_tokens += next_tokens

                if total_tokens >= self.config.min_chunk_size:
                    break
            else:
                break

        # Create aggregated chunk
        combined_content = "\n\n".join(
            self._get_element_content(a) for a in aggregated_articles
        )

        first_article = aggregated_articles[0]
        last_article = aggregated_articles[-1]

        return LegalChunk(
            chunk_id=f"chunk_aggregated_{getattr(first_article, 'element_id', 'unknown')}",
            content=combined_content,
            document_id='',
            document_type='contract',
            hierarchy_path=f"Článek {getattr(first_article, 'number', '?')}-{getattr(last_article, 'number', '?')}",
            legal_reference=f"Článek {getattr(first_article, 'number', '?')}-{getattr(last_article, 'number', '?')}",
            structural_level='article_group',
            metadata={
                'aggregated_articles': [getattr(a, 'number', None) for a in aggregated_articles],
                'is_aggregated': True,
                'token_count': total_tokens,
                'content_type': self.classifier.classify_content_type(combined_content)
            }
        )

    async def _split_by_points(
        self,
        article: Any,
        contract_doc: 'LegalDocument'
    ) -> List[LegalChunk]:
        """Split article into point chunks"""

        chunks = []
        points = getattr(article, 'points', [])

        if not points:
            # No points, return as single chunk
            return [self._create_article_chunk(article, contract_doc)]

        for point in points:
            content = getattr(point, 'content', '')

            chunk = LegalChunk(
                chunk_id=f"chunk_{getattr(point, 'element_id', 'unknown')}",
                content=content,
                document_id=getattr(contract_doc, 'document_id', ''),
                document_type='contract',
                hierarchy_path=f"Článek {getattr(article, 'number', '?')}.{getattr(point, 'number', '?')}",
                legal_reference=f"Článek {getattr(article, 'number', '?')}.{getattr(point, 'number', '?')}",
                structural_level='article_point',
                metadata={
                    'article': getattr(article, 'number', None),
                    'article_title': getattr(article, 'title', ''),
                    'point': getattr(point, 'number', None),
                    'token_count': self._count_tokens(content),
                    'content_type': self.classifier.classify_content_type(content)
                }
            )
            chunks.append(chunk)

        return chunks


# ============================================================================
# Hybrid Semantic Chunker
# ============================================================================

class HybridSemanticChunker(HierarchicalLegalChunker):
    """Combines structural and semantic chunking"""

    def __init__(self, config: ChunkingConfig, embedder=None):
        super().__init__(config)
        self.embedder = embedder

    async def chunk(self, legal_doc: 'LegalDocument') -> List[LegalChunk]:
        """
        1. Get structural boundaries (§, articles)
        2. For each boundary, check size
        3. If too large, apply semantic splitting
        4. Always preserve structural metadata

        Args:
            legal_doc: Legal document to chunk

        Returns:
            List of legal chunks
        """

        structural_chunks = self._get_structural_boundaries(legal_doc)

        final_chunks = []

        for struct_chunk in structural_chunks:
            tokens = struct_chunk.metadata.get('token_count', 0)

            if tokens <= self.config.max_chunk_size:
                # Keep as is
                final_chunks.append(struct_chunk)
            else:
                # Apply semantic splitting while preserving structure
                semantic_splits = await self._semantic_split(struct_chunk)
                final_chunks.extend(semantic_splits)

        return final_chunks

    def _get_structural_boundaries(self, legal_doc: 'LegalDocument') -> List[LegalChunk]:
        """Extract structural boundaries as initial chunks"""
        chunks = []

        sections = legal_doc.structure.sections if hasattr(legal_doc.structure, 'sections') else []

        for section in sections:
            content = self._get_element_content(section)

            chunk = LegalChunk(
                chunk_id=f"chunk_{getattr(section, 'element_id', 'unknown')}",
                content=content,
                document_id=getattr(legal_doc, 'document_id', ''),
                document_type=getattr(legal_doc, 'document_type', 'regulation'),
                hierarchy_path=self._get_hierarchy_path(section),
                legal_reference=getattr(section, 'legal_reference', ''),
                structural_level=getattr(section, 'element_type', 'section'),
                metadata={
                    'token_count': self._count_tokens(content),
                    'content_type': self.classifier.classify_content_type(content)
                }
            )
            chunks.append(chunk)

        return chunks

    def _get_element_content(self, element: Any) -> str:
        """Get full text content of an element"""
        if hasattr(element, 'get_full_text'):
            return element.get_full_text()
        return getattr(element, 'content', '')

    async def _semantic_split(self, large_chunk: LegalChunk) -> List[LegalChunk]:
        """
        Split large chunk semantically while preserving metadata

        Uses sentence boundaries and embeddings to find natural split points

        Args:
            large_chunk: Chunk that exceeds max size

        Returns:
            List of smaller chunks
        """
        sentences = self._split_into_sentences(large_chunk.content)

        if not sentences:
            return [large_chunk]

        # If embedder available, use semantic boundaries
        if self.embedder:
            try:
                # Generate embeddings for each sentence
                embeddings = await self.embedder.encode(sentences)
                boundaries = self._find_semantic_boundaries(embeddings, sentences)
            except Exception as e:
                logger.warning(f"Semantic splitting failed, using simple splitting: {e}")
                boundaries = self._find_token_boundaries(sentences)
        else:
            # Fallback to token-based splitting
            boundaries = self._find_token_boundaries(sentences)

        # Create chunks at boundaries
        sub_chunks = []
        start_idx = 0

        for boundary_idx in boundaries:
            sub_content = " ".join(sentences[start_idx:boundary_idx])

            if not sub_content.strip():
                start_idx = boundary_idx
                continue

            sub_chunk = LegalChunk(
                chunk_id=f"{large_chunk.chunk_id}_sub_{len(sub_chunks)}",
                content=sub_content,
                document_id=large_chunk.document_id,
                document_type=large_chunk.document_type,
                hierarchy_path=large_chunk.hierarchy_path,
                legal_reference=large_chunk.legal_reference,
                structural_level=f"{large_chunk.structural_level}_split",
                metadata={
                    **large_chunk.metadata,
                    'is_semantic_split': True,
                    'split_index': len(sub_chunks),
                    'parent_chunk': large_chunk.chunk_id,
                    'token_count': self._count_tokens(sub_content)
                }
            )
            sub_chunks.append(sub_chunk)
            start_idx = boundary_idx

        # Add remaining sentences
        if start_idx < len(sentences):
            sub_content = " ".join(sentences[start_idx:])
            if sub_content.strip():
                sub_chunk = LegalChunk(
                    chunk_id=f"{large_chunk.chunk_id}_sub_{len(sub_chunks)}",
                    content=sub_content,
                    document_id=large_chunk.document_id,
                    document_type=large_chunk.document_type,
                    hierarchy_path=large_chunk.hierarchy_path,
                    legal_reference=large_chunk.legal_reference,
                    structural_level=f"{large_chunk.structural_level}_split",
                    metadata={
                        **large_chunk.metadata,
                        'is_semantic_split': True,
                        'split_index': len(sub_chunks),
                        'parent_chunk': large_chunk.chunk_id,
                        'token_count': self._count_tokens(sub_content)
                    }
                )
                sub_chunks.append(sub_chunk)

        return sub_chunks if sub_chunks else [large_chunk]

    def _find_token_boundaries(self, sentences: List[str]) -> List[int]:
        """Find boundaries based on token count"""
        boundaries = []
        current_tokens = 0
        current_sentences = []

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.config.chunk_size:
                if current_sentences:
                    boundaries.append(i)
                    current_tokens = sentence_tokens
                    current_sentences = [sentence]
                else:
                    # Single sentence exceeds chunk size, force boundary
                    boundaries.append(i + 1)
            else:
                current_tokens += sentence_tokens
                current_sentences.append(sentence)

        return boundaries

    def _find_semantic_boundaries(self, embeddings: List, sentences: List[str]) -> List[int]:
        """
        Find semantic boundaries using cosine similarity

        Args:
            embeddings: Sentence embeddings
            sentences: List of sentences

        Returns:
            List of boundary indices
        """
        import numpy as np

        boundaries = []
        current_tokens = 0

        for i in range(len(sentences) - 1):
            sentence_tokens = self._count_tokens(sentences[i])

            # Calculate cosine similarity between consecutive sentences
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]

            # Normalize
            emb1_norm = emb1 / np.linalg.norm(emb1)
            emb2_norm = emb2 / np.linalg.norm(emb2)

            similarity = np.dot(emb1_norm, emb2_norm)

            # Low similarity indicates topic boundary
            if similarity < self.config.semantic_similarity_threshold:
                if current_tokens >= self.config.min_chunk_size:
                    boundaries.append(i + 1)
                    current_tokens = 0
                    continue

            current_tokens += sentence_tokens

            # Force boundary if approaching max size
            if current_tokens >= self.config.chunk_size:
                boundaries.append(i + 1)
                current_tokens = 0

        return boundaries


# ============================================================================
# Chunking Pipeline
# ============================================================================

class LegalChunkingPipeline:
    """Orchestrates chunking process"""

    def __init__(self, config: ChunkingConfig, embedder=None):
        self.config = config
        self.law_chunker = LawCodeChunker(config)
        self.contract_chunker = ContractChunker(config)
        self.hybrid_chunker = HybridSemanticChunker(config, embedder)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    async def chunk_document(
        self,
        legal_doc: 'LegalDocument'
    ) -> List[LegalChunk]:
        """
        Main entry point for chunking

        Args:
            legal_doc: Legal document to chunk

        Returns:
            List of legal chunks
        """

        # Select chunker based on document type
        doc_type = getattr(legal_doc, 'document_type', 'regulation')

        if doc_type == 'law_code':
            chunks = await self.law_chunker.chunk(legal_doc)

        elif doc_type == 'contract':
            chunks = await self.contract_chunker.chunk(legal_doc)

        else:  # regulation or unknown
            chunks = await self.hybrid_chunker.chunk(legal_doc)

        # Post-processing
        chunks = self._add_chunk_indices(chunks)
        chunks = self._enrich_metadata(chunks, legal_doc)
        chunks = self._validate_chunks(chunks)

        logger.info(f"Chunked {getattr(legal_doc, 'document_id', 'document')}: {len(chunks)} chunks")

        return chunks

    def _add_chunk_indices(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Add sequential indices to chunks"""
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.metadata['chunk_index'] = i
            chunk.metadata['total_chunks'] = len(chunks)
        return chunks

    def _enrich_metadata(self, chunks: List[LegalChunk], legal_doc: 'LegalDocument') -> List[LegalChunk]:
        """Enrich chunk metadata with document-level information"""
        doc_metadata = getattr(legal_doc, 'metadata', None)

        for chunk in chunks:
            if doc_metadata:
                # Add document-level metadata
                chunk.metadata['document_title'] = getattr(doc_metadata, 'title', '')
                chunk.metadata['document_number'] = getattr(doc_metadata, 'document_number', '')
                chunk.metadata['effective_date'] = getattr(doc_metadata, 'effective_date', None)

                # For laws
                if chunk.document_type == 'law_code':
                    chunk.metadata['law_type'] = getattr(doc_metadata, 'law_type', '')
                    chunk.metadata['law_citation'] = getattr(doc_metadata, 'document_number', '')

                # For contracts
                elif chunk.document_type == 'contract':
                    chunk.metadata['contract_type'] = getattr(doc_metadata, 'contract_type', '')
                    chunk.metadata['parties'] = getattr(doc_metadata, 'parties', [])

        return chunks

    def _validate_chunks(self, chunks: List[LegalChunk]) -> List[LegalChunk]:
        """Validate chunk quality"""
        validated = []

        for chunk in chunks:
            tokens = chunk.metadata.get('token_count', 0)

            # Skip empty chunks if configured
            if self.config.skip_empty and not chunk.content.strip():
                logger.warning(f"Skipping empty chunk: {chunk.chunk_id}")
                continue

            # Filter out too small chunks (unless structural boundary)
            if tokens < self.config.min_acceptable_size:
                if not chunk.metadata.get('is_structural_boundary'):
                    logger.warning(f"Skipping tiny chunk: {chunk.chunk_id} ({tokens} tokens)")
                    continue

            # Warn about very large chunks
            if tokens > self.config.max_acceptable_size:
                logger.warning(
                    f"Large chunk detected: {chunk.chunk_id} ({tokens} tokens)"
                )

            validated.append(chunk)

        return validated

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.tokenizer.encode(text))
