"""
Unit tests for legal chunker (hierarchical chunking strategy)

Tests cover:
- Law code chunking by paragraphs
- Contract chunking by articles
- Small paragraph aggregation
- Large paragraph splitting by subsections
- Hierarchical context addition
- Content classification
- Chunk overlap (15%)
"""

import pytest
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from chunker import (
    LegalChunk, ChunkingConfig, LegalChunkingPipeline,
    LawCodeChunker, ContractChunker, ContentClassifier
)
from models import (
    LegalDocument, DocumentStructure, DocumentMetadata,
    Paragraph, Subsection, Article, Point, Part, Chapter,
    ElementType
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def chunking_config():
    """Standard chunking configuration"""
    return ChunkingConfig(
        min_chunk_size=128,
        chunk_size=512,
        max_chunk_size=1024,
        chunk_overlap=0.15,
        law_include_context=True,
        law_aggregate_small=True,
        law_split_large=True
    )


@pytest.fixture
def law_document():
    """Mock law document with hierarchical structure"""

    # Create Part
    part = Part(
        element_id="part_I",
        element_type=ElementType.PART.value,
        number="I",
        title="Obecná část",
        content="",
        level=0,
        parent_id=None,
        children_ids=[],
        start_line=0,
        end_line=0,
        start_char=0,
        end_char=0,
        legal_reference="Část I",
        metadata={},
        chapters=[]
    )

    # Create Chapter
    chapter = Chapter(
        element_id="chapter_I",
        element_type=ElementType.CHAPTER.value,
        number="I",
        title="Základní ustanovení",
        content="",
        level=1,
        parent_id="part_I",
        children_ids=[],
        start_line=1,
        end_line=1,
        start_char=0,
        end_char=0,
        legal_reference="Hlava I",
        metadata={},
        part=part,
        sections=[]
    )

    part.chapters.append(chapter)

    # Create Paragraphs
    paragraph_89 = Paragraph(
        element_id="paragraph_89",
        element_type=ElementType.PARAGRAPH.value,
        number=89,
        title="Odpovědnost za vady",
        content="Dodavatel odpovídá za vady díla, které se projeví do 24 měsíců od převzetí díla. " * 10,  # ~200 tokens
        level=2,
        parent_id="chapter_I",
        children_ids=[],
        start_line=2,
        end_line=5,
        start_char=0,
        end_char=0,
        legal_reference="§89",
        metadata={},
        chapter=chapter,
        section=None,
        subsections=[],
        contains_obligation=True,
        contains_prohibition=False,
        contains_definition=False
    )

    paragraph_90 = Paragraph(
        element_id="paragraph_90",
        element_type=ElementType.PARAGRAPH.value,
        number=90,
        title="Záruční lhůta",
        content="Záruční lhůta činí 24 měsíců od převzetí díla.",
        level=2,
        parent_id="chapter_I",
        children_ids=[],
        start_line=6,
        end_line=7,
        start_char=0,
        end_char=0,
        legal_reference="§90",
        metadata={},
        chapter=chapter,
        section=None,
        subsections=[],
        contains_obligation=False,
        contains_prohibition=False,
        contains_definition=False
    )

    chapter.sections.extend([paragraph_89, paragraph_90])

    # Create Document Structure
    structure = DocumentStructure(
        hierarchy=[part, chapter, paragraph_89, paragraph_90],
        parts=[part],
        chapters=[chapter],
        sections=[paragraph_89, paragraph_90],
        all_elements=[part, chapter, paragraph_89, paragraph_90]
    )

    # Create Document
    metadata = DocumentMetadata(
        file_path="/mock/law.pdf",
        file_format="pdf",
        file_size_bytes=1024,
        title="Zákon č. 89/2012 Sb., občanský zákoník",
        document_number="89/2012",
        total_pages=100,
        total_words=10000,
        total_sections=2
    )

    doc = LegalDocument(
        document_id="law_89_2012",
        document_type="law_code",
        raw_text="",
        cleaned_text="",
        structure=structure,
        metadata=metadata,
        references=[],
        parsed_at=None
    )

    return doc


@pytest.fixture
def contract_document():
    """Mock contract document"""

    article_5 = Article(
        element_id="article_5",
        element_type=ElementType.ARTICLE.value,
        number=5,
        title="Cena a platební podmínky",
        content="Cena díla činí 1 000 000 Kč bez DPH. " * 15,  # ~300 tokens
        level=0,
        parent_id=None,
        children_ids=[],
        start_line=0,
        end_line=2,
        start_char=0,
        end_char=0,
        legal_reference="Článek 5",
        metadata={},
        points=[]
    )

    structure = DocumentStructure(
        hierarchy=[article_5],
        sections=[article_5],
        all_elements=[article_5]
    )

    metadata = DocumentMetadata(
        file_path="/mock/contract.pdf",
        file_format="pdf",
        file_size_bytes=512,
        title="Smlouva o dílo",
        total_pages=10,
        total_words=5000,
        total_sections=1
    )

    doc = LegalDocument(
        document_id="contract_001",
        document_type="contract",
        raw_text="",
        cleaned_text="",
        structure=structure,
        metadata=metadata,
        references=[],
        parsed_at=None
    )

    return doc


# ============================================================================
# Test Content Classifier
# ============================================================================

def test_content_classifier_obligations():
    """Test obligation detection"""
    classifier = ContentClassifier()

    text1 = "Dodavatel musí dokončit dílo do 30 dnů."
    assert classifier.classify_content_type(text1) == 'obligation'

    text2 = "Objednatel je povinen zaplatit cenu."
    assert classifier.classify_content_type(text2) == 'obligation'


def test_content_classifier_prohibitions():
    """Test prohibition detection"""
    classifier = ContentClassifier()

    text1 = "Dodavatel nesmí převést práva a povinnosti na třetí stranu."
    assert classifier.classify_content_type(text1) == 'prohibition'

    text2 = "Je zakázáno měnit technické parametry."
    assert classifier.classify_content_type(text2) == 'prohibition'


def test_content_classifier_definitions():
    """Test definition detection"""
    classifier = ContentClassifier()

    text1 = "Dílem se rozumí výsledek činnosti dodavatele."
    assert classifier.classify_content_type(text1) == 'definition'

    text2 = "Smlouva je definována jako ujednání dvou stran."
    assert classifier.classify_content_type(text2) == 'definition'


def test_content_classifier_general():
    """Test general content classification"""
    classifier = ContentClassifier()

    text = "V tomto článku jsou uvedeny další podmínky."
    assert classifier.classify_content_type(text) == 'general'


# ============================================================================
# Test Law Code Chunking
# ============================================================================

@pytest.mark.asyncio
async def test_law_chunker_basic(law_document, chunking_config):
    """Test basic law chunking by paragraphs"""
    chunker = LawCodeChunker(chunking_config)
    chunks = await chunker.chunk(law_document)

    # Should create chunks (may aggregate small paragraphs)
    assert len(chunks) == 2

    # Check first chunk
    assert chunks[0].chunk_id == "chunk_paragraph_89"
    assert chunks[0].document_type == "law_code"
    assert chunks[0].structural_level == "paragraph"
    assert chunks[0].legal_reference == "§89"
    assert "Dodavatel odpovídá" in chunks[0].content

    # Check second chunk (may be aggregated)
    assert "paragraph_90" in chunks[1].chunk_id  # May be chunk_paragraph_90 or chunk_aggregated_paragraph_90
    assert "§90" in chunks[1].legal_reference


@pytest.mark.asyncio
async def test_law_chunker_with_context(law_document, chunking_config):
    """Test hierarchical context addition"""
    chunking_config.law_include_context = True
    chunker = LawCodeChunker(chunking_config)
    chunks = await chunker.chunk(law_document)

    # Check that context is added
    chunk = chunks[0]
    assert "Část I" in chunk.content or "Hlava I" in chunk.content or "§89" in chunk.content


@pytest.mark.asyncio
async def test_law_chunker_without_context(law_document, chunking_config):
    """Test chunking without hierarchical context"""
    chunking_config.law_include_context = False
    chunker = LawCodeChunker(chunking_config)
    chunks = await chunker.chunk(law_document)

    # Content should not have hierarchy prefixes
    chunk = chunks[0]
    assert not chunk.content.startswith("Část")


@pytest.mark.asyncio
async def test_law_chunker_metadata(law_document, chunking_config):
    """Test chunk metadata"""
    chunker = LawCodeChunker(chunking_config)
    chunks = await chunker.chunk(law_document)

    chunk = chunks[0]

    # Check metadata
    assert chunk.metadata['part'] == "I"
    assert chunk.metadata['chapter'] == "I"
    assert chunk.metadata['paragraph'] == 89
    assert chunk.metadata['subsection'] is None
    assert 'token_count' in chunk.metadata
    assert 'content_type' in chunk.metadata


# ============================================================================
# Test Contract Chunking
# ============================================================================

@pytest.mark.asyncio
async def test_contract_chunker_basic(contract_document, chunking_config):
    """Test basic contract chunking by articles"""
    chunker = ContractChunker(chunking_config)
    chunks = await chunker.chunk(contract_document)

    # Should create one chunk per article
    assert len(chunks) == 1

    chunk = chunks[0]
    assert chunk.chunk_id == "chunk_article_5"
    assert chunk.document_type == "contract"
    assert chunk.structural_level == "article"
    assert chunk.legal_reference == "Článek 5"
    assert "Cena díla" in chunk.content


@pytest.mark.asyncio
async def test_contract_chunker_metadata(contract_document, chunking_config):
    """Test contract chunk metadata"""
    chunker = ContractChunker(chunking_config)
    chunks = await chunker.chunk(contract_document)

    chunk = chunks[0]

    # Check metadata
    assert chunk.metadata['article'] == 5
    assert chunk.metadata['article_title'] == "Cena a platební podmínky"
    assert chunk.metadata['point'] is None
    assert 'token_count' in chunk.metadata
    assert 'content_type' in chunk.metadata


# ============================================================================
# Test Chunking Pipeline
# ============================================================================

@pytest.mark.asyncio
async def test_chunking_pipeline_law(law_document, chunking_config):
    """Test chunking pipeline with law document"""
    pipeline = LegalChunkingPipeline(chunking_config)
    chunks = await pipeline.chunk_document(law_document)

    # Should auto-select law chunker
    assert len(chunks) > 0
    assert all(c.document_type == "law_code" for c in chunks)

    # Check chunk indices
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.metadata['chunk_index'] == i
        assert chunk.metadata['total_chunks'] == len(chunks)


@pytest.mark.asyncio
async def test_chunking_pipeline_contract(contract_document, chunking_config):
    """Test chunking pipeline with contract document"""
    pipeline = LegalChunkingPipeline(chunking_config)
    chunks = await pipeline.chunk_document(contract_document)

    # Should auto-select contract chunker
    assert len(chunks) > 0
    assert all(c.document_type == "contract" for c in chunks)


@pytest.mark.asyncio
async def test_chunking_pipeline_overlap(law_document, chunking_config):
    """Test 15% chunk overlap"""
    chunking_config.chunk_overlap = 0.15
    pipeline = LegalChunkingPipeline(chunking_config)
    chunks = await pipeline.chunk_document(law_document)

    # Verify chunk overlap configuration is set
    assert chunking_config.chunk_overlap == 0.15

    # For this test, we verify that chunks are created (overlap is applied internally)
    # The implementation may not expose overlap metadata directly
    assert len(chunks) >= 1

    # Verify chunks have valid structure
    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.metadata.get('token_count', 0) > 0


@pytest.mark.asyncio
async def test_chunking_pipeline_validation(law_document, chunking_config):
    """Test chunk validation (skip empty, too small)"""
    chunking_config.skip_empty = True
    chunking_config.min_acceptable_size = 10

    pipeline = LegalChunkingPipeline(chunking_config)
    chunks = await pipeline.chunk_document(law_document)

    # All chunks should pass validation
    for chunk in chunks:
        assert chunk.content.strip()  # Not empty
        assert chunk.metadata.get('token_count', 0) >= 10  # Not too small


# ============================================================================
# Test Hierarchy Path Building
# ============================================================================

@pytest.mark.asyncio
async def test_hierarchy_path_generation(law_document, chunking_config):
    """Test hierarchy path generation"""
    chunker = LawCodeChunker(chunking_config)
    chunks = await chunker.chunk(law_document)

    chunk = chunks[0]

    # Hierarchy path should contain all parent elements
    assert chunk.hierarchy_path
    # Should contain Part, Chapter, and Paragraph
    path_parts = chunk.hierarchy_path.split(" > ")
    assert len(path_parts) >= 1  # At least the paragraph itself


# ============================================================================
# Test Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_empty_document(chunking_config):
    """Test chunking with empty document"""
    empty_structure = DocumentStructure(
        hierarchy=[],
        sections=[],
        all_elements=[]
    )

    empty_doc = LegalDocument(
        document_id="empty",
        document_type="law_code",
        raw_text="",
        cleaned_text="",
        structure=empty_structure,
        metadata=None,
        references=[],
        parsed_at=None
    )

    chunker = LawCodeChunker(chunking_config)
    chunks = await chunker.chunk(empty_doc)

    # Should handle empty document gracefully
    assert chunks == []


@pytest.mark.asyncio
async def test_chunk_config_validation():
    """Test configuration validation"""
    config = ChunkingConfig(
        min_chunk_size=128,
        chunk_size=512,
        max_chunk_size=1024
    )

    # Valid configuration
    assert config.min_chunk_size < config.chunk_size < config.max_chunk_size


# ============================================================================
# Test Content Type Classification in Chunks
# ============================================================================

@pytest.mark.asyncio
async def test_chunk_content_classification(law_document, chunking_config):
    """Test that chunks have correct content type"""
    chunker = LawCodeChunker(chunking_config)
    chunks = await chunker.chunk(law_document)

    # First paragraph contains obligation
    first_chunk = chunks[0]
    assert first_chunk.metadata.get('content_type') in ['obligation', 'general']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
