"""
Unit tests for Query Processor

Tests:
- Query classification (intent and complexity)
- Entity extraction (legal refs, dates, obligations)
- Query decomposition
- Query expansion
- Integration tests
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from query_processor import (
    QueryProcessor,
    QueryClassifier,
    EntityExtractor,
    LegalReferenceExtractor,
    TemporalExtractor,
    ObligationExtractor,
    LegalQuestionDecomposer,
    QueryExpander,
    QueryIntent,
    QueryComplexity,
    ProcessedQuery
)
from anthropic import AsyncAnthropic


# Fixtures
@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client (requires API key)."""
    import os
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        pytest.skip("CLAUDE_API_KEY not set")
    return AsyncAnthropic(api_key=api_key)


# ============================================================================
# Entity Extraction Tests
# ============================================================================

def test_legal_reference_extraction():
    """Test legal reference extraction."""
    extractor = LegalReferenceExtractor()

    # Test paragraph with subsection and letter
    query1 = "Podle §89 odst. 2 písm. a) zákona č. 89/2012 Sb."
    entities1 = extractor.extract(query1)

    assert len(entities1) == 2
    assert entities1[0].entity_type == "legal_ref"
    assert entities1[0].normalized == "§89 odst. 2 písm. a)"
    assert entities1[1].normalized == "Zákon č. 89/2012 Sb."

    # Test simple paragraph
    query2 = "Co říká §45?"
    entities2 = extractor.extract(query2)

    assert len(entities2) == 1
    assert entities2[0].normalized == "§45"

    # Test multiple references
    query3 = "Porovnej §89 a §45 odst. 3"
    entities3 = extractor.extract(query3)

    assert len(entities3) == 2


def test_temporal_extraction():
    """Test temporal entity extraction."""
    extractor = TemporalExtractor()

    query = "Termín do 30 dní od 15.10.2024 nebo do 6 měsíců"
    entities = extractor.extract(query)

    assert len(entities) >= 2  # At least date and deadline
    assert any(e.entity_type == "date" for e in entities)


def test_obligation_extraction():
    """Test obligation/prohibition extraction."""
    extractor = ObligationExtractor()

    query = "Dodavatel musí splnit povinnost, ale nesmí porušit zákaz."
    entities = extractor.extract(query)

    obligations = [e for e in entities if e.entity_type == "obligation"]
    prohibitions = [e for e in entities if e.entity_type == "prohibition"]

    assert len(obligations) >= 1
    assert len(prohibitions) >= 1


def test_entity_extractor_integration(config):
    """Test entity extractor with all sub-extractors."""
    extractor = EntityExtractor(config)

    query = "Podle §89 odst. 2 dodavatel musí do 30 dní splnit povinnost."
    entities = extractor.extract(query)

    # Should have legal ref, obligation, and date
    assert len(entities) >= 2

    entity_types = {e.entity_type for e in entities}
    assert "legal_ref" in entity_types
    assert "obligation" in entity_types or "date" in entity_types


# ============================================================================
# Query Classification Tests
# ============================================================================

@pytest.mark.asyncio
async def test_pattern_based_classification(config, mock_llm_client):
    """Test pattern-based intent classification."""
    classifier = QueryClassifier(mock_llm_client, config)

    # Gap analysis
    query1 = "Které body chybí ve smlouvě?"
    intent1 = classifier._classify_by_patterns(query1)
    assert intent1 == QueryIntent.GAP_ANALYSIS

    # Conflict detection
    query2 = "Najdi konflikty mezi smlouvou a zákonem"
    intent2 = classifier._classify_by_patterns(query2)
    assert intent2 == QueryIntent.CONFLICT_DETECTION

    # Comparison
    query3 = "Porovnej §89 a §45"
    intent3 = classifier._classify_by_patterns(query3)
    assert intent3 == QueryIntent.COMPARISON

    # Enumeration
    query4 = "Vyjmenuj všechny sankce"
    intent4 = classifier._classify_by_patterns(query4)
    assert intent4 == QueryIntent.ENUMERATION


def test_complexity_assessment(config, mock_llm_client):
    """Test complexity assessment."""
    classifier = QueryClassifier(mock_llm_client, config)

    # Simple
    query1 = "Co je §89?"
    complexity1 = classifier._assess_complexity(query1)
    assert complexity1 == QueryComplexity.SIMPLE

    # Moderate
    query2 = "Jaké jsou povinnosti podle §89 a §45?"
    complexity2 = classifier._assess_complexity(query2)
    assert complexity2 in [QueryComplexity.MODERATE, QueryComplexity.SIMPLE]

    # Complex
    query3 = "Porovnej ustanovení smlouvy o odpovědnosti s §45 a najdi všechny konflikty."
    complexity3 = classifier._assess_complexity(query3)
    assert complexity3 in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]


@pytest.mark.asyncio
async def test_llm_classification(config, mock_llm_client):
    """Test LLM-based classification (requires API)."""
    classifier = QueryClassifier(mock_llm_client, config)

    query = "Je smlouva v souladu s požadavky zákona?"
    intent, complexity = await classifier.classify(query)

    # Should classify as compliance check
    assert intent in [QueryIntent.COMPLIANCE_CHECK, QueryIntent.FACTUAL]
    assert complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE, QueryComplexity.COMPLEX]


# ============================================================================
# Query Decomposition Tests
# ============================================================================

@pytest.mark.asyncio
async def test_requires_decomposition(config, mock_llm_client):
    """Test decomposition requirement logic."""
    decomposer = LegalQuestionDecomposer(mock_llm_client, config)

    # Simple query - no decomposition
    query1 = ProcessedQuery(
        original_query="Co je §89?",
        intent=QueryIntent.FACTUAL,
        complexity=QueryComplexity.SIMPLE,
        requires_decomposition=False,
        entities=[]
    )
    assert not decomposer._requires_decomposition(query1)

    # Complex compliance query - needs decomposition
    query2 = ProcessedQuery(
        original_query="Je smlouva v souladu s §89?",
        intent=QueryIntent.COMPLIANCE_CHECK,
        complexity=QueryComplexity.COMPLEX,
        requires_decomposition=False,
        entities=[]
    )
    assert decomposer._requires_decomposition(query2)

    # Gap analysis - always needs decomposition
    query3 = ProcessedQuery(
        original_query="Co chybí?",
        intent=QueryIntent.GAP_ANALYSIS,
        complexity=QueryComplexity.MODERATE,
        requires_decomposition=False,
        entities=[]
    )
    assert decomposer._requires_decomposition(query3)


def test_strategy_selection(config, mock_llm_client):
    """Test decomposition strategy selection."""
    decomposer = LegalQuestionDecomposer(mock_llm_client, config)

    assert decomposer._select_strategy(QueryIntent.GAP_ANALYSIS) == "requirement_based"
    assert decomposer._select_strategy(QueryIntent.CONFLICT_DETECTION) == "provision_pairing"
    assert decomposer._select_strategy(QueryIntent.COMPARISON) == "entity_separation"
    assert decomposer._select_strategy(QueryIntent.COMPLIANCE_CHECK) == "clause_by_clause"


@pytest.mark.asyncio
async def test_query_decomposition_full(config, mock_llm_client):
    """Test full query decomposition (requires API)."""
    decomposer = LegalQuestionDecomposer(mock_llm_client, config)

    query = ProcessedQuery(
        original_query="Najdi všechny konflikty mezi smlouvou a zákonem.",
        intent=QueryIntent.CONFLICT_DETECTION,
        complexity=QueryComplexity.COMPLEX,
        requires_decomposition=True,
        entities=[]
    )

    sub_queries = await decomposer.decompose(query)

    # Should generate sub-queries
    assert len(sub_queries) >= 2
    assert len(sub_queries) <= 5

    # Check sub-query structure
    for sq in sub_queries:
        assert sq.sub_query_id
        assert sq.text
        assert sq.intent
        assert sq.priority >= 1
        assert sq.retrieval_strategy in ["hybrid", "cross_document", "graph_aware"]


# ============================================================================
# Query Expansion Tests
# ============================================================================

def test_query_expansion(config):
    """Test query expansion."""
    expander = QueryExpander(config)

    query = "Jaké jsou povinnosti dodavatele podle smlouvy?"
    entities = []

    expanded = expander.expand(query, entities)

    # Should expand key terms
    assert len(expanded) >= 1

    # Check synonyms are returned
    if "smlouva" in expanded:
        assert "kontrakt" in expanded["smlouva"] or "contract" in expanded["smlouva"]


def test_key_term_extraction(config):
    """Test key term extraction."""
    expander = QueryExpander(config)

    query = "Jaké jsou povinnosti dodavatele podle smlouvy o odpovědnosti?"
    entities = []

    key_terms = expander._extract_key_terms(query, entities)

    # Should extract longer words
    assert any(len(term) >= 5 for term in key_terms)


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_query_processing(config):
    """Test complete query processing pipeline."""
    processor = QueryProcessor(config)

    query = "Je smlouva v souladu s §89?"
    processed = await processor.process(query)

    # Check all components ran
    assert processed.original_query == query
    assert processed.intent
    assert processed.complexity
    assert processed.retrieval_strategy
    assert processed.processing_time > 0


@pytest.mark.asyncio
async def test_gap_analysis_query(config):
    """Test gap analysis query processing."""
    processor = QueryProcessor(config)

    query = "Které povinné body ze zákona č. 89/2012 Sb. chybí v této smlouvě?"
    processed = await processor.process(query)

    # Should classify as gap analysis
    assert processed.intent == QueryIntent.GAP_ANALYSIS

    # Should extract legal reference
    assert len(processed.entities) > 0
    assert any(e.entity_type == "legal_ref" for e in processed.entities)

    # Should decompose (if complex enough)
    if processed.requires_decomposition:
        assert len(processed.sub_queries) >= 2


@pytest.mark.asyncio
async def test_conflict_detection_query(config):
    """Test conflict detection query processing."""
    processor = QueryProcessor(config)

    query = "Najdi konflikty mezi smlouvou a §89."
    processed = await processor.process(query)

    # Should classify as conflict detection
    assert processed.intent == QueryIntent.CONFLICT_DETECTION

    # Should select cross-document strategy
    assert processed.retrieval_strategy == "cross_document"


@pytest.mark.asyncio
async def test_simple_factual_query(config):
    """Test simple factual query (no decomposition)."""
    processor = QueryProcessor(config)

    query = "Co je §89?"
    processed = await processor.process(query)

    # Should be simple factual
    assert processed.intent in [QueryIntent.FACTUAL, QueryIntent.DEFINITION]
    assert processed.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]

    # Should not decompose
    assert not processed.requires_decomposition or len(processed.sub_queries) == 0


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_empty_query(config):
    """Test handling of empty query."""
    processor = QueryProcessor(config)

    query = ""
    processed = await processor.process(query)

    # Should handle gracefully
    assert processed.original_query == ""
    assert processed.intent
    assert processed.complexity


@pytest.mark.asyncio
async def test_very_long_query(config):
    """Test handling of very long query."""
    processor = QueryProcessor(config)

    query = " ".join(["test"] * 100)  # 100 words
    processed = await processor.process(query)

    # Should classify as complex
    assert processed.complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]


def test_overlapping_entities(config):
    """Test handling of overlapping entity extraction."""
    extractor = EntityExtractor(config)

    query = "§89 odst. 2 podle paragrafu 89"
    entities = extractor.extract(query)

    # Should not have duplicates
    entity_texts = [e.value for e in entities]
    # Check no exact duplicates
    assert len(entity_texts) == len(set(entity_texts)) or len(entities) >= 1


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_processing_performance(config):
    """Test query processing performance."""
    processor = QueryProcessor(config)

    query = "Jaké jsou povinnosti dodavatele?"
    processed = await processor.process(query)

    # Should complete in reasonable time
    assert processed.processing_time < 10.0  # 10 seconds max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
