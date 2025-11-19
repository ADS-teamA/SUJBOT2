"""
Tests for DefinitionAlignerTool (TIER 3).

The DefinitionAlignerTool aligns legal terminology across documents using
knowledge graph + pgvector semantic search to resolve term mismatches.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agent.tools.tier3_analysis import DefinitionAlignerTool, DefinitionAlignerInput
from src.agent.tools.base import ToolResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    embedder = Mock()
    embedder.embed_texts.return_value = [[0.1] * 768]  # 768D embedding (text-embedding-3-large)
    embedder.model_name = "text-embedding-3-large"
    embedder.dimensions = 768
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    vs = Mock()
    # Default empty hierarchical search
    vs.hierarchical_search.return_value = {"layer1": [], "layer2": [], "layer3": []}
    return vs


@pytest.fixture
def mock_knowledge_graph():
    """Create mock knowledge graph with legal terminology."""
    kg = Mock()

    # Mock entities: LEGAL_TERM and DEFINITION entities
    term_entity = Mock()
    term_entity.id = "term_data_controller"
    term_entity.type = "LEGAL_TERM"
    term_entity.value = "Data Controller"
    term_entity.confidence = 0.95

    definition_entity = Mock()
    definition_entity.id = "def_data_controller_gdpr"
    definition_entity.type = "DEFINITION"
    definition_entity.value = "Natural or legal person who determines purposes and means of processing"
    definition_entity.confidence = 0.90
    definition_entity.metadata = {
        "source_provision": "GDPR Article 4(7)",
        "document_id": "gdpr.pdf"
    }

    kg.entities = {
        "term_data_controller": term_entity,
        "def_data_controller_gdpr": definition_entity
    }

    # Mock DEFINITION_OF relationship
    rel = Mock()
    rel.id = "rel_def_of_1"
    rel.type = "DEFINITION_OF"
    rel.source_entity_id = "def_data_controller_gdpr"
    rel.target_entity_id = "term_data_controller"
    rel.confidence = 0.90

    kg.relationships = {"rel_def_of_1": rel}
    kg.get_outgoing_relationships.return_value = [rel]

    return kg


@pytest.fixture
def mock_context_assembler():
    """Create mock context assembler."""
    assembler = Mock()
    return assembler


@pytest.fixture
def definition_aligner_tool(mock_vector_store, mock_embedder, mock_knowledge_graph, mock_context_assembler):
    """Create DefinitionAlignerTool instance."""
    return DefinitionAlignerTool(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        knowledge_graph=mock_knowledge_graph,
        context_assembler=mock_context_assembler,
        config=Mock()
    )


# ============================================================================
# Input Sanitization Tests (Critical Security Fix)
# ============================================================================

def test_sanitizes_special_characters_in_term(definition_aligner_tool):
    """Should sanitize special characters to prevent injection attacks."""
    # Attempt SQL injection via term parameter
    malicious_term = "Client'; DROP TABLE entities; --"

    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=[]):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', return_value=[]):
            result = definition_aligner_tool.execute(term=malicious_term)

    # Should succeed with sanitized term (special chars removed except - and _)
    assert result.success
    # Verify sanitized term was used (hyphens -- preserved by sanitizer)
    assert result.data["query_term"] == "Client DROP TABLE entities --"


def test_rejects_term_with_only_special_characters(definition_aligner_tool):
    """Should reject terms that become empty after sanitization."""
    malicious_term = "@#$%^&*()"  # Only special chars (no hyphens/underscores)

    result = definition_aligner_tool.execute(term=malicious_term)

    # Should fail validation
    assert result.success is False
    assert "Invalid term" in result.error
    assert "only special characters" in result.error


def test_preserves_original_term_for_transparency(definition_aligner_tool):
    """Should return both sanitized and original term for transparency."""
    original_term = "Data Controller (GDPR)"

    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=[]):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', return_value=[]):
            result = definition_aligner_tool.execute(term=original_term)

    assert result.success
    assert result.data["original_term"] == original_term
    assert result.data["query_term"] == "Data Controller GDPR"  # Parentheses removed


# ============================================================================
# Knowledge Graph Search Tests
# ============================================================================

def test_finds_definitions_from_knowledge_graph(definition_aligner_tool):
    """Should find term definitions from knowledge graph via DEFINITION_OF relationships."""
    kg_definitions = [{
        "term": "Data Controller",
        "definition": "Natural or legal person who determines purposes and means",
        "document_id": "gdpr.pdf",
        "breadcrumb": "GDPR > Article 4 > Definitions",
        "confidence": 0.90
    }]

    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=kg_definitions):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', return_value=[]):
            result = definition_aligner_tool.execute(term="Data Controller")

    assert result.success
    assert len(result.data["alignments"]) == 1
    alignment = result.data["alignments"][0]
    assert alignment["source"] == "knowledge_graph"
    assert alignment["term"] == "Data Controller"
    assert alignment["alignment_type"] == "exact_match"


def test_filters_kg_results_by_reference_laws(definition_aligner_tool):
    """Should filter KG results to only specified reference laws."""
    with patch.object(definition_aligner_tool, '_search_kg_definitions') as mock_kg_search:
        mock_kg_search.return_value = []
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', return_value=[]):
            result = definition_aligner_tool.execute(
                term="Consumer",
                reference_laws=["gdpr.pdf", "consumer_rights_act.pdf"]
            )

    # Verify reference_laws passed to KG search
    mock_kg_search.assert_called_once()
    args = mock_kg_search.call_args
    assert args[0][1] == ["gdpr.pdf", "consumer_rights_act.pdf"]


# ============================================================================
# Semantic Search Tests
# ============================================================================

def test_finds_semantically_similar_terms(definition_aligner_tool):
    """Should find semantically similar terms via pgvector embedding search."""
    semantic_matches = [{
        "term": "Data Custodian",
        "definition": "Entity responsible for data management",
        "document_id": "contract.pdf",
        "breadcrumb": "Contract > Definitions > Section 2.1",
        "similarity_score": 0.82
    }]

    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=[]):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', return_value=semantic_matches):
            result = definition_aligner_tool.execute(term="Data Controller", include_related_terms=True)

    assert result.success
    assert len(result.data["alignments"]) == 1
    alignment = result.data["alignments"][0]
    assert alignment["source"] == "semantic_search"
    assert alignment["term"] == "Data Custodian"
    assert alignment["alignment_type"] == "semantic_similar"
    assert alignment["confidence"] == 0.82


def test_skips_semantic_search_when_disabled(definition_aligner_tool):
    """Should skip semantic search when include_related_terms=False."""
    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=[]):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions') as mock_semantic:
            mock_semantic.return_value = []
            result = definition_aligner_tool.execute(term="Test", include_related_terms=False)

    # Semantic search should NOT be called
    mock_semantic.assert_not_called()


def test_respects_similarity_threshold(definition_aligner_tool):
    """Should pass similarity threshold to semantic search."""
    custom_threshold = 0.85

    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=[]):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions') as mock_semantic:
            mock_semantic.return_value = []
            result = definition_aligner_tool.execute(
                term="Test",
                similarity_threshold=custom_threshold
            )

    # Verify threshold passed correctly
    mock_semantic.assert_called_once()
    args = mock_semantic.call_args
    assert args[0][3] == custom_threshold  # 4th positional argument


# ============================================================================
# Deduplication Tests
# ============================================================================

def test_deduplicates_kg_and_semantic_results(definition_aligner_tool):
    """Should avoid duplicates when term appears in both KG and semantic search."""
    # Same term found in both sources
    kg_results = [{
        "term": "Client",
        "definition": "Person receiving services",
        "document_id": "contract.pdf",
        "confidence": 0.90
    }]

    semantic_results = [{
        "term": "Client",
        "definition": "Person receiving services",
        "document_id": "contract.pdf",
        "similarity_score": 0.85
    }]

    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=kg_results):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', return_value=semantic_results):
            result = definition_aligner_tool.execute(term="Client")

    # Should only have 1 alignment (duplicate removed)
    assert result.success
    assert len(result.data["alignments"]) == 1
    # KG result should be preferred (checked first)
    assert result.data["alignments"][0]["source"] == "knowledge_graph"


# ============================================================================
# Sorting and Confidence Tests
# ============================================================================

def test_sorts_alignments_by_confidence_descending(definition_aligner_tool):
    """Should sort alignments by confidence score (highest first)."""
    kg_results = [
        {"term": "Low", "definition": "Low conf", "document_id": "doc1", "confidence": 0.50},
        {"term": "High", "definition": "High conf", "document_id": "doc2", "confidence": 0.95},
        {"term": "Medium", "definition": "Med conf", "document_id": "doc3", "confidence": 0.75}
    ]

    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=kg_results):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', return_value=[]):
            result = definition_aligner_tool.execute(term="Test")

    # Verify descending order
    assert result.success
    confidences = [a["confidence"] for a in result.data["alignments"]]
    assert confidences == [0.95, 0.75, 0.50]
    assert result.data["alignments"][0]["term"] == "High"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_handles_empty_search_results(definition_aligner_tool):
    """Should return success with empty alignments when no matches found."""
    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=[]):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', return_value=[]):
            result = definition_aligner_tool.execute(term="NonexistentTerm")

    assert result.success
    assert result.data["alignments"] == []
    # Check summary contains indication of no results
    assert "No definitions found" in result.data["summary"]


def test_handles_kg_search_exception(definition_aligner_tool):
    """Should return error when KG search fails (fail-fast for critical tool)."""
    with patch.object(definition_aligner_tool, '_search_kg_definitions', side_effect=Exception("KG unavailable")):
        result = definition_aligner_tool.execute(term="Test")

    # Tool fails immediately on exception (fail-fast behavior)
    assert result.success is False
    assert "KG unavailable" in result.error


def test_handles_semantic_search_exception(definition_aligner_tool):
    """Should return error when semantic search fails (fail-fast for critical tool)."""
    kg_results = [{
        "term": "KG Term",
        "definition": "From knowledge graph",
        "document_id": "law.pdf",
        "confidence": 0.90
    }]

    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=kg_results):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions', side_effect=Exception("Semantic unavailable")):
            result = definition_aligner_tool.execute(term="Test")

    # Tool fails immediately on exception (fail-fast behavior)
    assert result.success is False
    assert "Semantic unavailable" in result.error


def test_context_document_filtering(definition_aligner_tool):
    """Should filter results by context_document_id when provided."""
    with patch.object(definition_aligner_tool, '_search_kg_definitions', return_value=[]):
        with patch.object(definition_aligner_tool, '_search_semantic_definitions') as mock_semantic:
            mock_semantic.return_value = []
            result = definition_aligner_tool.execute(
                term="Test",
                context_document_id="specific_doc.pdf"
            )

    # Verify context_document_id passed to semantic search
    mock_semantic.assert_called_once()
    args = mock_semantic.call_args
    assert args[0][1] == "specific_doc.pdf"  # 2nd positional argument


# ============================================================================
# Input Schema Validation Tests
# ============================================================================

def test_input_schema_validates_required_fields():
    """Input schema should require 'term' field."""
    # Valid input
    valid_input = DefinitionAlignerInput(term="Data Controller")
    assert valid_input.term == "Data Controller"

    # Invalid: missing term (should fail pydantic validation)
    with pytest.raises(Exception):  # Pydantic ValidationError
        DefinitionAlignerInput()


def test_input_schema_has_correct_defaults():
    """Input schema should have correct default values."""
    input_data = DefinitionAlignerInput(term="Test")

    assert input_data.context_document_id is None
    assert input_data.reference_laws is None
    assert input_data.similarity_threshold == 0.75  # Default threshold
    assert input_data.include_related_terms is True  # Default enabled
