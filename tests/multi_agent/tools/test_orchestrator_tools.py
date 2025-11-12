"""
Tests for orchestrator-specific tools.

These tools provide meta-information to help the orchestrator make routing decisions.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.multi_agent.tools.orchestrator_tools import (
    OrchestratorTools,
    create_orchestrator_tools
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Mock vector store with document metadata."""
    store = Mock()
    store.metadata_store = {
        "doc1": {"filename": "GDPR.pdf", "title": "General Data Protection Regulation"},
        "doc2": {"filename": "ISO27001.pdf", "title": "ISO 27001 Standard"},
        "doc3": {"filename": "CCPA.pdf", "title": "California Consumer Privacy Act"}
    }
    return store


@pytest.fixture
def mock_agent_registry():
    """Mock agent registry with specialized agents."""
    registry = Mock()

    # Mock agent instances
    extractor = Mock()
    extractor.config = Mock()
    extractor.config.role = Mock(value="extract")
    extractor.config.tools = {"hierarchical_search", "similarity_search"}

    classifier = Mock()
    classifier.config = Mock()
    classifier.config.role = Mock(value="classify")
    classifier.config.tools = {"get_document_info", "list_available_documents"}

    compliance = Mock()
    compliance.config = Mock()
    compliance.config.role = Mock(value="verify")
    compliance.config.tools = {"hierarchical_search", "assess_confidence"}

    orchestrator = Mock()
    orchestrator.config = Mock()
    orchestrator.config.role = Mock(value="coordinate")
    orchestrator.config.tools = {"list_available_documents", "list_available_agents"}

    registry._agent_instances = {
        "extractor": extractor,
        "classifier": classifier,
        "compliance": compliance,
        "orchestrator": orchestrator
    }

    return registry


@pytest.fixture
def orchestrator_tools(mock_vector_store, mock_agent_registry):
    """Create OrchestratorTools instance with mocked dependencies."""
    return OrchestratorTools(
        vector_store=mock_vector_store,
        agent_registry=mock_agent_registry
    )


# ============================================================================
# list_available_documents Tests
# ============================================================================

def test_list_documents_returns_all_documents(orchestrator_tools, mock_vector_store):
    """Should return list of all documents from vector store."""
    result = orchestrator_tools.list_available_documents()

    assert result["count"] == 3
    assert len(result["documents"]) == 3

    # Verify document structure
    doc_ids = [doc["id"] for doc in result["documents"]]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    assert "doc3" in doc_ids

    # Verify filenames extracted
    doc1 = next(d for d in result["documents"] if d["id"] == "doc1")
    assert doc1["filename"] == "GDPR.pdf"


def test_list_documents_includes_message(orchestrator_tools):
    """Should include descriptive message in result."""
    result = orchestrator_tools.list_available_documents()

    assert "message" in result
    assert "3 documents" in result["message"]


def test_list_documents_handles_empty_store(mock_agent_registry):
    """Should handle empty vector store gracefully."""
    empty_store = Mock()
    empty_store.metadata_store = {}

    tools = OrchestratorTools(vector_store=empty_store, agent_registry=mock_agent_registry)
    result = tools.list_available_documents()

    assert result["count"] == 0
    assert result["documents"] == []
    assert "message" in result


def test_list_documents_handles_no_vector_store(mock_agent_registry):
    """Should handle missing vector store gracefully."""
    tools = OrchestratorTools(vector_store=None, agent_registry=mock_agent_registry)
    result = tools.list_available_documents()

    assert result["count"] == 0
    assert result["documents"] == []
    assert "not initialized" in result["message"]


def test_list_documents_limits_large_stores(mock_agent_registry):
    """Should limit results to 50 documents for performance."""
    large_store = Mock()
    # Create 100 documents
    large_store.metadata_store = {
        f"doc{i}": {"filename": f"doc{i}.pdf"}
        for i in range(100)
    }

    tools = OrchestratorTools(vector_store=large_store, agent_registry=mock_agent_registry)
    result = tools.list_available_documents()

    # Should be limited to 50
    assert result["count"] == 50
    assert len(result["documents"]) == 50


def test_list_documents_handles_malformed_metadata(mock_agent_registry):
    """Should handle malformed metadata gracefully."""
    broken_store = Mock()
    broken_store.metadata_store = {
        "doc1": {"filename": "valid.pdf"},
        "doc2": None,  # Malformed: None instead of dict
        "doc3": "string",  # Malformed: string instead of dict
        "doc4": {"filename": "valid2.pdf"}
    }

    tools = OrchestratorTools(vector_store=broken_store, agent_registry=mock_agent_registry)
    result = tools.list_available_documents()

    # Should include all documents, handling malformed ones
    assert result["count"] >= 2  # At least valid ones
    assert len(result["documents"]) >= 2


def test_list_documents_handles_exception(mock_agent_registry):
    """Should handle exceptions gracefully and return error info."""
    broken_store = Mock()
    broken_store.metadata_store = Mock(side_effect=Exception("Database connection failed"))

    tools = OrchestratorTools(vector_store=broken_store, agent_registry=mock_agent_registry)
    result = tools.list_available_documents()

    assert result["count"] == 0
    assert result["documents"] == []
    assert "error" in result
    assert "Database connection failed" in result["error"]


# ============================================================================
# list_available_agents Tests
# ============================================================================

def test_list_agents_returns_all_agents(orchestrator_tools):
    """Should return list of all agents from registry."""
    result = orchestrator_tools.list_available_agents()

    # Should return 3 agents (excluding orchestrator itself)
    assert result["count"] == 3
    assert len(result["agents"]) == 3

    # Verify agent names
    agent_names = [agent["name"] for agent in result["agents"]]
    assert "extractor" in agent_names
    assert "classifier" in agent_names
    assert "compliance" in agent_names
    assert "orchestrator" not in agent_names  # Should exclude self


def test_list_agents_includes_capabilities(orchestrator_tools):
    """Should include agent roles and tools."""
    result = orchestrator_tools.list_available_agents()

    # Find extractor agent
    extractor = next(a for a in result["agents"] if a["name"] == "extractor")

    assert extractor["role"] == "extract"
    assert "hierarchical_search" in extractor["tools"]
    assert "similarity_search" in extractor["tools"]


def test_list_agents_includes_message(orchestrator_tools):
    """Should include descriptive message in result."""
    result = orchestrator_tools.list_available_agents()

    assert "message" in result
    assert "3 available agents" in result["message"]


def test_list_agents_handles_no_registry(mock_vector_store):
    """Should handle missing agent registry gracefully."""
    tools = OrchestratorTools(vector_store=mock_vector_store, agent_registry=None)
    result = tools.list_available_agents()

    assert result["count"] == 0
    assert result["agents"] == []
    assert "not initialized" in result["message"]


def test_list_agents_handles_empty_registry(mock_vector_store):
    """Should handle empty agent registry gracefully."""
    empty_registry = Mock()
    empty_registry._agent_instances = {}

    tools = OrchestratorTools(vector_store=mock_vector_store, agent_registry=empty_registry)
    result = tools.list_available_agents()

    assert result["count"] == 0
    assert result["agents"] == []


def test_list_agents_handles_malformed_agent(mock_vector_store):
    """Should handle agents with missing attributes gracefully."""
    broken_registry = Mock()

    # Agent with missing role attribute
    broken_agent = Mock()
    broken_agent.config = Mock(spec=[])  # No role or tools attributes

    broken_registry._agent_instances = {
        "broken_agent": broken_agent
    }

    tools = OrchestratorTools(vector_store=mock_vector_store, agent_registry=broken_registry)
    result = tools.list_available_agents()

    # Should handle gracefully
    assert result["count"] == 1
    agent = result["agents"][0]
    assert agent["name"] == "broken_agent"
    assert agent["role"] == "unknown"
    assert agent["tools"] == []


def test_list_agents_handles_exception(mock_vector_store):
    """Should handle exceptions gracefully and return error info."""
    broken_registry = Mock()
    broken_registry._agent_instances = Mock(side_effect=Exception("Registry initialization failed"))

    tools = OrchestratorTools(vector_store=mock_vector_store, agent_registry=broken_registry)
    result = tools.list_available_agents()

    assert result["count"] == 0
    assert result["agents"] == []
    assert "error" in result
    assert "Registry initialization failed" in result["error"]


def test_list_agents_excludes_orchestrator(orchestrator_tools):
    """CRITICAL: Should not return orchestrator agent in list (avoid self-reference)."""
    result = orchestrator_tools.list_available_agents()

    agent_names = [agent["name"] for agent in result["agents"]]
    assert "orchestrator" not in agent_names


# ============================================================================
# create_orchestrator_tools Tests
# ============================================================================

def test_create_tools_returns_schemas_and_instance(mock_vector_store, mock_agent_registry):
    """Should return both tool schemas and tools instance."""
    schemas, instance = create_orchestrator_tools(
        vector_store=mock_vector_store,
        agent_registry=mock_agent_registry
    )

    # Verify schemas
    assert isinstance(schemas, list)
    assert len(schemas) == 2

    # Verify instance
    assert isinstance(instance, OrchestratorTools)
    assert instance.vector_store == mock_vector_store
    assert instance.agent_registry == mock_agent_registry


def test_create_tools_schemas_have_correct_structure():
    """Tool schemas should follow LLM tool calling format."""
    schemas, _ = create_orchestrator_tools()

    for schema in schemas:
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema

        # Verify input_schema structure
        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema
        assert "required" in input_schema


def test_create_tools_includes_list_documents_schema():
    """Should include list_available_documents tool schema."""
    schemas, _ = create_orchestrator_tools()

    doc_schema = next(s for s in schemas if s["name"] == "list_available_documents")

    assert "documents" in doc_schema["description"].lower()
    assert doc_schema["input_schema"]["properties"] == {}  # No parameters
    assert doc_schema["input_schema"]["required"] == []


def test_create_tools_includes_list_agents_schema():
    """Should include list_available_agents tool schema."""
    schemas, _ = create_orchestrator_tools()

    agent_schema = next(s for s in schemas if s["name"] == "list_available_agents")

    assert "agents" in agent_schema["description"].lower()
    assert agent_schema["input_schema"]["properties"] == {}  # No parameters
    assert agent_schema["input_schema"]["required"] == []


def test_create_tools_works_without_dependencies():
    """Should work even without vector_store or agent_registry."""
    schemas, instance = create_orchestrator_tools()

    assert len(schemas) == 2
    assert instance.vector_store is None
    assert instance.agent_registry is None

    # Should still be callable (with graceful handling)
    result = instance.list_available_documents()
    assert result["count"] == 0


# ============================================================================
# Integration Tests
# ============================================================================

def test_tools_integration_workflow(mock_vector_store, mock_agent_registry):
    """End-to-end test of orchestrator tools workflow."""
    # Create tools
    schemas, tools = create_orchestrator_tools(
        vector_store=mock_vector_store,
        agent_registry=mock_agent_registry
    )

    # Simulate LLM calling list_available_documents
    doc_result = tools.list_available_documents()
    assert doc_result["count"] == 3
    assert any(d["id"] == "doc1" for d in doc_result["documents"])

    # Simulate LLM calling list_available_agents
    agent_result = tools.list_available_agents()
    assert agent_result["count"] == 3
    assert any(a["name"] == "extractor" for a in agent_result["agents"])


def test_tools_support_orchestrator_routing_decision():
    """Tools should provide information needed for orchestrator routing."""
    # Simulate orchestrator query: "Compare GDPR and CCPA requirements"

    # Mock vector store with GDPR and CCPA
    store = Mock()
    store.metadata_store = {
        "gdpr": {"filename": "GDPR.pdf"},
        "ccpa": {"filename": "CCPA.pdf"}
    }

    # Mock agent registry with compliance agent
    registry = Mock()
    compliance = Mock()
    compliance.config = Mock()
    compliance.config.role = Mock(value="verify")
    compliance.config.tools = {"hierarchical_search", "assess_confidence"}

    registry._agent_instances = {"compliance": compliance}

    tools = OrchestratorTools(vector_store=store, agent_registry=registry)

    # Orchestrator calls tools to make decision
    docs = tools.list_available_documents()
    agents = tools.list_available_agents()

    # Should have info to route to compliance agent
    assert docs["count"] == 2
    assert any("gdpr" in d["id"] for d in docs["documents"])
    assert any("ccpa" in d["id"] for d in docs["documents"])

    assert agents["count"] == 1
    assert agents["agents"][0]["name"] == "compliance"
    assert "hierarchical_search" in agents["agents"][0]["tools"]


def test_tools_handle_system_degradation():
    """Tools should degrade gracefully when dependencies fail."""
    # Vector store fails
    broken_store = Mock()
    broken_store.metadata_store = Mock(side_effect=Exception("Store failed"))

    # Agent registry works
    registry = Mock()
    registry._agent_instances = {}

    tools = OrchestratorTools(vector_store=broken_store, agent_registry=registry)

    # list_available_documents should fail gracefully
    doc_result = tools.list_available_documents()
    assert "error" in doc_result
    assert doc_result["count"] == 0

    # list_available_agents should still work
    agent_result = tools.list_available_agents()
    assert "error" not in agent_result
    assert agent_result["count"] == 0


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_tools_handle_unicode_in_filenames():
    """Should handle unicode characters in document filenames."""
    store = Mock()
    store.metadata_store = {
        "doc1": {"filename": "Směrnice GDPR.pdf"},
        "doc2": {"filename": "中文文档.pdf"},
        "doc3": {"filename": "Документ.pdf"}
    }

    tools = OrchestratorTools(vector_store=store, agent_registry=None)
    result = tools.list_available_documents()

    assert result["count"] == 3
    filenames = [d["filename"] for d in result["documents"]]
    assert "Směrnice GDPR.pdf" in filenames
    assert "中文文档.pdf" in filenames
    assert "Документ.pdf" in filenames


def test_tools_handle_very_long_agent_names():
    """Should handle very long agent names."""
    registry = Mock()

    long_name_agent = Mock()
    long_name_agent.config = Mock()
    long_name_agent.config.role = Mock(value="analyze")
    long_name_agent.config.tools = {"tool1"}

    registry._agent_instances = {
        "very_long_agent_name_that_describes_complex_functionality_in_detail": long_name_agent
    }

    tools = OrchestratorTools(vector_store=None, agent_registry=registry)
    result = tools.list_available_agents()

    assert result["count"] == 1
    assert result["agents"][0]["name"] == "very_long_agent_name_that_describes_complex_functionality_in_detail"


def test_tools_handle_agent_with_many_tools():
    """Should handle agents with large tool sets."""
    registry = Mock()

    super_agent = Mock()
    super_agent.config = Mock()
    super_agent.config.role = Mock(value="super_analyze")
    super_agent.config.tools = {f"tool_{i}" for i in range(100)}  # 100 tools

    registry._agent_instances = {"super_agent": super_agent}

    tools = OrchestratorTools(vector_store=None, agent_registry=registry)
    result = tools.list_available_agents()

    assert result["count"] == 1
    assert len(result["agents"][0]["tools"]) == 100
