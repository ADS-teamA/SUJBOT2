"""
Shared pytest fixtures for multi-agent tests.

Provides common test fixtures for agents, tools, providers, and storage.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List
import numpy as np


# ============================================================================
# Async Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock LLM Provider Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    def _create_response(
        text: str = "Test response",
        tool_calls: List[Dict] = None,
        stop_reason: str = "end_turn"
    ):
        response = Mock()
        response.text = text
        response.stop_reason = stop_reason

        if tool_calls:
            response.content = tool_calls
        else:
            response.content = text

        return response

    return _create_response


@pytest.fixture
def mock_provider(mock_llm_response):
    """
    Mock LLM provider for agent testing.

    IMPORTANT: create_message is AsyncMock.
    When you set return_value on AsyncMock, that value is what gets returned when awaited.

    Usage in tests:
        # Set return value for this test
        mock_provider.create_message.return_value = mock_llm_response(text="Custom response")

        # Agent internally does: response = await provider.create_message(...)
        # AsyncMock returns a coroutine that when awaited returns the return_value
    """
    provider = Mock()
    provider.create_message = AsyncMock()

    # Default return value (synchronous Mock object that is returned after await)
    provider.create_message.return_value = mock_llm_response(
        text="This is a test response from the LLM."
    )

    return provider


@pytest.fixture
def mock_anthropic_provider(mock_provider):
    """Mock Anthropic provider with specific behavior."""
    provider = mock_provider
    provider.model = "claude-sonnet-4-5-20250929"
    provider.provider_name = "anthropic"
    return provider


# ============================================================================
# Mock Tool Adapter Fixtures
# ============================================================================

@pytest.fixture
def mock_tool_result():
    """Factory for creating mock tool results."""
    def _create_result(
        success: bool = True,
        data: Any = None,
        error: str = None,
        metadata: Dict = None
    ):
        return {
            "success": success,
            "data": data or {"results": ["test result 1", "test result 2"]},
            "error": error,
            "metadata": metadata or {"api_cost_usd": 0.001}
        }

    return _create_result


@pytest.fixture
def mock_tool_adapter(mock_tool_result):
    """Mock tool adapter for testing agent tool calling."""
    adapter = Mock()
    adapter.execute = AsyncMock()

    # Default: successful tool execution
    adapter.execute.return_value = mock_tool_result(success=True)

    adapter.get_tool_schema = Mock(return_value={
        "name": "test_tool",
        "description": "A test tool",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    })

    return adapter


# ============================================================================
# Mock Vector Store Fixtures
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing retrieval."""
    store = Mock()

    # Mock hierarchical search
    store.hierarchical_search = Mock(return_value={
        "layer1": [
            {
                "chunk_id": "doc1:sec1:0",
                "text": "Test chunk 1",
                "relevance_score": 0.95,
                "document_id": "doc1",
                "metadata": {"section": "Introduction"}
            }
        ],
        "layer3": [
            {
                "chunk_id": "doc1:sec1:0",
                "text": "Test chunk 1",
                "relevance_score": 0.95,
                "document_id": "doc1",
                "metadata": {"section": "Introduction"}
            }
        ]
    })

    # Mock similarity search
    store.similarity_search = Mock(return_value=[
        {
            "chunk_id": "doc1:sec1:0",
            "text": "Test chunk",
            "relevance_score": 0.9,
            "document_id": "doc1"
        }
    ])

    # Mock document enumeration
    store.get_all_document_ids = Mock(return_value=["doc1", "doc2", "doc3"])

    # Mock metadata access
    store.metadata_layer3 = [
        {"document_id": "doc1", "title": "Test Document 1"},
        {"document_id": "doc2", "title": "Test Document 2"}
    ]

    return store


@pytest.fixture
def mock_faiss_vector_store(mock_vector_store):
    """Mock FAISS vector store with specific backend behavior."""
    store = mock_vector_store
    store.backend = "faiss"
    store.index_type = "IVFFlat"
    return store


@pytest.fixture
def mock_postgres_vector_store(mock_vector_store):
    """Mock PostgreSQL vector store with specific backend behavior."""
    store = mock_vector_store
    store.backend = "postgresql"
    store.index_type = "hnsw"
    return store


# ============================================================================
# Mock Agent Registry Fixtures
# ============================================================================

@pytest.fixture
def mock_agent_registry():
    """Mock agent registry for testing workflow building."""
    registry = Mock()

    # Mock agent retrieval
    def get_agent(name: str):
        agent = Mock()
        agent.name = name
        agent.config = Mock(name=name, tools=["test_tool"])
        agent.execute = AsyncMock(return_value={
            "agent_outputs": {name: {"result": f"Output from {name}"}},
            "final_answer": f"Answer from {name}"
        })
        return agent

    registry.get_agent = Mock(side_effect=get_agent)
    registry.list_agents = Mock(return_value=[
        "orchestrator", "extractor", "classifier", "compliance",
        "risk_verifier", "citation_auditor", "gap_synthesizer", "report_generator"
    ])

    return registry


# ============================================================================
# Mock State Fixtures
# ============================================================================

@pytest.fixture
def mock_state():
    """Factory for creating mock multi-agent states."""
    def _create_state(
        query: str = "Test query",
        complexity_score: int = 50,
        agent_outputs: Dict = None,
        errors: List[str] = None
    ):
        return {
            "query": query,
            "complexity_score": complexity_score,
            "query_type": "retrieval",
            "agent_sequence": [],
            "agent_outputs": agent_outputs or {},
            "execution_phase": "planning",
            "errors": errors or [],
            "metadata": {}
        }

    return _create_state


# ============================================================================
# Mock Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_agent_config():
    """Factory for creating mock agent configurations."""
    def _create_config(
        name: str = "test_agent",
        model: str = "claude-sonnet-4-5-20250929",
        tools: List[str] = None,
        **kwargs
    ):
        from src.multi_agent.core.agent_base import AgentConfig, AgentRole, AgentTier

        return AgentConfig(
            name=name,
            role=kwargs.get("role", AgentRole.EXTRACT),
            tier=kwargs.get("tier", AgentTier.WORKER),
            model=model,
            max_tokens=1024,
            temperature=0.1,
            tools=set(tools) if tools else {"search", "get_info"},
            enable_prompt_caching=True,
            **{k: v for k, v in kwargs.items() if k not in ["role", "tier"]}
        )

    return _create_config


# ============================================================================
# Mock Checkpointer Fixtures
# ============================================================================

@pytest.fixture
def mock_checkpointer():
    """Mock PostgreSQL checkpointer for testing workflow persistence."""
    checkpointer = Mock()

    checkpointer.save = AsyncMock()
    checkpointer.load = AsyncMock(return_value={
        "query": "Test query",
        "agent_outputs": {}
    })
    checkpointer.delete = AsyncMock()

    return checkpointer


# ============================================================================
# Embeddings Fixtures
# ============================================================================

@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    def _generate(dim: int = 768, count: int = 5):
        return [np.random.rand(dim).astype(np.float32) for _ in range(count)]

    return _generate


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "document_id": "doc1",
            "title": "GDPR Overview",
            "content": "General Data Protection Regulation is a regulation in EU law...",
            "metadata": {"source": "eu-law", "year": 2018}
        },
        {
            "document_id": "doc2",
            "title": "CCPA Guide",
            "content": "California Consumer Privacy Act is a state statute...",
            "metadata": {"source": "ca-law", "year": 2020}
        }
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing retrieval."""
    return [
        {
            "chunk_id": "doc1:sec1:0",
            "document_id": "doc1",
            "text": "GDPR applies to all companies processing personal data of EU citizens.",
            "relevance_score": 0.95,
            "metadata": {"section": "Scope", "layer": 3}
        },
        {
            "chunk_id": "doc1:sec2:1",
            "document_id": "doc1",
            "text": "Personal data means any information relating to an identified person.",
            "relevance_score": 0.88,
            "metadata": {"section": "Definitions", "layer": 3}
        },
        {
            "chunk_id": "doc2:sec1:0",
            "document_id": "doc2",
            "text": "CCPA gives California residents rights over their personal information.",
            "relevance_score": 0.82,
            "metadata": {"section": "Rights", "layer": 3}
        }
    ]


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_postgres: marks tests requiring PostgreSQL"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: marks tests requiring real API keys"
    )
