"""
Tests for Anthropic provider - specifically tools parameter handling.

This test suite verifies the fix for PR #102: Anthropic BadRequestError
when tools=None or tools=[] is passed to the API.

The Anthropic API requires:
- If tools parameter is present, it must be a non-empty list
- If no tools, the parameter must be OMITTED entirely (not None, not [])

Bug scenario: Orchestrator synthesis phase calls LLM with tools=None,
causing BadRequestError: "tools: Field required" or similar.
"""

import pytest
from unittest.mock import Mock, patch
from src.agent.providers.anthropic_provider import AnthropicProvider


def _create_mock_response(text="response", stop_reason="end_turn", input_tokens=100, output_tokens=50):
    """Helper to create properly structured mock Anthropic response."""
    mock_content_block = Mock()
    mock_content_block.model_dump.return_value = {"type": "text", "text": text}

    mock_response = Mock()
    mock_response.content = [mock_content_block]
    mock_response.stop_reason = stop_reason
    mock_response.usage = Mock(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0
    )
    mock_response.model = "claude-sonnet-4-5"
    return mock_response


class TestAnthropicProviderToolsHandling:
    """Test tools parameter handling to prevent BadRequestError."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        with patch("anthropic.Anthropic") as mock:
            yield mock.return_value

    @pytest.fixture
    def provider(self, mock_anthropic_client):
        """Create provider with mocked client."""
        provider = AnthropicProvider(
            model="claude-sonnet-4-5",
            api_key="sk-ant-test123"
        )
        provider._client = mock_anthropic_client
        return provider

    def test_create_message_with_none_tools(self, provider, mock_anthropic_client):
        """Should NOT pass tools parameter when tools=None."""
        # Mock successful response
        mock_anthropic_client.messages.create.return_value = _create_mock_response()

        # Call with tools=None
        provider.create_message(
            messages=[{"role": "user", "content": "test"}],
            tools=None,  # Should NOT be passed to API
            system="test system",
            max_tokens=100,
            temperature=0.7
        )

        # Verify tools parameter NOT passed
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs, "tools parameter should be omitted when None"

    def test_create_message_with_empty_tools(self, provider, mock_anthropic_client):
        """Should NOT pass tools parameter when tools=[]."""
        # Mock successful response
        mock_anthropic_client.messages.create.return_value = _create_mock_response()

        # Call with tools=[]
        provider.create_message(
            messages=[{"role": "user", "content": "test"}],
            tools=[],  # Empty list - should NOT be passed to API
            system="test system",
            max_tokens=100,
            temperature=0.7
        )

        # Verify tools parameter NOT passed
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs, "tools parameter should be omitted when empty list"

    def test_create_message_with_valid_tools(self, provider, mock_anthropic_client):
        """SHOULD pass tools parameter when tools contain schemas."""
        # Mock successful response
        mock_anthropic_client.messages.create.return_value = _create_mock_response(stop_reason="tool_use")

        # Call with valid tools
        tools = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]

        provider.create_message(
            messages=[{"role": "user", "content": "test"}],
            tools=tools,  # Valid tools - SHOULD be passed to API
            system="test system",
            max_tokens=100,
            temperature=0.7
        )

        # Verify tools parameter WAS passed
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_kwargs, "tools parameter should be passed when non-empty"
        assert call_kwargs["tools"] == tools

    def test_stream_message_with_none_tools(self, provider, mock_anthropic_client):
        """Should NOT pass tools parameter when tools=None (streaming)."""
        # Mock stream context manager
        mock_stream = Mock()
        mock_anthropic_client.messages.stream.return_value = mock_stream

        # Call with tools=None
        provider.stream_message(
            messages=[{"role": "user", "content": "test"}],
            tools=None,
            system="test system",
            max_tokens=100,
            temperature=0.7
        )

        # Verify tools parameter NOT passed
        call_kwargs = mock_anthropic_client.messages.stream.call_args[1]
        assert "tools" not in call_kwargs, "tools parameter should be omitted when None (streaming)"

    def test_stream_message_with_empty_tools(self, provider, mock_anthropic_client):
        """Should NOT pass tools parameter when tools=[] (streaming)."""
        # Mock stream context manager
        mock_stream = Mock()
        mock_anthropic_client.messages.stream.return_value = mock_stream

        # Call with tools=[]
        provider.stream_message(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            system="test system",
            max_tokens=100,
            temperature=0.7
        )

        # Verify tools parameter NOT passed
        call_kwargs = mock_anthropic_client.messages.stream.call_args[1]
        assert "tools" not in call_kwargs, "tools parameter should be omitted when empty list (streaming)"

    def test_stream_message_with_valid_tools(self, provider, mock_anthropic_client):
        """SHOULD pass tools parameter when tools contain schemas (streaming)."""
        # Mock stream context manager
        mock_stream = Mock()
        mock_anthropic_client.messages.stream.return_value = mock_stream

        # Call with valid tools
        tools = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]

        provider.stream_message(
            messages=[{"role": "user", "content": "test"}],
            tools=tools,
            system="test system",
            max_tokens=100,
            temperature=0.7
        )

        # Verify tools parameter WAS passed
        call_kwargs = mock_anthropic_client.messages.stream.call_args[1]
        assert "tools" in call_kwargs, "tools parameter should be passed when non-empty (streaming)"
        assert call_kwargs["tools"] == tools


class TestAnthropicProviderSynthesisScenario:
    """Test the actual synthesis phase scenario from orchestrator."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        with patch("anthropic.Anthropic") as mock:
            yield mock.return_value

    @pytest.fixture
    def provider(self, mock_anthropic_client):
        """Create provider with mocked client."""
        provider = AnthropicProvider(
            model="claude-sonnet-4-5",
            api_key="sk-ant-test123"
        )
        provider._client = mock_anthropic_client
        return provider

    def test_orchestrator_synthesis_phase_scenario(self, provider, mock_anthropic_client):
        """
        Integration test: Orchestrator synthesis phase doesn't crash.

        The orchestrator calls provider.create_message(tools=None) in synthesis phase
        (line 379 in orchestrator.py). This should NOT cause BadRequestError.
        """
        # Mock successful synthesis response
        mock_anthropic_client.messages.create.return_value = _create_mock_response(
            text="Based on the agent outputs, here is the final answer...",
            input_tokens=500,
            output_tokens=200
        )

        # Simulate orchestrator synthesis call
        result = provider.create_message(
            messages=[
                {
                    "role": "user",
                    "content": "You are synthesizing final answer from agent outputs:\n\n"
                    "Extractor: Found data...\nCompliance: Verified...\n\n"
                    "Provide final answer:"
                }
            ],
            tools=None,  # Synthesis phase has NO tools
            system="You are orchestrator in synthesis phase",
            max_tokens=300,
            temperature=0.7
        )

        # Verify:
        # 1. No BadRequestError raised
        # 2. Response returned successfully
        # 3. tools parameter NOT passed to API
        assert result is not None
        assert result.content[0]["text"].startswith("Based on")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs
