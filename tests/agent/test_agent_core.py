"""
Tests for Agent Core functionality.

Tests critical features:
- Query length validation
- Conversation history trimming
- Tool execution error handling
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agent.agent_core import AgentCore, MAX_HISTORY_MESSAGES, MAX_QUERY_LENGTH
from src.agent.config import AgentConfig


@pytest.fixture
def mock_config():
    """Create a mock AgentConfig for testing."""
    config = Mock(spec=AgentConfig)
    config.model = "claude-sonnet-4-5-20250929"
    config.max_tokens = 4096
    config.temperature = 0.3
    config.anthropic_api_key = "sk-ant-test-key"
    config.openai_api_key = ""
    config.google_api_key = ""
    config.enable_prompt_caching = True
    config.enable_context_management = True
    config.context_management_trigger = 50000
    config.context_management_keep = 3
    config.system_prompt = "You are a helpful assistant."
    config.debug_mode = False
    config.cli_config = Mock()
    config.cli_config.enable_streaming = False
    config.cli_config.show_tool_calls = True
    config.validate = Mock()
    return config


@pytest.fixture
def mock_registry():
    """Create a mock tool registry."""
    registry = Mock()
    registry.get_claude_sdk_tools = Mock(return_value=[])
    registry.execute_tool = Mock()
    registry.__len__ = Mock(return_value=17)
    registry._tool_classes = {"test_tool": Mock}
    return registry


@pytest.fixture
def agent_core(mock_config, mock_registry):
    """Create AgentCore instance with mocked dependencies."""
    with patch("src.agent.agent_core.anthropic.Anthropic"):
        with patch("src.agent.agent_core.get_registry", return_value=mock_registry):
            agent = AgentCore(mock_config)
            return agent


class TestQueryValidation:
    """Test query length validation."""

    def test_normal_query_accepted(self, agent_core):
        """Test that normal-length queries are accepted."""
        query = "What is GRI 306?"
        # Should not raise
        with patch.object(agent_core, "_process_non_streaming", return_value="response"):
            result = agent_core.process_message(query)
            assert result == "response"

    def test_query_too_long_rejected(self, agent_core):
        """Test that overly long queries are rejected."""
        long_query = "x" * (MAX_QUERY_LENGTH + 1)

        with pytest.raises(ValueError) as exc_info:
            agent_core.process_message(long_query)

        assert "Query too long" in str(exc_info.value)
        assert str(MAX_QUERY_LENGTH) in str(exc_info.value)

    def test_max_length_query_accepted(self, agent_core):
        """Test that queries at exact max length are accepted."""
        query = "x" * MAX_QUERY_LENGTH

        with patch.object(agent_core, "_process_non_streaming", return_value="response"):
            result = agent_core.process_message(query)
            assert result == "response"


class TestConversationHistory:
    """Test conversation history management."""

    def test_history_starts_empty(self, agent_core):
        """Test that conversation history starts empty."""
        assert len(agent_core.conversation_history) == 0

    def test_message_added_to_history(self, agent_core):
        """Test that messages are added to history."""
        with patch.object(agent_core, "_process_non_streaming", return_value="response"):
            agent_core.process_message("Test query")

        assert len(agent_core.conversation_history) == 1
        assert agent_core.conversation_history[0]["role"] == "user"
        assert agent_core.conversation_history[0]["content"] == "Test query"

    def test_history_trimmed_when_exceeds_max(self, agent_core):
        """Test that history is trimmed when it exceeds MAX_HISTORY_MESSAGES."""
        # Add MAX_HISTORY_MESSAGES + 5 messages
        num_messages = MAX_HISTORY_MESSAGES + 5

        with patch.object(agent_core, "_process_non_streaming", return_value="response"):
            for i in range(num_messages):
                agent_core.process_message(f"Message {i}")

        # Should be trimmed to MAX_HISTORY_MESSAGES
        assert len(agent_core.conversation_history) <= MAX_HISTORY_MESSAGES

    def test_trimming_keeps_most_recent(self, agent_core):
        """Test that trimming keeps the most recent messages."""
        # Add messages beyond limit
        num_messages = MAX_HISTORY_MESSAGES + 10

        with patch.object(agent_core, "_process_non_streaming", return_value="response"):
            for i in range(num_messages):
                agent_core.process_message(f"Message {i}")

        # Check that we kept the most recent messages
        # The last message should be present
        last_message_content = f"Message {num_messages - 1}"
        assert any(
            msg.get("content") == last_message_content for msg in agent_core.conversation_history
        )

    def test_reset_conversation_clears_history(self, agent_core):
        """Test that reset_conversation clears all history."""
        # Add some messages
        with patch.object(agent_core, "_process_non_streaming", return_value="response"):
            agent_core.process_message("Test 1")
            agent_core.process_message("Test 2")

        assert len(agent_core.conversation_history) > 0

        # Reset
        agent_core.reset_conversation()

        assert len(agent_core.conversation_history) == 0
        assert len(agent_core.tool_call_history) == 0


class TestConversationStats:
    """Test conversation statistics."""

    def test_stats_empty_initially(self, agent_core):
        """Test that stats are empty initially."""
        stats = agent_core.get_conversation_stats()

        assert stats["message_count"] == 0
        assert stats["tool_calls"] == 0
        assert stats["tools_used"] == []

    def test_stats_track_messages(self, agent_core):
        """Test that stats track message count."""
        with patch.object(agent_core, "_process_non_streaming", return_value="response"):
            agent_core.process_message("Test 1")
            agent_core.process_message("Test 2")

        stats = agent_core.get_conversation_stats()
        assert stats["message_count"] == 2


class TestStreamingToggle:
    """Test streaming vs non-streaming mode."""

    def test_non_streaming_by_default(self, agent_core):
        """Test that non-streaming is used by default if config says so."""
        agent_core.config.cli_config.enable_streaming = False

        with patch.object(
            agent_core, "_process_non_streaming", return_value="response"
        ) as mock_non_streaming:
            with patch.object(agent_core, "_process_streaming") as mock_streaming:
                agent_core.process_message("Test")

                mock_non_streaming.assert_called_once()
                mock_streaming.assert_not_called()

    def test_streaming_when_configured(self, agent_core):
        """Test that streaming is used when configured."""
        agent_core.config.cli_config.enable_streaming = True

        with patch.object(
            agent_core, "_process_streaming", return_value=iter([])
        ) as mock_streaming:
            with patch.object(agent_core, "_process_non_streaming") as mock_non_streaming:
                result = agent_core.process_message("Test")

                # Consume generator
                list(result)

                mock_streaming.assert_called_once()
                mock_non_streaming.assert_not_called()

    def test_explicit_stream_parameter(self, agent_core):
        """Test that explicit stream parameter overrides config."""
        agent_core.config.cli_config.enable_streaming = False

        # Force streaming with explicit parameter
        with patch.object(
            agent_core, "_process_streaming", return_value=iter([])
        ) as mock_streaming:
            result = agent_core.process_message("Test", stream=True)
            list(result)  # Consume generator

            mock_streaming.assert_called_once()


class TestTrimHistory:
    """Test the _trim_history method directly."""

    def test_trim_does_nothing_when_under_limit(self, agent_core):
        """Test that trimming doesn't affect history under the limit."""
        # Add a few messages
        for i in range(10):
            agent_core.conversation_history.append({"role": "user", "content": f"Message {i}"})

        original_length = len(agent_core.conversation_history)
        agent_core._trim_history()

        assert len(agent_core.conversation_history) == original_length

    def test_trim_reduces_to_max_when_over_limit(self, agent_core):
        """Test that trimming reduces history to MAX when over limit."""
        # Add messages beyond limit
        for i in range(MAX_HISTORY_MESSAGES + 20):
            agent_core.conversation_history.append({"role": "user", "content": f"Message {i}"})

        agent_core._trim_history()

        assert len(agent_core.conversation_history) == MAX_HISTORY_MESSAGES
