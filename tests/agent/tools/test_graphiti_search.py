"""
Tests for GraphitiSearchTool.

Tests search recipe selection, episode hydration, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


@pytest.fixture
def anyio_backend():
    """Use asyncio backend for async tests."""
    return "asyncio"


# =============================================================================
# TEST FIXTURES
# =============================================================================


@dataclass
class MockGraphitiSearchResult:
    """Mock search result for testing."""

    facts: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    total_results: int = 0
    query: str = ""
    mode: str = "semantic"


@pytest.fixture
def mock_graphiti_search_tool():
    """Create a mock GraphitiSearchTool for testing."""
    from src.agent.tools.graphiti_search import GraphitiSearchTool, GraphitiSearchResult

    # Create tool with mocked dependencies
    tool = GraphitiSearchTool.__new__(GraphitiSearchTool)

    # Mock the graphiti client
    tool._graphiti = MagicMock()
    tool._graphiti.driver = MagicMock()

    return tool


@pytest.fixture
def search_result_with_episodes():
    """Create a search result with facts containing episode references."""
    from src.agent.tools.graphiti_search import GraphitiSearchResult

    return GraphitiSearchResult(
        facts=[
            {"fact": "Test fact 1", "episodes": ["ep-uuid-1", "ep-uuid-2"]},
            {"fact": "Test fact 2", "episodes": ["ep-uuid-2", "ep-uuid-3"]},  # ep-uuid-2 duplicate
            {"fact": "Test fact 3", "episodes": []},  # Empty
            {"fact": "Test fact 4", "episodes": None},  # None
        ],
        entities=[],
        total_results=4,
    )


# =============================================================================
# SEARCH RECIPE SELECTION TESTS
# =============================================================================


class TestSearchRecipeSelection:
    """Tests for _pick_search_config method."""

    def test_pick_search_config_returns_none_when_no_recipe(self, mock_graphiti_search_tool):
        """Test None recipe returns None config."""
        config = mock_graphiti_search_tool._pick_search_config(None, num_results=10)
        assert config is None

    def test_pick_search_config_returns_none_for_empty_string(self, mock_graphiti_search_tool):
        """Test empty string recipe returns None config."""
        config = mock_graphiti_search_tool._pick_search_config("", num_results=10)
        assert config is None

    def test_pick_search_config_raises_on_unknown_recipe(self, mock_graphiti_search_tool):
        """Test unknown recipe raises ValueError with valid options."""
        with pytest.raises(ValueError) as exc_info:
            mock_graphiti_search_tool._pick_search_config("invalid_recipe", num_results=10)

        error_msg = str(exc_info.value)
        assert "Unknown search recipe: 'invalid_recipe'" in error_msg
        assert "combined_rrf" in error_msg  # Should list valid recipes
        assert "node_rrf" in error_msg

    @pytest.mark.parametrize(
        "recipe",
        [
            "combined_rrf",
            "combined_mmr",
            "combined_cross_encoder",
            "edge_distance",
            "edge_rrf",
            "node_rrf",
            "node_cross_encoder",
        ],
    )
    def test_pick_search_config_valid_recipes(self, mock_graphiti_search_tool, recipe):
        """Test all valid recipes return a config object."""
        config = mock_graphiti_search_tool._pick_search_config(recipe, num_results=10)
        assert config is not None

    def test_pick_search_config_sets_limit(self, mock_graphiti_search_tool):
        """Test num_results is propagated to config limit."""
        config = mock_graphiti_search_tool._pick_search_config("combined_rrf", num_results=25)
        assert config is not None

        # Check limit is set on main config or subconfigs
        limit_found = False
        if hasattr(config, "limit"):
            assert config.limit == 25
            limit_found = True

        for subconfig_name in ("edge_config", "node_config", "community_config", "episode_config"):
            subconfig = getattr(config, subconfig_name, None)
            if subconfig and hasattr(subconfig, "limit"):
                assert subconfig.limit == 25
                limit_found = True

        assert limit_found, "No limit was set on config or subconfigs"

    def test_pick_search_config_deep_copies(self, mock_graphiti_search_tool):
        """Test that config is deep copied to avoid mutation."""
        config1 = mock_graphiti_search_tool._pick_search_config("combined_rrf", num_results=10)
        config2 = mock_graphiti_search_tool._pick_search_config("combined_rrf", num_results=20)

        # Should be different objects
        assert config1 is not config2

        # Limits should be different
        if hasattr(config1, "limit") and hasattr(config2, "limit"):
            assert config1.limit != config2.limit


# =============================================================================
# EPISODE HYDRATION TESTS
# =============================================================================


class TestEpisodeHydration:
    """Tests for _append_episodes method."""

    @pytest.mark.anyio
    async def test_append_episodes_collects_unique_ids(
        self, mock_graphiti_search_tool, search_result_with_episodes, anyio_backend
    ):
        """Test unique episode IDs are collected from all facts."""
        mock_episode = MagicMock()
        mock_episode.uuid = "ep-uuid-1"
        mock_episode.name = "Episode 1"
        mock_episode.content = "Content 1"
        mock_episode.valid_at = None
        mock_episode.source = MagicMock(value="user")
        mock_episode.group_id = "doc-1"

        with patch("graphiti_core.nodes.EpisodicNode") as mock_episodic_node:
            mock_episodic_node.get_by_uuids = AsyncMock(return_value=[mock_episode])

            await mock_graphiti_search_tool._append_episodes(
                search_result_with_episodes, episode_limit=10
            )

            # Check get_by_uuids was called with unique IDs
            call_args = mock_episodic_node.get_by_uuids.call_args
            episode_ids = call_args[0][1]

            # Should have 3 unique IDs (ep-uuid-1, ep-uuid-2, ep-uuid-3)
            assert len(episode_ids) == 3
            assert "ep-uuid-1" in episode_ids
            assert "ep-uuid-2" in episode_ids
            assert "ep-uuid-3" in episode_ids

    @pytest.mark.anyio
    async def test_append_episodes_respects_limit(self, mock_graphiti_search_tool, anyio_backend):
        """Test episode_limit caps the fetched episodes."""
        from src.agent.tools.graphiti_search import GraphitiSearchResult

        # Create result with many episodes
        result = GraphitiSearchResult(
            facts=[{"episodes": [f"ep-{i}" for i in range(20)]}],
            entities=[],
            total_results=1,
        )

        with patch("graphiti_core.nodes.EpisodicNode") as mock_episodic_node:
            mock_episodic_node.get_by_uuids = AsyncMock(return_value=[])

            await mock_graphiti_search_tool._append_episodes(result, episode_limit=5)

            call_args = mock_episodic_node.get_by_uuids.call_args
            episode_ids = call_args[0][1]

            # Should be limited to 5
            assert len(episode_ids) == 5

    @pytest.mark.anyio
    async def test_append_episodes_handles_empty_facts(self, mock_graphiti_search_tool, anyio_backend):
        """Test graceful handling when no facts have episodes."""
        from src.agent.tools.graphiti_search import GraphitiSearchResult

        result = GraphitiSearchResult(
            facts=[
                {"episodes": []},
                {"episodes": None},
                {},  # No episodes key
            ],
            entities=[],
            total_results=3,
        )

        with patch("graphiti_core.nodes.EpisodicNode") as mock_episodic_node:
            mock_episodic_node.get_by_uuids = AsyncMock(return_value=[])

            # Should not call get_by_uuids when no episode IDs
            await mock_graphiti_search_tool._append_episodes(result, episode_limit=10)

            mock_episodic_node.get_by_uuids.assert_not_called()

    @pytest.mark.anyio
    async def test_append_episodes_handles_fetch_failure_gracefully(
        self, mock_graphiti_search_tool, search_result_with_episodes, anyio_backend
    ):
        """Test search result is not corrupted when episode fetch fails."""
        with patch("graphiti_core.nodes.EpisodicNode") as mock_episodic_node:
            mock_episodic_node.get_by_uuids = AsyncMock(
                side_effect=Exception("Neo4j connection error")
            )

            # Should not raise, just log warning
            await mock_graphiti_search_tool._append_episodes(
                search_result_with_episodes, episode_limit=10
            )

            # Episodes should remain empty (not crash)
            assert search_result_with_episodes.episodes == []

    @pytest.mark.anyio
    async def test_append_episodes_converts_to_dict_format(self, mock_graphiti_search_tool, anyio_backend):
        """Test episodes are converted to dict format correctly."""
        from src.agent.tools.graphiti_search import GraphitiSearchResult
        from datetime import datetime, timezone

        result = GraphitiSearchResult(
            facts=[{"episodes": ["ep-1"]}],
            entities=[],
            total_results=1,
        )

        mock_episode = MagicMock()
        mock_episode.uuid = "ep-1"
        mock_episode.name = "Test Episode"
        mock_episode.content = "Episode content"
        mock_episode.valid_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mock_episode.source = MagicMock(value="user")
        mock_episode.group_id = "doc-123"

        with patch("graphiti_core.nodes.EpisodicNode") as mock_episodic_node:
            mock_episodic_node.get_by_uuids = AsyncMock(return_value=[mock_episode])

            await mock_graphiti_search_tool._append_episodes(result, episode_limit=10)

            assert len(result.episodes) == 1
            ep = result.episodes[0]
            assert ep["uuid"] == "ep-1"
            assert ep["name"] == "Test Episode"
            assert ep["content"] == "Episode content"
            assert ep["valid_at"] == "2024-01-01T00:00:00+00:00"
            assert ep["source"] == "user"
            assert ep["group_id"] == "doc-123"


# =============================================================================
# INTEGRATION TESTS (require graphiti_core)
# =============================================================================


@pytest.mark.skipif(
    not pytest.importorskip("graphiti_core", reason="graphiti_core not installed"),
    reason="graphiti_core not installed",
)
class TestGraphitiSearchIntegration:
    """Integration tests that require graphiti_core."""

    def test_search_recipe_imports_succeed(self):
        """Test that all recipe imports work."""
        from graphiti_core.search.search_config_recipes import (
            COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
            COMBINED_HYBRID_SEARCH_MMR,
            COMBINED_HYBRID_SEARCH_RRF,
            EDGE_HYBRID_SEARCH_NODE_DISTANCE,
            EDGE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_CROSS_ENCODER,
            NODE_HYBRID_SEARCH_RRF,
        )

        assert COMBINED_HYBRID_SEARCH_RRF is not None
        assert COMBINED_HYBRID_SEARCH_MMR is not None
        assert COMBINED_HYBRID_SEARCH_CROSS_ENCODER is not None
        assert EDGE_HYBRID_SEARCH_NODE_DISTANCE is not None
        assert EDGE_HYBRID_SEARCH_RRF is not None
        assert NODE_HYBRID_SEARCH_RRF is not None
        assert NODE_HYBRID_SEARCH_CROSS_ENCODER is not None
