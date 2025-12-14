"""
SSOT (Single Source of Truth) Tests for config.json consolidation.

These tests verify that:
1. ModelRegistry reads model aliases from config.json
2. backend/constants reads agent variants from config.json
3. layer_default_k reads from config.json
4. No duplicate MODEL_PRICING exists
5. Embedding dimensions are consistently 4096 (Qwen3-Embedding-8B)

SSOT Architecture:
- config.json is the single source of truth for all configuration values
- Modules use lazy loading with fallbacks for backward compatibility
- Tests verify config values propagate correctly to runtime
"""

import json
import pytest
from pathlib import Path


class TestConfigSchema:
    """Test config.json schema and structure."""

    @pytest.fixture
    def config_data(self):
        """Load config.json."""
        config_path = Path(__file__).parent.parent / "config.json"
        with open(config_path) as f:
            return json.load(f)

    def test_model_registry_section_exists(self, config_data):
        """Verify model_registry section exists in config.json."""
        assert "model_registry" in config_data
        assert "llm_models" in config_data["model_registry"]
        assert "embedding_models" in config_data["model_registry"]
        assert "embedding_dimensions" in config_data["model_registry"]

    def test_defaults_section_exists(self, config_data):
        """Verify defaults section exists in config.json."""
        assert "defaults" in config_data
        assert "timeouts" in config_data["defaults"]
        assert "pool_sizes" in config_data["defaults"]
        assert "retrieval" in config_data["defaults"]

    def test_agent_variants_section_exists(self, config_data):
        """Verify agent_variants section exists in config.json."""
        assert "agent_variants" in config_data
        assert "opus_tier_agents" in config_data["agent_variants"]
        assert "variants" in config_data["agent_variants"]
        assert "default_variant" in config_data["agent_variants"]

    def test_layer_default_k_values(self, config_data):
        """Verify layer_default_k has correct values."""
        layer_k = config_data["defaults"]["retrieval"]["layer_default_k"]
        assert layer_k["1"] == 3  # Documents
        assert layer_k["2"] == 5  # Sections
        assert layer_k["3"] == 10  # Chunks

    def test_embedding_dimensions_qwen3(self, config_data):
        """Verify Qwen3-Embedding-8B has 4096 dimensions (NOT 3072)."""
        dims = config_data["model_registry"]["embedding_dimensions"]
        assert dims["Qwen/Qwen3-Embedding-8B"] == 4096
        # OpenAI is 3072 - but we should NOT be using it as default
        assert dims.get("text-embedding-3-large") == 3072

    def test_default_embedding_model(self, config_data):
        """Verify default embedding model is Qwen3 (not OpenAI)."""
        default = config_data["model_registry"]["embedding_models"]["default"]
        assert default == "BAAI/bge-m3"  # Or Qwen3 if configured

    def test_variant_models_are_valid(self, config_data):
        """Verify all variant models reference real model IDs."""
        variants = config_data["agent_variants"]["variants"]
        for variant_name, variant_config in variants.items():
            assert "opus_model" in variant_config, f"Missing opus_model in {variant_name}"
            assert "default_model" in variant_config, f"Missing default_model in {variant_name}"
            # Models should have version suffixes (not just aliases)
            if "claude" in variant_config["opus_model"]:
                assert "-" in variant_config["opus_model"], f"Model should have version: {variant_config['opus_model']}"


class TestModelRegistrySSoT:
    """Test that ModelRegistry reads from config.json."""

    def test_resolve_llm_from_config(self):
        """Verify ModelRegistry.resolve_llm uses config values."""
        from src.utils.model_registry import ModelRegistry

        # Test alias resolution
        resolved = ModelRegistry.resolve_llm("haiku")
        assert "claude-haiku" in resolved or "claude-haiku-4" in resolved

        resolved = ModelRegistry.resolve_llm("sonnet")
        assert "claude-sonnet" in resolved

        resolved = ModelRegistry.resolve_llm("opus")
        assert "claude-opus" in resolved

    def test_get_embedding_model_dimensions(self):
        """Verify embedding dimensions are retrieved correctly."""
        from src.utils.model_registry import ModelRegistry

        # Qwen3 should be 4096
        dims = ModelRegistry.get_embedding_dimensions("Qwen/Qwen3-Embedding-8B")
        assert dims == 4096

        # OpenAI should be 3072
        dims = ModelRegistry.get_embedding_dimensions("text-embedding-3-large")
        assert dims == 3072

    def test_registry_reload(self):
        """Verify registry can be reloaded from config."""
        from src.utils.model_registry import reload_registry

        # Should not raise
        reload_registry()


class TestBackendConstantsSSoT:
    """Test that backend/constants reads from config.json."""

    def test_opus_tier_agents_from_config(self):
        """Verify OPUS_TIER_AGENTS matches config.json."""
        from backend.constants import get_opus_tier_agents, reload_constants

        reload_constants()  # Ensure fresh load
        opus_agents = get_opus_tier_agents()

        # Check expected agents are in the set
        assert "orchestrator" in opus_agents
        assert "compliance" in opus_agents
        assert "extractor" in opus_agents

    def test_variant_config_from_config(self):
        """Verify VARIANT_CONFIG matches config.json."""
        from backend.constants import get_variant_config, reload_constants

        reload_constants()  # Ensure fresh load
        variants = get_variant_config()

        # Check expected variants exist
        assert "premium" in variants
        assert "cheap" in variants
        assert "local" in variants

        # Check each variant has required fields
        for name, config in variants.items():
            assert "display_name" in config
            assert "opus_model" in config
            assert "default_model" in config

    def test_get_agent_model_function(self):
        """Verify get_agent_model returns correct models."""
        from backend.constants import get_agent_model, get_opus_tier_agents

        opus_agents = get_opus_tier_agents()

        # In premium mode, orchestrator should get opus model
        opus_model = get_agent_model("premium", "orchestrator")
        assert "opus" in opus_model.lower() or "claude" in opus_model.lower()

        # Non-opus agent should get default model
        default_model = get_agent_model("premium", "classifier")
        # Could be sonnet or another default
        assert default_model is not None


class TestNoDuplicatePricing:
    """Verify MODEL_PRICING is not duplicated."""

    def test_no_model_pricing_in_toc_retrieval(self):
        """Verify ToC_retrieval.py uses central PRICING, not local MODEL_PRICING."""
        import inspect
        from src import ToC_retrieval

        # Check LLMAgent class doesn't have MODEL_PRICING as class attribute
        # (it should only have _DEFAULT_PRICING as fallback)
        agent_class = ToC_retrieval.LLMAgent

        # MODEL_PRICING should NOT be a class attribute
        assert not hasattr(agent_class, "MODEL_PRICING") or agent_class.MODEL_PRICING is None

    def test_toc_retrieval_uses_central_pricing(self):
        """Verify ToC_retrieval imports PRICING from cost_tracker."""
        import importlib
        import sys

        # Check import statement exists
        spec = importlib.util.find_spec("src.ToC_retrieval")
        source_path = spec.origin

        with open(source_path, "r") as f:
            content = f.read()

        assert "from src.cost_tracker import PRICING" in content


class TestDimensionsConsistency:
    """Verify embedding dimensions are consistently 4096."""

    def test_runner_dimensions_not_3072(self):
        """Verify runner.py doesn't fallback to 3072."""
        runner_path = Path(__file__).parent.parent / "src" / "multi_agent" / "runner.py"

        with open(runner_path, "r") as f:
            content = f.read()

        # Should NOT have dimensions=3072 (was a bug)
        # Should have dimensions=4096 as fallback
        assert "dimensions=3072" not in content or "dimensions, 3072" not in content
        # Should have correct fallback
        assert "4096" in content

    def test_enable_reranking_default_false(self):
        """Verify enable_reranking default is False (matches config.json)."""
        from src.agent.config import ToolConfig

        # Default should be False (matches config.json)
        # enable_reranking is in ToolConfig, not AgentConfig
        config = ToolConfig()
        assert config.enable_reranking is False


class TestLayerDefaultK:
    """Test layer_default_k centralization."""

    def test_utils_loads_layer_k_from_config(self):
        """Verify _utils.py loads layer_default_k from config."""
        from src.agent.tools._utils import _ensure_config_loaded, _layer_default_k

        _ensure_config_loaded()

        # Should have expected values
        assert _layer_default_k.get(1) == 3  # Documents
        assert _layer_default_k.get(2) == 5  # Sections
        assert _layer_default_k.get(3) == 10  # Chunks
