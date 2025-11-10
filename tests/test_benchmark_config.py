"""
Tests for BenchmarkConfig module.

Critical tests for configuration validation and environment variable parsing.
"""

import json
import os
import pytest
from pathlib import Path

from src.benchmark.config import BenchmarkConfig


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_validation_missing_dataset(self, tmp_path):
        """Missing dataset file should raise FileNotFoundError."""
        # Create valid vector store
        vector_store = tmp_path / "vector_store"
        vector_store.mkdir()
        (vector_store / "faiss_metadata.json").write_text("{}")

        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            BenchmarkConfig(
                dataset_path="nonexistent/dataset.json", vector_store_path=str(vector_store)
            )

    def test_validation_missing_vector_store(self, tmp_path):
        """Missing vector store should raise FileNotFoundError."""
        # Create valid dataset
        dataset = tmp_path / "dataset.json"
        dataset.write_text(json.dumps({"tests": []}))

        with pytest.raises(FileNotFoundError, match="Vector store not found"):
            BenchmarkConfig(dataset_path=str(dataset), vector_store_path="nonexistent_vector_store")

    def test_validation_valid_paths(self, tmp_path):
        """Valid paths should pass validation."""
        # Create valid dataset
        dataset = tmp_path / "dataset.json"
        dataset.write_text(json.dumps({"tests": []}))

        # Create valid vector store
        vector_store = tmp_path / "vector_store"
        vector_store.mkdir()
        (vector_store / "faiss_metadata.json").write_text("{}")

        # Should not raise
        config = BenchmarkConfig(dataset_path=str(dataset), vector_store_path=str(vector_store))

        assert config.dataset_path == str(dataset)
        assert config.vector_store_path == str(vector_store)

    def test_debug_mode_enables_save_per_query(self, tmp_path):
        """debug_mode=True should automatically enable save_per_query."""
        dataset = tmp_path / "dataset.json"
        dataset.write_text(json.dumps({"tests": []}))

        vector_store = tmp_path / "vector_store"
        vector_store.mkdir()
        (vector_store / "faiss_metadata.json").write_text("{}")

        config = BenchmarkConfig(
            dataset_path=str(dataset),
            vector_store_path=str(vector_store),
            debug_mode=True,
            save_per_query=False,  # Will be overridden
        )

        assert config.save_per_query is True  # Auto-enabled by debug_mode

    def test_output_dir_created_if_missing(self, tmp_path):
        """Output directory should be created if it doesn't exist."""
        dataset = tmp_path / "dataset.json"
        dataset.write_text(json.dumps({"tests": []}))

        vector_store = tmp_path / "vector_store"
        vector_store.mkdir()
        (vector_store / "faiss_metadata.json").write_text("{}")

        output_dir = tmp_path / "nonexistent_output"

        config = BenchmarkConfig(
            dataset_path=str(dataset),
            vector_store_path=str(vector_store),
            output_dir=str(output_dir),
        )

        assert output_dir.exists()  # Should be created during validation


class TestConfigSerialization:
    """Test configuration serialization (to_dict)."""

    def test_to_dict_contains_all_fields(self, tmp_path):
        """to_dict() should return all configuration fields."""
        dataset = tmp_path / "dataset.json"
        dataset.write_text(json.dumps({"tests": []}))

        vector_store = tmp_path / "vector_store"
        vector_store.mkdir()
        (vector_store / "faiss_metadata.json").write_text("{}")

        config = BenchmarkConfig(
            dataset_path=str(dataset),
            vector_store_path=str(vector_store),
            k=10,
            enable_reranking=False,
            max_queries=5,
        )

        config_dict = config.to_dict()

        assert config_dict["k"] == 10
        assert config_dict["enable_reranking"] is False
        assert config_dict["max_queries"] == 5
        assert "dataset_path" in config_dict
        assert "vector_store_path" in config_dict


class TestEnvironmentVariableParsing:
    """Test configuration loading from environment variables."""

    def test_from_env_with_overrides(self, tmp_path, monkeypatch):
        """from_env() should respect overrides."""
        # Setup environment
        dataset = tmp_path / "dataset.json"
        dataset.write_text(json.dumps({"tests": []}))

        vector_store = tmp_path / "vector_store"
        vector_store.mkdir()
        (vector_store / "faiss_metadata.json").write_text("{}")

        monkeypatch.setenv("BENCHMARK_K", "20")

        # Override k and provide paths via overrides (not env)
        config = BenchmarkConfig.from_env(
            k=10, dataset_path=str(dataset), vector_store_path=str(vector_store)
        )

        assert config.k == 10  # Override should take precedence
        assert config.dataset_path == str(dataset)  # From override

    def test_from_env_invalid_type_raises_error(self, tmp_path, monkeypatch):
        """from_env() with invalid type should raise ValueError."""
        dataset = tmp_path / "dataset.json"
        dataset.write_text(json.dumps({"tests": []}))

        vector_store = tmp_path / "vector_store"
        vector_store.mkdir()
        (vector_store / "faiss_metadata.json").write_text("{}")

        monkeypatch.setenv("BENCHMARK_K", "not_a_number")  # Invalid

        # Should raise ValueError when parsing invalid int
        with pytest.raises(ValueError, match="invalid literal"):
            BenchmarkConfig.from_env(
                dataset_path=str(dataset), vector_store_path=str(vector_store)
            )
