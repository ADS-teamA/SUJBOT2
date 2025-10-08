"""
Configuration System Usage Examples

This file demonstrates how to use the configuration system in SUJBOT2.
"""

from pathlib import Path
from src.config import Config, load_config, get_default_config
from src.logging_config import setup_logging_from_config, logger
from src.utils import ensure_dir, count_tokens, extract_legal_references


# ============================================================================
# Example 1: Load from YAML file
# ============================================================================

def example_load_from_yaml():
    """Load configuration from YAML file."""
    print("Example 1: Loading from YAML file")
    print("=" * 60)

    # Load config
    config = load_config("config.yaml")

    # Access configuration values
    print(f"Main model: {config.get('llm.main_model')}")
    print(f"Hybrid alpha: {config.get('retrieval.semantic_weight')}")
    print(f"Chunk size: {config.get('chunking.target_chunk_size')}")
    print(f"Index type: {config.get('indexing.index_type')}")

    # Setup logging from config
    setup_logging_from_config(config.to_dict())
    logger.info("Configuration loaded successfully!")

    print()


# ============================================================================
# Example 2: Load with programmatic overrides
# ============================================================================

def example_programmatic_overrides():
    """Load config with programmatic overrides."""
    print("Example 2: Programmatic overrides")
    print("=" * 60)

    # Load base config from file
    config = load_config(
        config_path="config.yaml",
        config={
            "retrieval": {
                "semantic_weight": 0.6,  # Override
                "keyword_weight": 0.3,
                "structural_weight": 0.1,
            },
            "llm": {
                "temperature": 0.0,  # Override for deterministic output
            }
        }
    )

    print(f"Semantic weight (overridden): {config.get('retrieval.semantic_weight')}")
    print(f"Temperature (overridden): {config.get('llm.temperature')}")

    print()


# ============================================================================
# Example 3: Environment variable overrides
# ============================================================================

def example_env_overrides():
    """Show environment variable overrides."""
    print("Example 3: Environment variable overrides")
    print("=" * 60)

    import os

    # Set environment variables (in practice, set these in shell or .env)
    os.environ["CLAUDE_API_KEY"] = "sk-ant-test-key"
    os.environ["MAIN_AGENT_MODEL"] = "claude-opus-4-1-20250805"
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Load config - env vars take precedence
    config = load_config("config.yaml")

    print(f"Main model (from env): {config.get('llm.main_model')}")
    print(f"Log level (from env): {config.get('api.logging.level')}")

    # API key is masked in display
    print(f"API key set: {'api_key' in config['llm']}")

    print()


# ============================================================================
# Example 4: Get default config
# ============================================================================

def example_defaults():
    """Get and inspect default configuration."""
    print("Example 4: Default configuration")
    print("=" * 60)

    defaults = get_default_config()

    print("Default configuration structure:")
    print(f"  - API settings: {list(defaults.get('api', {}).keys())}")
    print(f"  - LLM settings: {list(defaults.get('llm', {}).keys())}")
    print(f"  - Retrieval settings: {list(defaults.get('retrieval', {}).keys())}")
    print(f"  - Compliance settings: {list(defaults.get('compliance', {}).keys())}")

    print()


# ============================================================================
# Example 5: Save configuration
# ============================================================================

def example_save_config():
    """Save configuration to file."""
    print("Example 5: Save configuration")
    print("=" * 60)

    # Create custom config
    config = Config(config={
        "llm": {
            "main_model": "claude-sonnet-4-5-20250929",
            "temperature": 0.2,
        },
        "retrieval": {
            "semantic_weight": 0.5,
            "keyword_weight": 0.3,
            "structural_weight": 0.2,
        }
    })

    # Save to file (API key will be redacted)
    output_path = "my_custom_config.yaml"
    config.save(output_path)

    print(f"Configuration saved to: {output_path}")
    print(f"API key in saved file: REDACTED (for security)")

    # Clean up
    Path(output_path).unlink(missing_ok=True)

    print()


# ============================================================================
# Example 6: Using utilities with config
# ============================================================================

def example_utilities():
    """Demonstrate utility functions."""
    print("Example 6: Utility functions")
    print("=" * 60)

    # Text processing
    text = "§89 odst. 2 stanoví, že dodavatel odpovídá za vady."
    refs = extract_legal_references(text)
    print(f"Legal references found: {refs}")

    # Token counting
    tokens = count_tokens(text)
    print(f"Token count: {tokens}")

    # Directory management
    index_dir = ensure_dir("./temp_indexes")
    print(f"Ensured directory exists: {index_dir}")

    # Clean up
    index_dir.rmdir()

    print()


# ============================================================================
# Example 7: Configuration presets
# ============================================================================

def example_presets():
    """Show different configuration presets."""
    print("Example 7: Configuration presets")
    print("=" * 60)

    # High precision preset
    high_precision = {
        "retrieval": {
            "semantic_weight": 0.6,
            "keyword_weight": 0.25,
            "structural_weight": 0.15,
            "top_k": 10,
            "min_score_threshold": 0.3,
        },
        "chunking": {
            "target_chunk_size": 256,
        }
    }

    # High recall preset
    high_recall = {
        "retrieval": {
            "semantic_weight": 0.4,
            "keyword_weight": 0.4,
            "structural_weight": 0.2,
            "top_k": 30,
            "min_score_threshold": 0.05,
            "enable_query_expansion": True,
        },
        "chunking": {
            "target_chunk_size": 768,
        }
    }

    # Speed optimized preset
    speed_optimized = {
        "retrieval": {
            "semantic_weight": 0.3,
            "keyword_weight": 0.5,
            "structural_weight": 0.2,
            "parallel_retrieval": True,
            "enable_caching": True,
        },
        "indexing": {
            "index_type": "hnsw",
        }
    }

    print("Available presets:")
    print("  1. High Precision (legal/compliance)")
    print("  2. High Recall (research/exploration)")
    print("  3. Speed Optimized (fast queries)")

    # Use a preset
    config = Config(config=high_precision)
    print(f"\nUsing High Precision preset:")
    print(f"  Semantic weight: {config.get('retrieval.semantic_weight')}")
    print(f"  Top-K: {config.get('retrieval.top_k')}")
    print(f"  Chunk size: {config.get('chunking.target_chunk_size')}")

    print()


# ============================================================================
# Example 8: Validation
# ============================================================================

def example_validation():
    """Show configuration validation."""
    print("Example 8: Configuration validation")
    print("=" * 60)

    from src.exceptions import ConfigurationError

    # This will fail validation (invalid hybrid alpha)
    try:
        config = Config(config={
            "retrieval": {
                "hybrid_alpha": 1.5,  # Invalid: must be [0, 1]
            }
        })
    except ConfigurationError as e:
        print(f"Validation error caught: {e}")

    # This will fail (missing API key)
    try:
        import os
        # Remove API key from environment
        os.environ.pop("CLAUDE_API_KEY", None)

        config = Config(config={
            "llm": {}  # No API key
        })
    except Exception as e:
        print(f"Missing API key error: {type(e).__name__}")

    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SUJBOT2 Configuration System Examples")
    print("=" * 60 + "\n")

    try:
        example_load_from_yaml()
        example_programmatic_overrides()
        example_env_overrides()
        example_defaults()
        example_save_config()
        example_utilities()
        example_presets()
        example_validation()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set a dummy API key for examples
    import os
    os.environ["CLAUDE_API_KEY"] = "sk-ant-example-key-for-testing"

    main()
