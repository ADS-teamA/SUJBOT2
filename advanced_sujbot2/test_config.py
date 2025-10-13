#!/usr/bin/env python3
"""
Test script to verify configuration system

This script tests:
1. Loading config from YAML
2. ENV variable overrides
3. Factory functions for component configs
4. Configuration validation
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config
from src.config_factory import (
    create_retrieval_config,
    create_embedding_config,
    create_reranking_config,
    create_cross_doc_config,
    create_knowledge_graph_config,
    validate_config,
)


def test_basic_loading():
    """Test basic config loading from YAML"""
    print("\n" + "="*80)
    print("TEST 1: Basic Config Loading")
    print("="*80)

    try:
        # Load config (will fail without CLAUDE_API_KEY, but that's expected in test)
        config = load_config("config.yaml")
        print("✓ Config loaded successfully from config.yaml")

        # Check some values
        semantic_weight = config.get("retrieval.semantic_weight")
        print(f"✓ retrieval.semantic_weight = {semantic_weight}")

        bm25_k1 = config.get("retrieval.bm25.k1")
        print(f"✓ retrieval.bm25.k1 = {bm25_k1}")

        reranking_model = config.get("reranking.cross_encoder_model")
        print(f"✓ reranking.cross_encoder_model = {reranking_model}")

        cross_doc_explicit = config.get("cross_document.explicit_weight")
        print(f"✓ cross_document.explicit_weight = {cross_doc_explicit}")

        graph_boost = config.get("knowledge_graph.graph_boost_factor")
        print(f"✓ knowledge_graph.graph_boost_factor = {graph_boost}")

        return True
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False


def test_env_override():
    """Test ENV variable overrides"""
    print("\n" + "="*80)
    print("TEST 2: ENV Variable Overrides")
    print("="*80)

    try:
        # Set some ENV variables
        os.environ["RETRIEVAL_SEMANTIC_WEIGHT"] = "0.6"
        os.environ["RETRIEVAL_TOP_K"] = "30"
        os.environ["RERANKING_FINAL_TOP_K"] = "10"
        os.environ["CROSS_DOC_EXPLICIT_WEIGHT"] = "0.7"

        # Reload config
        config = load_config("config.yaml")

        # Check overrides
        semantic_weight = config.get("retrieval.semantic_weight")
        expected = 0.6
        assert semantic_weight == expected, f"Expected {expected}, got {semantic_weight}"
        print(f"✓ ENV override works: RETRIEVAL_SEMANTIC_WEIGHT = {semantic_weight}")

        top_k = config.get("retrieval.top_k")
        expected = 30
        assert top_k == expected, f"Expected {expected}, got {top_k}"
        print(f"✓ ENV override works: RETRIEVAL_TOP_K = {top_k}")

        rerank_k = config.get("reranking.final_top_k")
        expected = 10
        assert rerank_k == expected, f"Expected {expected}, got {rerank_k}"
        print(f"✓ ENV override works: RERANKING_FINAL_TOP_K = {rerank_k}")

        explicit_weight = config.get("cross_document.explicit_weight")
        expected = 0.7
        assert explicit_weight == expected, f"Expected {expected}, got {explicit_weight}"
        print(f"✓ ENV override works: CROSS_DOC_EXPLICIT_WEIGHT = {explicit_weight}")

        # Clean up
        del os.environ["RETRIEVAL_SEMANTIC_WEIGHT"]
        del os.environ["RETRIEVAL_TOP_K"]
        del os.environ["RERANKING_FINAL_TOP_K"]
        del os.environ["CROSS_DOC_EXPLICIT_WEIGHT"]

        return True
    except Exception as e:
        print(f"✗ ENV override test failed: {e}")
        return False


def test_factory_functions():
    """Test factory functions for component configs"""
    print("\n" + "="*80)
    print("TEST 3: Factory Functions")
    print("="*80)

    try:
        # Set dummy API key for testing
        os.environ["CLAUDE_API_KEY"] = "test-key"

        config = load_config("config.yaml")

        # Test retrieval config
        retrieval_config = create_retrieval_config(config)
        print(f"✓ RetrievalConfig created: semantic_weight={retrieval_config.semantic_weight}")
        assert retrieval_config.semantic_weight == 0.5
        assert retrieval_config.keyword_weight == 0.3
        assert retrieval_config.structural_weight == 0.2
        print(f"  - Weights: {retrieval_config.semantic_weight}, {retrieval_config.keyword_weight}, {retrieval_config.structural_weight}")

        # Test embedding config
        embedding_config = create_embedding_config(config)
        print(f"✓ EmbeddingConfig created: model={embedding_config.model_name}")
        print(f"  - Device: {embedding_config.device}, batch_size: {embedding_config.batch_size}")

        # Test reranking config
        reranking_config = create_reranking_config(config)
        print(f"✓ RerankingConfig created: model={reranking_config.cross_encoder_model}")
        print(f"  - final_top_k: {reranking_config.final_top_k}")

        # Test cross-doc config
        cross_doc_config = create_cross_doc_config(config)
        print(f"✓ Cross-doc config created: explicit_weight={cross_doc_config['explicit_weight']}")
        print(f"  - Weights: {cross_doc_config['explicit_weight']}, {cross_doc_config['semantic_weight']}, {cross_doc_config['structural_weight']}")

        # Test knowledge graph config
        kg_config = create_knowledge_graph_config(config)
        print(f"✓ Knowledge graph config created: boost_factor={kg_config['graph_boost_factor']}")
        print(f"  - max_hops: {kg_config['max_hops']}, semantic_threshold: {kg_config['semantic_link_threshold']}")

        # Clean up
        del os.environ["CLAUDE_API_KEY"]

        return True
    except Exception as e:
        print(f"✗ Factory function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation():
    """Test configuration validation"""
    print("\n" + "="*80)
    print("TEST 4: Configuration Validation")
    print("="*80)

    try:
        os.environ["CLAUDE_API_KEY"] = "test-key"

        config = load_config("config.yaml")

        # Should pass with default config
        validate_config(config)
        print("✓ Default configuration is valid")

        # Test invalid weights (don't sum to 1.0)
        config.set("retrieval.semantic_weight", 0.6)
        config.set("retrieval.keyword_weight", 0.6)  # Now sum > 1.0

        try:
            validate_config(config)
            print("✗ Should have failed validation with invalid weights")
            del os.environ["CLAUDE_API_KEY"]
            return False
        except ValueError as e:
            print(f"✓ Correctly caught invalid weights: {str(e)[:100]}")

        # Clean up
        del os.environ["CLAUDE_API_KEY"]

        return True
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  SUJBOT2 Configuration System Test".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)

    results = []

    # Run tests
    results.append(("Basic Loading", test_basic_loading()))
    results.append(("ENV Overrides", test_env_override()))
    results.append(("Factory Functions", test_factory_functions()))
    results.append(("Validation", test_validation()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Configuration system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
