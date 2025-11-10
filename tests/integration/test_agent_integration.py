#!/usr/bin/env python
"""
Integration Test Script for RAG Agent

Tests all imports, component compatibility, and basic functionality.
Run this to verify the agent is correctly set up.

Usage:
    python test_agent_integration.py
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_test(name: str):
    """Print test name."""
    print(f"\n{BLUE}Testing:{RESET} {name}")


def print_pass(message: str):
    """Print success message."""
    print(f"  {GREEN}✓{RESET} {message}")


def print_fail(message: str):
    """Print failure message."""
    print(f"  {RED}✗{RESET} {message}")


def print_warn(message: str):
    """Print warning message."""
    print(f"  {YELLOW}⚠{RESET} {message}")


def test_python_version():
    """Test Python version compatibility."""
    print_test("Python version")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 10:
        print_pass(f"Python {version_str} (compatible)")
        return True
    else:
        print_fail(f"Python {version_str} - requires Python 3.10+")
        return False


def test_core_dependencies():
    """Test core dependency imports."""
    print_test("Core dependencies")

    deps = [
        ("anthropic", "Claude SDK"),
        ("pydantic", "Input validation"),
        ("numpy", "Numerical operations"),
    ]

    all_ok = True
    for module_name, desc in deps:
        try:
            __import__(module_name)
            print_pass(f"{desc} ({module_name})")
        except ImportError as e:
            print_fail(f"{desc} ({module_name}) - {str(e)}")
            all_ok = False

    return all_ok


def test_pipeline_dependencies():
    """Test RAG pipeline dependency imports."""
    print_test("RAG pipeline dependencies")

    deps = [
        ("faiss", "Vector search"),
    ]

    all_ok = True
    for module_name, desc in deps:
        try:
            __import__(module_name)
            print_pass(f"{desc} ({module_name})")
        except ImportError as e:
            print_fail(f"{desc} ({module_name}) - {str(e)}")
            all_ok = False

    return all_ok


def test_optional_dependencies():
    """Test optional dependency imports."""
    print_test("Optional dependencies")

    deps = [
        ("sentence_transformers", "Local embeddings"),
        ("torch", "PyTorch for local models"),
    ]

    for module_name, desc in deps:
        try:
            __import__(module_name)
            print_pass(f"{desc} ({module_name})")
        except ImportError:
            print_warn(f"{desc} ({module_name}) - not installed (optional)")


def test_agent_imports():
    """Test agent module imports."""
    print_test("Agent module imports")

    imports = [
        ("src.agent.config", "AgentConfig"),
        ("src.agent.agent_core", "AgentCore"),
        ("src.agent.cli", "CLI module"),
        ("src.agent.tools.base", "Tool base classes"),
        ("src.agent.tools.registry", "Tool registry"),
        ("src.agent.tools.tier1_basic", "Tier 1 tools"),
        ("src.agent.tools.tier2_advanced", "Tier 2 tools"),
        ("src.agent.tools.tier3_analysis", "Tier 3 tools"),
        ("src.agent.tools.token_manager", "Token manager module"),
        ("src.agent.validation", "Validation module"),
    ]

    all_ok = True
    for module_name, desc in imports:
        try:
            __import__(module_name)
            print_pass(f"{desc} ({module_name})")
        except ImportError as e:
            print_fail(f"{desc} ({module_name}) - {str(e)}")
            all_ok = False

    return all_ok


def test_pipeline_imports():
    """Test RAG pipeline imports."""
    print_test("RAG pipeline imports")

    imports = [
        ("src.hybrid_search", "Hybrid search"),
        ("src.embedding_generator", "Embedding generator"),
        ("src.reranker", "Cross-encoder reranker"),
        ("src.context_assembly", "Context assembler"),
        ("src.graph_retrieval", "Graph retrieval"),
        ("src.graph.models", "Knowledge graph models"),
    ]

    all_ok = True
    for module_name, desc in imports:
        try:
            __import__(module_name)
            print_pass(f"{desc} ({module_name})")
        except ImportError as e:
            print_fail(f"{desc} ({module_name}) - {str(e)}")
            all_ok = False

    return all_ok


def test_tool_registry():
    """Test tool registry functionality."""
    print_test("Tool registry")

    try:
        from src.agent.tools.registry import get_registry

        registry = get_registry()
        tool_count = len(registry._tool_classes)

        if tool_count == 14:
            print_pass(f"All 14 tools registered")
            return True
        else:
            print_warn(f"Expected 14 tools, found {tool_count}")
            return False

    except Exception as e:
        print_fail(f"Failed to check registry: {str(e)}")
        return False


def test_config_creation():
    """Test config creation."""
    print_test("Config creation")

    try:
        from src.agent.config import AgentConfig
        from pathlib import Path

        # Create minimal config (without validation)
        config = AgentConfig(
            anthropic_api_key="test-key",  # Fake key for testing
            vector_store_path=Path("output/hybrid_store"),  # May not exist
        )

        print_pass("AgentConfig created")

        # Check debug_mode field exists
        if hasattr(config, "debug_mode"):
            print_pass("debug_mode field present")
        else:
            print_fail("debug_mode field missing")
            return False

        return True

    except Exception as e:
        print_fail(f"Failed to create config: {str(e)}")
        return False


def test_api_keys():
    """Test API key environment variables."""
    print_test("API keys")

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    has_anthropic = bool(anthropic_key)
    has_openai = bool(openai_key)

    if has_anthropic:
        print_pass("ANTHROPIC_API_KEY set")
    else:
        print_warn("ANTHROPIC_API_KEY not set (required for agent)")

    if has_openai:
        print_pass("OPENAI_API_KEY set")
    else:
        print_warn("OPENAI_API_KEY not set (required for cloud embeddings)")

    return has_anthropic  # Only Anthropic key is critical


def test_vector_store_exists():
    """Test if vector store exists."""
    print_test("Vector store")

    # Try default path
    default_path = Path("output/hybrid_store")

    if default_path.exists() and default_path.is_dir():
        # Check for required files
        required_files = [
            "index_layer1.faiss",
            "index_layer2.faiss",
            "index_layer3.faiss",
        ]

        missing = []
        for filename in required_files:
            if not (default_path / filename).exists():
                missing.append(filename)

        if not missing:
            print_pass(f"Vector store found at {default_path}")
            return True
        else:
            print_warn(f"Vector store incomplete (missing: {missing})")
            return False
    else:
        print_warn(f"Vector store not found at {default_path}")
        print_warn("Run indexing pipeline: python run_pipeline.py data/documents/")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print(f"{BLUE}RAG AGENT INTEGRATION TEST{RESET}")
    print("=" * 80)

    results = {
        "Python version": test_python_version(),
        "Core dependencies": test_core_dependencies(),
        "Pipeline dependencies": test_pipeline_dependencies(),
        "Agent imports": test_agent_imports(),
        "Pipeline imports": test_pipeline_imports(),
        "Tool registry": test_tool_registry(),
        "Config creation": test_config_creation(),
        "API keys": test_api_keys(),
        "Vector store": test_vector_store_exists(),
    }

    # Test optional dependencies (doesn't affect pass/fail)
    test_optional_dependencies()

    # Summary
    print("\n" + "=" * 80)
    print(f"{BLUE}TEST SUMMARY{RESET}")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print(f"\n{GREEN}✓ ALL TESTS PASSED{RESET}")
        print("\n🚀 Agent is ready to use!")
        print(f"   Run: ./run_cli.sh output/hybrid_store")
        print(f"   Or:  uv run python -m src.agent.cli --vector-store output/hybrid_store")
        return 0
    else:
        print(f"\n{YELLOW}⚠  SOME TESTS FAILED{RESET}")
        print("\nFailed tests:")
        for name, passed in results.items():
            if not passed:
                print(f"  • {name}")

        print("\n💡 Fix the issues above and run this test again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
