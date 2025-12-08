#!/usr/bin/env python3
"""
Comprehensive test script for tools refactoring.

Tests:
1. Python syntax validation
2. Import chain verification
3. Tool registration count
4. No tier references
5. Import path consistency
"""

import ast
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_syntax():
    """Test 1: Validate Python syntax of all tool files."""
    print("\n" + "="*60)
    print("TEST 1: Python Syntax Validation")
    print("="*60)

    tools_dir = Path(__file__).parent / "src" / "agent" / "tools"
    tool_files = list(tools_dir.glob("*.py"))

    errors = []
    for file_path in sorted(tool_files):
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            ast.parse(code)
            print(f"✓ {file_path.name}: Syntax OK")
        except SyntaxError as e:
            errors.append(f"✗ {file_path.name}: {e}")
            print(f"✗ {file_path.name}: Syntax Error - {e}")

    if errors:
        print(f"\n❌ FAILED: {len(errors)} syntax errors found")
        return False
    else:
        print(f"\n✅ PASSED: All {len(tool_files)} files have valid syntax")
        return True


def test_imports():
    """Test 2: Verify import chain works."""
    print("\n" + "="*60)
    print("TEST 2: Import Chain Verification")
    print("="*60)

    try:
        # Test infrastructure imports
        print("Testing infrastructure imports...")
        from agent.tools._base import BaseTool, ToolInput, ToolResult
        print("  ✓ _base imports successful")

        from agent.tools._registry import ToolRegistry, get_registry
        print("  ✓ _registry imports successful")

        from agent.tools._utils import format_chunk_result, generate_citation
        print("  ✓ _utils imports successful")

        # Test main package import
        print("\nTesting main package import...")
        from agent.tools import get_registry as get_reg
        print("  ✓ tools package imports successful")

        print("\n✅ PASSED: All imports work correctly")
        return True

    except ImportError as e:
        print(f"\n❌ FAILED: Import error - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registration():
    """Test 3: Verify all 15 tools are registered."""
    print("\n" + "="*60)
    print("TEST 3: Tool Registration Count")
    print("="*60)

    try:
        from agent.tools import get_registry

        registry = get_registry()
        tool_count = len(registry._tool_classes)

        print(f"Registered tools: {tool_count}")
        print("\nRegistered tool names:")
        for name in sorted(registry._tool_classes.keys()):
            print(f"  - {name}")

        if tool_count == 15:
            print(f"\n✅ PASSED: All 15 tools registered correctly")
            return True
        else:
            print(f"\n❌ FAILED: Expected 15 tools, found {tool_count}")
            return False

    except Exception as e:
        print(f"\n❌ FAILED: Registration test error - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_tier_references():
    """Test 4: Verify no tier references remain."""
    print("\n" + "="*60)
    print("TEST 4: No Tier References")
    print("="*60)

    tools_dir = Path(__file__).parent / "src" / "agent" / "tools"
    tool_files = [f for f in tools_dir.glob("*.py") if not f.name.startswith("_")]

    tier_refs = []
    for file_path in tool_files:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Check for tier assignments
                if "tier =" in line and not line.strip().startswith("#"):
                    tier_refs.append(f"{file_path.name}:{line_num}: {line.strip()}")

    if tier_refs:
        print("❌ FAILED: Found tier references:")
        for ref in tier_refs:
            print(f"  {ref}")
        return False
    else:
        print(f"✅ PASSED: No tier assignments found in {len(tool_files)} tool files")
        return True


def test_import_consistency():
    """Test 5: Verify import paths use underscore prefix."""
    print("\n" + "="*60)
    print("TEST 5: Import Path Consistency")
    print("="*60)

    tools_dir = Path(__file__).parent / "src" / "agent" / "tools"
    tool_files = [f for f in tools_dir.glob("*.py") if not f.name.startswith("_") and f.name != "__init__.py"]

    bad_imports = []
    for file_path in tool_files:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Check for old-style imports (without underscore)
                if "from .base import" in line or "from .registry import" in line or \
                   "from .utils import" in line or "from .token_manager import" in line:
                    bad_imports.append(f"{file_path.name}:{line_num}: {line.strip()}")

    if bad_imports:
        print("❌ FAILED: Found old-style imports (should use underscore prefix):")
        for imp in bad_imports:
            print(f"  {imp}")
        return False
    else:
        print(f"✅ PASSED: All imports use correct underscore prefix in {len(tool_files)} files")
        return True


def test_file_structure():
    """Test 6: Verify expected file structure."""
    print("\n" + "="*60)
    print("TEST 6: File Structure")
    print("="*60)

    tools_dir = Path(__file__).parent / "src" / "agent" / "tools"

    # Expected infrastructure files
    expected_infra = ["_base.py", "_registry.py", "_utils.py", "_token_manager.py", "__init__.py"]

    # Expected tool files (17)
    expected_tools = [
        "get_tool_help.py", "search.py", "get_document_list.py",
        "list_available_tools.py", "get_document_info.py",
        "graph_search.py", "multi_doc_synthesizer.py", "contextual_chunk_enricher.py",
        "explain_search_results.py", "assess_retrieval_confidence.py", "filtered_search.py",
        "similarity_search.py", "expand_context.py", "browse_entities.py", "cluster_search.py",
        "get_stats.py", "definition_aligner.py"
    ]

    # Check infrastructure files
    missing_infra = []
    for file in expected_infra:
        if not (tools_dir / file).exists():
            missing_infra.append(file)

    # Check tool files
    missing_tools = []
    for file in expected_tools:
        if not (tools_dir / file).exists():
            missing_tools.append(file)

    # Check for old tier files
    old_tier_files = []
    for old_file in ["tier1_basic.py", "tier2_advanced.py", "tier3_analysis.py"]:
        if (tools_dir / old_file).exists():
            old_tier_files.append(old_file)

    # Report results
    print(f"Infrastructure files: {len(expected_infra) - len(missing_infra)}/{len(expected_infra)}")
    print(f"Tool files: {len(expected_tools) - len(missing_tools)}/{len(expected_tools)}")

    issues = []
    if missing_infra:
        issues.append(f"Missing infrastructure files: {missing_infra}")
    if missing_tools:
        issues.append(f"Missing tool files: {missing_tools}")
    if old_tier_files:
        issues.append(f"Old tier files still present: {old_tier_files}")

    if issues:
        print("\n❌ FAILED:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ PASSED: All expected files present, no old tier files")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("COMPREHENSIVE REFACTORING TEST SUITE")
    print("="*60)

    results = {
        "Syntax Validation": test_syntax(),
        "Import Chain": test_imports(),
        "Tool Registration": test_registration(),
        "No Tier References": test_no_tier_references(),
        "Import Consistency": test_import_consistency(),
        "File Structure": test_file_structure(),
    }

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Refactoring successful!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
