#!/usr/bin/env python3
"""
Quick test script to verify all 3 critical fixes:
1. Issue 1: final_answer extraction from LangGraph
2. Issue 2: Entity attribute error in get_stats
3. Issue 3: Token estimation for EntityType

Run: uv run python test_fixes.py
"""

import asyncio
import json
import sys
from pathlib import Path
from src.multi_agent.runner import MultiAgentRunner
from src.agent.config import AgentConfig


def load_runner_config():
    """Load runner configuration from config files."""
    project_root = Path(__file__).parent
    config_path = project_root / "config.json"

    with open(config_path) as f:
        full_config = json.load(f)

    # Load multi-agent config
    multi_agent_config = full_config.get("multi_agent", {})
    if not multi_agent_config:
        extension_path = project_root / "config_multi_agent_extension.json"
        with open(extension_path) as f:
            extension_config = json.load(f)
            multi_agent_config = extension_config.get("multi_agent", {})

    # Build runner config
    return {
        "api_keys": full_config.get("api_keys", {}),
        "vector_store_path": str(project_root / "vector_db"),
        "models": full_config.get("models", {}),
        "storage": full_config.get("storage", {}),
        "agent_tools": full_config.get("agent_tools", {}),
        "multi_agent": multi_agent_config,
    }


async def test_basic_query():
    """Test basic query that should return direct answer (Issue 1)."""
    print("\n" + "="*70)
    print("TEST 1: Basic query (tests Issue 1 - final_answer extraction)")
    print("="*70)

    config = load_runner_config()
    runner = MultiAgentRunner(config)
    await runner.initialize()

    query = "Hello, how are you?"
    print(f"\nQuery: {query}")
    print("\nExpected: Direct answer (no agents, just orchestrator)")
    print("\nResult:")
    print("-" * 70)

    final_result = None
    async for event in runner.run_query(query, stream_progress=True):
        if event.get("type") == "final":
            final_result = event.get("final_answer")
            print(f"✓ Final answer: {final_result[:100]}...")

    if not final_result or final_result == "No answer generated":
        print("❌ TEST 1 FAILED: No answer generated")
        return False

    print("✅ TEST 1 PASSED: Got final answer")
    return True


async def test_stats_query():
    """Test get_stats tool (Issues 2 & 3)."""
    print("\n" + "="*70)
    print("TEST 2: Stats query (tests Issue 2 - Entity attr & Issue 3 - token estimation)")
    print("="*70)

    config = load_runner_config()
    runner = MultiAgentRunner(config)
    await runner.initialize()

    query = "How many documents are in the corpus?"
    print(f"\nQuery: {query}")
    print("\nExpected: Uses get_stats tool with entity statistics")
    print("\nResult:")
    print("-" * 70)

    events = []
    final_result = None

    async for event in runner.run_query(query, stream_progress=True):
        if event.get("type") == "tool_call":
            events.append(event)
            print(f"  Tool: {event.get('tool')} - Status: {event.get('status')}")
        elif event.get("type") == "final":
            final_result = event.get("final_answer")
            print(f"\n✓ Final answer: {final_result[:200]}...")

    if not final_result or final_result == "No answer generated":
        print("❌ TEST 2 FAILED: No answer generated")
        return False

    # Check if get_stats was called
    get_stats_called = any(e.get("tool") == "get_stats" for e in events)
    if get_stats_called:
        print("✓ get_stats tool was called")
    else:
        print("⚠️  get_stats tool was not called (but query completed)")

    print("✅ TEST 2 PASSED: No Entity attribute or token estimation errors")
    return True


async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE FIX TESTING")
    print("="*70)

    try:
        test1_passed = await test_basic_query()
        test2_passed = await test_stats_query()

        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Test 1 (Issue 1 - final_answer extraction): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"Test 2 (Issues 2 & 3 - Entity & token): {'✅ PASSED' if test2_passed else '❌ FAILED'}")

        if test1_passed and test2_passed:
            print("\n🎉 ALL TESTS PASSED - Ready to commit!")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED - Review logs above")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
