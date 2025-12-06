#!/usr/bin/env python3
"""Quick test for Q6 and Q9 problematic queries after prompt fix."""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Apply nest_asyncio for PostgreSQL compatibility
import nest_asyncio
nest_asyncio.apply()

async def test_queries():
    # Load config
    config_path = Path(__file__).parent.parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    from src.multi_agent.runner import MultiAgentRunner
    runner = MultiAgentRunner(config)
    await runner.initialize()

    queries = [
        ("Q6", "Jaké události jsou v havarijním plánu reaktoru VR-1 klasifikovány jako radiační mimořádné události prvního stupně?"),
        ("Q9", "Jaké jsou bezpečnostní limity teploty pro moderátor a pokrytí paliva reaktoru VR-1?")
    ]

    for qid, q in queries:
        print(f"\n{'='*60}")
        print(f"{qid}: {q[:60]}...")
        print("="*60)

        result = None
        async for event in runner.run_query(q, stream_progress=False):
            if event.get("type") == "final":
                result = event

        if result:
            answer = result.get("final_answer", "NO ANSWER")
            tools = result.get("tools_used", [])
            print(f"\nTools used: {tools}")
            print(f"\nAnswer ({len(answer)} chars):")
            print("-" * 40)
            print(answer[:1500])
            print("-" * 40)

            # Check for issues
            if "REFLECTION" in answer and "Tool call" in answer:
                print("\n❌ REFLECTION LEAK DETECTED!")
            elif "\\cite" in answer or "cite{" in answer:
                print("\n✓ Answer has citations - looks good!")
            else:
                print("\n⚠ Needs manual review (no citations found)")
        else:
            print("\n❌ No result returned!")

    print("\n" + "="*60)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(test_queries())
