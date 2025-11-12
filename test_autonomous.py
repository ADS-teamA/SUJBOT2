#!/usr/bin/env python3
"""Quick test for autonomous agent architecture."""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

async def test_autonomous_compliance():
    """Test autonomous ComplianceAgent."""
    print("=" * 80)
    print("TESTING AUTONOMOUS COMPLIANCE AGENT")
    print("=" * 80)

    # Import after logging setup
    from src.multi_agent.core.agent_base import AgentConfig, AgentRole, AgentTier
    from src.multi_agent.agents.compliance import ComplianceAgent

    # Create minimal config
    config = AgentConfig(
        name="compliance",
        role=AgentRole.VERIFY,
        tier=AgentTier.SPECIALIST,
        model="claude-sonnet-4-5",  # Use Claude
        tools={"graph_search", "assess_confidence", "similarity_search"},
        max_tokens=1000,
        temperature=0.3
    )

    print(f"\n1. Created config for {config.name}")
    print(f"   Tools: {config.tools}")

    # Initialize agent
    try:
        agent = ComplianceAgent(config)
        print(f"\n2. Initialized ComplianceAgent")
        print(f"   Has provider: {hasattr(agent, 'provider')}")
    except Exception as e:
        print(f"\n✗ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create test state
    state = {
        "query": "Je tento dokument v souladu s GDPR?",
        "agent_outputs": {
            "extractor": {
                "chunks": [
                    {"text": "Dokument obsahuje zpracování osobních údajů.", "document_id": "test1"},
                    {"text": "Není zde uvedena právní základna pro zpracování.", "document_id": "test1"}
                ],
                "num_chunks_retrieved": 2
            }
        },
        "agent_sequence": ["extractor", "compliance"]
    }

    print(f"\n3. Created test state")
    print(f"   Query: {state['query']}")
    print(f"   Extractor chunks: {len(state['agent_outputs']['extractor']['chunks'])}")

    # Execute agent
    try:
        print(f"\n4. Executing autonomous ComplianceAgent...")
        result = await agent.execute(state)

        print(f"\n5. ✓ Agent execution completed!")
        print(f"   Final answer: {result.get('final_answer', 'N/A')[:200]}...")
        print(f"   Compliance output keys: {list(result.get('agent_outputs', {}).get('compliance', {}).keys())}")

        compliance_output = result.get('agent_outputs', {}).get('compliance', {})
        if 'tool_calls_made' in compliance_output:
            print(f"   Tools called: {compliance_output['tool_calls_made']}")
            print(f"   Iterations: {compliance_output.get('iterations', 'N/A')}")

    except Exception as e:
        print(f"\n✗ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_autonomous_compliance())
