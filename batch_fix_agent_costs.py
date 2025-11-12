#!/usr/bin/env python3
"""
Batch fix all agents to store total_tool_cost_usd from autonomous loop results.

This script updates 6 agents (extractor, classifier, compliance, risk_verifier,
citation_auditor, gap_synthesizer) to include cost tracking in their outputs.
"""

import re
from pathlib import Path

AGENTS_TO_FIX = [
    "extractor",
    "classifier",
    "compliance",
    "risk_verifier",
    "citation_auditor",
    "gap_synthesizer",
]

BASE_DIR = Path(__file__).parent / "src" / "multi_agent" / "agents"


def fix_agent_cost_tracking(agent_name: str) -> bool:
    """
    Update agent to store total_tool_cost_usd from autonomous loop result.

    Args:
        agent_name: Name of agent file (without .py extension)

    Returns:
        True if file was modified, False otherwise
    """
    filepath = BASE_DIR / f"{agent_name}.py"

    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False

    content = filepath.read_text()

    # Pattern 1: Extract total_tool_cost_usd from autonomous loop result
    # Look for: result = await self._run_autonomous_tool_loop(...)
    #           final_answer = result.get("final_answer", "")
    #           tool_calls = result.get("tool_calls", [])
    # Add:      agent_cost = result.get("total_tool_cost_usd", 0.0)

    pattern1 = r'(final_answer = result\.get\("final_answer", ""\)\s*\n\s*tool_calls = result\.get\("tool_calls", \[\]\))'
    replacement1 = r'\1\n            agent_cost = result.get("total_tool_cost_usd", 0.0)'

    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        print(f"✅ Added agent_cost extraction in {agent_name}")
    else:
        print(f"⚠️  Could not find pattern in {agent_name} (might already be fixed)")
        return False

    # Pattern 2: Add total_tool_cost_usd to output dict
    # Look for output dict creation with "tool_calls_made" key
    # Add "total_tool_cost_usd": agent_cost after iterations

    # This is more complex because each agent has slightly different output structure
    # We'll search for patterns like:
    #   "iterations": result.get("iterations", 0),
    #   ... (possibly other fields)
    # }
    # And add before the closing }:
    #   "total_tool_cost_usd": agent_cost

    # Match output dict with iterations field
    pattern2 = r'("iterations": result\.get\("iterations", 0\),?\s*\n)(\s*"[^"]+": [^,\n]+,?\s*\n)*(\s*\})'

    def add_cost_field(match):
        iterations_line = match.group(1)
        other_fields = match.group(2) or ""
        closing_brace = match.group(3)

        # Check if total_tool_cost_usd already exists
        if "total_tool_cost_usd" in other_fields:
            return match.group(0)  # Already has cost field

        # Add cost field before closing brace
        # Remove comma from iterations line if it exists
        iterations_line = iterations_line.rstrip(',\n') + ',\n'
        return f'{iterations_line}{other_fields}{"                " if other_fields else "            "}"total_tool_cost_usd": agent_cost\n{closing_brace}'

    if re.search(pattern2, content):
        content = re.sub(pattern2, add_cost_field, content)
        print(f"✅ Added total_tool_cost_usd field to output dict in {agent_name}")
    else:
        print(f"⚠️  Could not find output dict pattern in {agent_name}")
        return False

    # Write back
    filepath.write_text(content)
    print(f"✅ Successfully updated {agent_name}.py")
    return True


def main():
    print("🔧 Batch fixing agent cost tracking...\n")

    fixed_count = 0
    for agent_name in AGENTS_TO_FIX:
        print(f"\n📝 Processing {agent_name}...")
        if fix_agent_cost_tracking(agent_name):
            fixed_count += 1

    print(f"\n\n✅ Fixed {fixed_count}/{len(AGENTS_TO_FIX)} agents")
    print("\nℹ️  Please verify changes with: git diff src/multi_agent/agents/")


if __name__ == "__main__":
    main()
