#!/usr/bin/env python3
"""Batch refactor remaining agents to autonomous pattern."""

import re
from pathlib import Path
from string import Template

AGENTS_TO_REFACTOR = [
    "citation_auditor.py",
    "gap_synthesizer.py",
    "classifier.py",
    "report_generator.py"
]

# Template for autonomous execute_impl
AUTONOMOUS_EXECUTE_TEMPLATE = '''    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        $docstring_first_line (AUTONOMOUS).

        LLM autonomously decides which tools to call for $agent_purpose.

        Args:
            state: Current workflow state

        Returns:
            Updated state with $agent_output
        """
        query = state.get("query", "")
        extractor_output = state.get("agent_outputs", {}).get("extractor", {})

        if not extractor_output:
            logger.warning("No extractor output found, skipping $agent_name")
            return state

        logger.info("Running autonomous $agent_name...")

        try:
            # Run autonomous tool calling loop
            # LLM decides which tools to call
            result = await self._run_autonomous_tool_loop(
                system_prompt=self.system_prompt,
                state=state,
                max_iterations=10
            )

            # Parse result from autonomous loop
            final_answer = result.get("final_answer", "")
            tool_calls = result.get("tool_calls", [])

            # Store output
            output = {
                "analysis": final_answer,
                "tool_calls_made": [t["tool"] for t in tool_calls],
                "iterations": result.get("iterations", 0)
            }

            # Update state
            state["agent_outputs"] = state.get("agent_outputs", {})
            state["agent_outputs"]["$agent_key"] = output

            logger.info(
                f"Autonomous $agent_name complete: "
                f"tools_used={len(tool_calls)}, iterations={result.get('iterations', 0)}"
            )

            return state

        except Exception as e:
            logger.error(f"Autonomous $agent_name failed: {e}", exc_info=True)
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"$agent_name error: {str(e)}")
            return state

    # Old hardcoded methods removed - autonomous pattern handles everything via LLM
'''

# Agent-specific metadata
AGENT_METADATA = {
    "citation_auditor.py": {
        "docstring_first_line": "Audit citations for accuracy",
        "agent_purpose": "citation verification",
        "agent_output": "citation audit results",
        "agent_name": "citation auditor",
        "agent_key": "citation_auditor"
    },
    "gap_synthesizer.py": {
        "docstring_first_line": "Synthesize gaps between requirements",
        "agent_purpose": "gap analysis",
        "agent_output": "identified gaps",
        "agent_name": "gap synthesizer",
        "agent_key": "gap_synthesizer"
    },
    "classifier.py": {
        "docstring_first_line": "Classify query intent and complexity",
        "agent_purpose": "query classification",
        "agent_output": "classification results",
        "agent_name": "classifier",
        "agent_key": "classifier"
    },
    "report_generator.py": {
        "docstring_first_line": "Generate comprehensive report",
        "agent_purpose": "report generation",
        "agent_output": "formatted report",
        "agent_name": "report generator",
        "agent_key": "report_generator"
    }
}


def refactor_agent(file_path: Path):
    """Refactor single agent to autonomous pattern."""
    print(f"Refactoring {file_path.name}...")

    content = file_path.read_text()
    metadata = AGENT_METADATA[file_path.name]

    # Find execute_impl method
    execute_pattern = r'(    async def execute_impl\(self, state: Dict\[str, Any\]\) -> Dict\[str, Any\]:.*?)(\n    (async )?def |$)'

    match = re.search(execute_pattern, content, re.DOTALL)
    if not match:
        print(f"  ✗ Could not find execute_impl in {file_path.name}")
        return False

    # Generate new execute_impl using Template
    template = Template(AUTONOMOUS_EXECUTE_TEMPLATE)
    new_execute = template.substitute(**metadata)

    # Replace execute_impl + remove everything after it (old helper methods)
    # Keep everything before execute_impl
    before_execute = content[:match.start()]

    # New content = before + new execute_impl
    new_content = before_execute + new_execute

    # Write back
    file_path.write_text(new_content)
    print(f"  ✓ {file_path.name} refactored")
    return True


def main():
    agents_dir = Path("src/multi_agent/agents")
    refactored = 0

    for agent_file in AGENTS_TO_REFACTOR:
        file_path = agents_dir / agent_file
        if file_path.exists():
            if refactor_agent(file_path):
                refactored += 1
        else:
            print(f"  ✗ {agent_file} not found")

    print(f"\n✓ Refactored {refactored}/{len(AGENTS_TO_REFACTOR)} agents to autonomous pattern")


if __name__ == "__main__":
    main()
