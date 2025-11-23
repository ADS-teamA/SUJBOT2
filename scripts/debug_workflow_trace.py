"""
Debug Workflow Trace Script

Spustí agentní workflow a zachytí VŠECHNY interní detaily:
- Orchestrator reasoning (routing + synthesis)
- Všechny LLM calls (system prompts, messages, responses, usage)
- Tool calls (inputs, outputs, metadata)
- Agent state transitions
- Kompletní trace ve formátu JSON

Usage:
    python scripts/debug_workflow_trace.py "Jaké jsou požadavky GDPR?"
    python scripts/debug_workflow_trace.py "Jaké jsou požadavky GDPR?" --output trace.json
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multi_agent.runner import MultiAgentRunner


# ============================================================================
# Trace Data Models
# ============================================================================

@dataclass
class LLMCall:
    """Single LLM API call trace."""
    iteration: int
    timestamp: str
    model: str
    provider: str

    # Request
    messages: List[Dict[str, Any]]
    system_prompt: Any  # str or list (with caching)
    tools: Optional[List[Dict]] = None
    temperature: float = 0.3
    max_tokens: int = 4096

    # Response
    response_text: str = ""
    stop_reason: str = ""
    content_blocks: List[Dict] = field(default_factory=list)

    # Tool uses in this call
    tool_uses: List[Dict] = field(default_factory=list)

    # Usage
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    # Timing & Cost
    response_time_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class ToolCall:
    """Single tool execution trace."""
    tool_name: str
    timestamp: str
    agent_name: str

    # Input
    tool_input: Dict[str, Any] = field(default_factory=dict)

    # Output
    success: bool = True
    result: Any = None
    error: Optional[str] = None

    # Metadata
    execution_time_ms: float = 0.0
    api_cost_usd: float = 0.0


@dataclass
class AgentTrace:
    """Complete trace for single agent execution."""
    agent_name: str
    start_time: str
    end_time: Optional[str] = None

    # Input
    input_state: Dict[str, Any] = field(default_factory=dict)

    # Execution
    llm_calls: List[LLMCall] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    iterations: int = 0

    # Output
    output: Dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""

    # Stats
    total_time_ms: float = 0.0
    total_cost_usd: float = 0.0


@dataclass
class WorkflowTrace:
    """Complete workflow trace."""
    query: str
    timestamp: str

    # Orchestrator
    orchestrator_routing: Optional[AgentTrace] = None
    orchestrator_synthesis: Optional[AgentTrace] = None

    # Agents
    agents: List[AgentTrace] = field(default_factory=list)

    # Results
    final_answer: str = ""
    complexity_score: int = 0
    query_type: str = ""
    agent_sequence: List[str] = field(default_factory=list)

    # Stats
    total_time_ms: float = 0.0
    total_cost_usd: float = 0.0
    success: bool = True
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Tracing Wrappers
# ============================================================================

class TracingProvider:
    """Wrapper around LLM provider that traces all calls."""

    def __init__(self, wrapped_provider, agent_name: str, trace_list: List[LLMCall]):
        self.wrapped = wrapped_provider
        self.agent_name = agent_name
        self.trace_list = trace_list
        self.iteration = 0

    def create_message(self, messages, tools=None, system=None, max_tokens=4096, temperature=0.3):
        """Trace LLM call and forward to wrapped provider."""
        self.iteration += 1
        start_time = time.time()

        # Create trace entry
        trace = LLMCall(
            iteration=self.iteration,
            timestamp=datetime.now().isoformat(),
            model=self.wrapped.get_model_name(),
            provider=self.wrapped.get_provider_name(),
            messages=self._sanitize_messages(messages),
            system_prompt=self._sanitize_system(system),
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Call wrapped provider
        try:
            response = self.wrapped.create_message(
                messages=messages,
                tools=tools,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Record response
            trace.response_time_ms = (time.time() - start_time) * 1000
            trace.response_text = getattr(response, 'text', '')
            trace.stop_reason = getattr(response, 'stop_reason', '')

            # Extract content blocks
            if hasattr(response, 'content'):
                trace.content_blocks = self._sanitize_content(response.content)

                # Extract tool uses
                trace.tool_uses = [
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {})
                    }
                    for block in trace.content_blocks
                    if block.get("type") == "tool_use"
                ]

            # Extract usage
            if hasattr(response, 'usage'):
                usage = response.usage
                trace.input_tokens = usage.get('input_tokens', 0)
                trace.output_tokens = usage.get('output_tokens', 0)
                trace.cache_read_tokens = usage.get('cache_read_input_tokens', 0)
                trace.cache_creation_tokens = usage.get('cache_creation_input_tokens', 0)

            # Add to trace list
            self.trace_list.append(trace)

            return response

        except Exception as e:
            trace.response_time_ms = (time.time() - start_time) * 1000
            trace.error = str(e)
            self.trace_list.append(trace)
            raise

    def _sanitize_messages(self, messages):
        """Convert messages to JSON-serializable format."""
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append({
                    "role": msg.get("role"),
                    "content": self._sanitize_content(msg.get("content"))
                })
        return result

    def _sanitize_content(self, content):
        """Convert content to JSON-serializable format."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return [
                {
                    "type": block.get("type") if isinstance(block, dict) else getattr(block, "type", "unknown"),
                    "text": block.get("text") if isinstance(block, dict) else getattr(block, "text", None),
                    "id": block.get("id") if isinstance(block, dict) else getattr(block, "id", None),
                    "name": block.get("name") if isinstance(block, dict) else getattr(block, "name", None),
                    "input": block.get("input") if isinstance(block, dict) else getattr(block, "input", None),
                }
                for block in content
            ]
        else:
            return str(content)

    def _sanitize_system(self, system):
        """Convert system prompt to JSON-serializable format."""
        if isinstance(system, str):
            return system
        elif isinstance(system, list):
            # Anthropic caching format
            return [
                {
                    "type": item.get("type", "text"),
                    "text": item.get("text", "")[:500] + "..." if len(item.get("text", "")) > 500 else item.get("text", ""),
                    "cache_control": item.get("cache_control")
                }
                for item in system
            ]
        else:
            return str(system)

    def get_provider_name(self):
        return self.wrapped.get_provider_name()

    def get_model_name(self):
        return self.wrapped.get_model_name()


class TracingToolAdapter:
    """Wrapper around tool adapter that traces all executions."""

    def __init__(self, wrapped_adapter, trace_list: List[ToolCall]):
        self.wrapped = wrapped_adapter
        self.trace_list = trace_list

    async def execute(self, tool_name: str, inputs: Dict, agent_name: str = "unknown"):
        """Trace tool execution and forward to wrapped adapter."""
        start_time = time.time()

        trace = ToolCall(
            tool_name=tool_name,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            tool_input=inputs
        )

        try:
            result = await self.wrapped.execute(tool_name, inputs, agent_name)

            trace.execution_time_ms = (time.time() - start_time) * 1000
            trace.success = result.get("success", False)
            trace.result = result.get("data")
            trace.error = result.get("error")
            trace.api_cost_usd = result.get("metadata", {}).get("api_cost_usd", 0.0)

            self.trace_list.append(trace)

            return result

        except Exception as e:
            trace.execution_time_ms = (time.time() - start_time) * 1000
            trace.success = False
            trace.error = str(e)
            self.trace_list.append(trace)
            raise

    def __getattr__(self, name):
        """Forward all other methods to wrapped adapter."""
        return getattr(self.wrapped, name)


# ============================================================================
# Trace Runner
# ============================================================================

async def trace_workflow(query: str, config_path: Optional[Path] = None) -> WorkflowTrace:
    """
    Execute workflow with complete tracing.

    Args:
        query: User query
        config_path: Optional path to config.json (default: project root)

    Returns:
        WorkflowTrace with complete execution details
    """
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.json"

    with open(config_path) as f:
        config = json.load(f)

    # Initialize trace
    trace = WorkflowTrace(
        query=query,
        timestamp=datetime.now().isoformat()
    )

    workflow_start = time.time()

    try:
        # Initialize runner
        runner = MultiAgentRunner(config)
        await runner.initialize()

        # Patch providers and tool adapter for tracing
        _patch_agents_for_tracing(runner, trace)

        # Run workflow
        final_result = None
        async for event in runner.run_query(query, stream_progress=True):
            if event.get("type") == "final":
                final_result = event
                break

        # Extract results
        if final_result:
            trace.final_answer = final_result.get("final_answer", "")
            trace.complexity_score = final_result.get("complexity_score", 0)
            trace.query_type = final_result.get("query_type", "")
            trace.agent_sequence = final_result.get("agent_sequence", [])
            trace.success = final_result.get("success", True)
            trace.errors = final_result.get("errors", [])

        trace.total_time_ms = (time.time() - workflow_start) * 1000

        # Calculate total cost
        trace.total_cost_usd = sum(
            agent.total_cost_usd for agent in trace.agents
        )
        if trace.orchestrator_routing:
            trace.total_cost_usd += trace.orchestrator_routing.total_cost_usd
        if trace.orchestrator_synthesis:
            trace.total_cost_usd += trace.orchestrator_synthesis.total_cost_usd

    except Exception as e:
        trace.success = False
        trace.errors.append(f"Workflow failed: {str(e)}")
        trace.total_time_ms = (time.time() - workflow_start) * 1000

    return trace


def _patch_agents_for_tracing(runner: MultiAgentRunner, trace: WorkflowTrace):
    """Patch all agents with tracing wrappers."""
    # Get all registered agents
    all_agents = runner.agent_registry.get_all_agents()
    agent_names = [agent.config.name for agent in all_agents]

    for agent_name in agent_names:
        agent = runner.agent_registry.get_agent(agent_name)

        if not hasattr(agent, 'provider'):
            continue

        # Create agent trace
        agent_trace = AgentTrace(
            agent_name=agent_name,
            start_time=datetime.now().isoformat()
        )

        # Patch provider
        original_provider = agent.provider
        agent.provider = TracingProvider(
            wrapped_provider=original_provider,
            agent_name=agent_name,
            trace_list=agent_trace.llm_calls
        )

        # Store reference for later extraction
        if agent_name == "orchestrator":
            # Orchestrator has 2 phases - we'll split traces later
            trace.orchestrator_routing = agent_trace
        else:
            trace.agents.append(agent_trace)

    # Patch tool adapter (global singleton)
    from src.multi_agent.tools.adapter import get_tool_adapter
    tool_adapter = get_tool_adapter()

    # Create shared tool trace list for all agents
    all_tool_calls = []

    # Wrap execute method
    original_execute = tool_adapter.execute

    async def traced_execute(tool_name: str, inputs: Dict, agent_name: str = "unknown"):
        start_time = time.time()

        tool_trace = ToolCall(
            tool_name=tool_name,
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            tool_input=inputs
        )

        try:
            result = await original_execute(tool_name, inputs, agent_name)

            tool_trace.execution_time_ms = (time.time() - start_time) * 1000
            tool_trace.success = result.get("success", False)
            tool_trace.result = result.get("data")
            tool_trace.error = result.get("error")
            tool_trace.api_cost_usd = result.get("metadata", {}).get("api_cost_usd", 0.0)

            # Add to agent's tool calls
            for agent_trace in trace.agents:
                if agent_trace.agent_name == agent_name:
                    agent_trace.tool_calls.append(tool_trace)
                    break
            else:
                # Orchestrator or unknown agent
                if trace.orchestrator_routing and agent_name == "orchestrator":
                    trace.orchestrator_routing.tool_calls.append(tool_trace)

            return result

        except Exception as e:
            tool_trace.execution_time_ms = (time.time() - start_time) * 1000
            tool_trace.success = False
            tool_trace.error = str(e)

            # Add to agent's tool calls
            for agent_trace in trace.agents:
                if agent_trace.agent_name == agent_name:
                    agent_trace.tool_calls.append(tool_trace)
                    break

            raise

    tool_adapter.execute = traced_execute


# ============================================================================
# JSON Serialization
# ============================================================================

def trace_to_dict(trace: WorkflowTrace) -> Dict[str, Any]:
    """Convert trace to JSON-serializable dict."""
    return {
        "query": trace.query,
        "timestamp": trace.timestamp,
        "success": trace.success,
        "errors": trace.errors,

        "orchestrator": {
            "routing": asdict(trace.orchestrator_routing) if trace.orchestrator_routing else None,
            "synthesis": asdict(trace.orchestrator_synthesis) if trace.orchestrator_synthesis else None,
        },

        "agents": [asdict(agent) for agent in trace.agents],

        "results": {
            "final_answer": trace.final_answer,
            "complexity_score": trace.complexity_score,
            "query_type": trace.query_type,
            "agent_sequence": trace.agent_sequence,
        },

        "statistics": {
            "total_time_ms": trace.total_time_ms,
            "total_cost_usd": trace.total_cost_usd,
            "llm_calls": sum(len(a.llm_calls) for a in trace.agents) +
                        (len(trace.orchestrator_routing.llm_calls) if trace.orchestrator_routing else 0),
            "tool_calls": sum(len(a.tool_calls) for a in trace.agents) +
                         (len(trace.orchestrator_routing.tool_calls) if trace.orchestrator_routing else 0),
        }
    }


# ============================================================================
# CLI
# ============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Trace agent workflow execution")
    parser.add_argument("query", help="Query to execute")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--config", help="Path to config.json (default: project root)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run trace
    print(f"🔍 Tracing workflow for query: {args.query}")
    print("=" * 80)

    config_path = Path(args.config) if args.config else None
    trace = await trace_workflow(args.query, config_path)

    # Convert to dict
    trace_dict = trace_to_dict(trace)

    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trace_dict, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Trace saved to {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        print(json.dumps(trace_dict, indent=2, ensure_ascii=False))

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TRACE SUMMARY")
    print("=" * 80)
    print(f"Success: {'✅' if trace.success else '❌'}")
    print(f"Total time: {trace.total_time_ms:.0f}ms")
    print(f"Total cost: ${trace.total_cost_usd:.6f}")
    print(f"LLM calls: {trace_dict['statistics']['llm_calls']}")
    print(f"Tool calls: {trace_dict['statistics']['tool_calls']}")
    print(f"Agent sequence: {', '.join(trace.agent_sequence)}")

    if trace.errors:
        print(f"\n❌ Errors ({len(trace.errors)}):")
        for error in trace.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    asyncio.run(main())
