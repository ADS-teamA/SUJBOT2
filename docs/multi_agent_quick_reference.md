# SUJBOT2 Multi-Agent System - Quick Reference Guide

## Quick Start

### Run the Multi-Agent System

```bash
# Single query
uv run python -m src.multi_agent.runner --query "your question here"

# Interactive mode
uv run python -m src.multi_agent.runner --interactive

# Debug mode (verbose logging)
uv run python -m src.multi_agent.runner --query "question" --debug
```

### Prerequisites
- `config.json` with `api_keys.anthropic_api_key` set
- Agent prompts in `prompts/agents/*.txt`
- Indexed documents in vector store
- PostgreSQL running (optional, for checkpointing)

---

## Architecture at a Glance

```
Query → Orchestrator (complexity analysis)
     → Workflow Builder (construct LangGraph)
     → Agent Sequence Execution (2-8 agents)
     → Report Generator (compile results)
     → Final Markdown Report
```

### The 8 Agents

1. **Orchestrator** - Routes query based on complexity (0-100)
2. **Extractor** - Retrieves documents via hybrid search
3. **Classifier** - Categorizes content and domain
4. **Compliance** - Checks GDPR/CCPA/HIPAA/SOX
5. **Risk Verifier** - Assesses risks and hazards
6. **Citation Auditor** - Validates sources and citations
7. **Gap Synthesizer** - Analyzes knowledge gaps
8. **Report Generator** - Creates final Markdown report

### Execution Patterns

**Simple** (complexity < 30)
- Extractor → Report Generator

**Standard** (complexity 30-70)
- Extractor → Classifier → Domain Agent → Report Generator

**Complex** (complexity > 70)
- All 8 agents in sequence

---

## Key Files & Locations

| File | Purpose | Key Class/Function |
|------|---------|-----------|
| `src/multi_agent/runner.py` | Main entry point | `MultiAgentRunner` |
| `src/multi_agent/core/state.py` | State schema | `MultiAgentState` |
| `src/multi_agent/core/agent_base.py` | Base agent class | `BaseAgent` |
| `src/multi_agent/core/agent_initializer.py` | **SSOT** agent init | `initialize_agent()` |
| `src/multi_agent/core/agent_registry.py` | Agent registry | `AgentRegistry` |
| `src/exceptions.py` | Typed exceptions | `SujbotError`, `APIKeyError` |
| `src/utils/cache.py` | Cache abstractions | `LRUCache`, `TTLCache` |
| `src/multi_agent/agents/orchestrator.py` | Routing logic | `OrchestratorAgent` |
| `src/multi_agent/agents/extractor.py` | Document retrieval | `ExtractorAgent` |
| `src/multi_agent/agents/classifier.py` | Content categorization | `ClassifierAgent` |
| `src/multi_agent/agents/compliance.py` | Regulatory checking | `ComplianceAgent` |
| `src/multi_agent/agents/risk_verifier.py` | Risk assessment | `RiskVerifierAgent` |
| `src/multi_agent/agents/citation_auditor.py` | Citation validation | `CitationAuditorAgent` |
| `src/multi_agent/agents/gap_synthesizer.py` | Gap analysis | `GapSynthesizerAgent` |
| `src/multi_agent/agents/report_generator.py` | Report compilation | `ReportGeneratorAgent` |
| `src/multi_agent/routing/complexity_analyzer.py` | Complexity scoring | `ComplexityAnalyzer` |
| `src/multi_agent/routing/workflow_builder.py` | Workflow construction | `WorkflowBuilder` |
| `src/multi_agent/tools/adapter.py` | Tool integration | `ToolAdapter` |
| `src/multi_agent/checkpointing/postgres_checkpointer.py` | State persistence | `PostgresCheckpointer` |
| `src/multi_agent/checkpointing/state_manager.py` | State operations | `StateManager` |
| `src/multi_agent/caching/cache_manager.py` | Multi-level caching | `CacheManager` |
| `src/multi_agent/prompts/loader.py` | Prompt loading | `PromptLoader` |
| `src/multi_agent/observability/langsmith_integration.py` | Tracing | `LangSmithIntegration` |

---

## State Flow

```python
# Initial state
state = MultiAgentState(
    query="What are GDPR requirements?",
    execution_phase=ExecutionPhase.ROUTING,
)

# After Orchestrator
state.complexity_score = 75
state.query_type = QueryType.COMPLIANCE
state.agent_sequence = ["extractor", "classifier", "compliance", ...]

# After each agent
state.agent_outputs["agent_name"] = {...}
state.current_agent = "agent_name"
state.execution_phase = ExecutionPhase.AGENT_EXECUTION

# After Report Generator
state.final_answer = "# Markdown Report\n..."
state.execution_phase = ExecutionPhase.COMPLETE
```

---

## Common Patterns

### Adding a New Agent

1. Create file `src/multi_agent/agents/my_agent.py`:
```python
from typing import Any, Dict
from ..core.agent_base import BaseAgent
from ..core.agent_initializer import initialize_agent  # SSOT
from ..core.agent_registry import register_agent

@register_agent("my_agent")
class MyAgent(BaseAgent):
    def __init__(self, config, vector_store=None, agent_registry=None):
        super().__init__(config)
        # Use SSOT initialization (handles provider, prompts, tools)
        components = initialize_agent(config, "my_agent")
        self.provider = components.provider
        self.system_prompt = components.system_prompt
        self.tool_adapter = components.tool_adapter

    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Your logic here
        return self.update_state_output(state, {"result": "..."})  # SSOT helper
```

2. Create prompt file `prompts/agents/my_agent.txt`:
```
You are the MY_AGENT agent. Your responsibilities are...
```

3. Import in `src/multi_agent/agents/__init__.py`:
```python
from .my_agent import MyAgent
```

4. Agent automatically registers via `@register_agent()` decorator

### Using Tools

```python
# In an agent's execute_impl()
result = await self.tool_adapter.execute(
    tool_name="search_documents",
    inputs={"query": query, "k": 10},
    agent_name=self.config.name
)

if result["success"]:
    documents = result["data"]
    citations = result.get("citations", [])
else:
    error = result.get("error")
```

### Accessing Agent Outputs

```python
# In a downstream agent
extractor_output = state.get("agent_outputs", {}).get("extractor", {})
classifier_output = state.get("agent_outputs", {}).get("classifier", {})
```

### Recording Errors

```python
state["errors"] = state.get("errors", [])
state["errors"].append(f"Error: {error_message}")
```

---

## Debugging

### Enable Debug Logging
```bash
uv run python -m src.multi_agent.runner --query "q" --debug
```

### Check State at Each Step
```python
# In an agent
print(f"Current state: {state.get('complexity_score')}")
print(f"Agent sequence: {state.get('agent_sequence')}")
```

### View LangSmith Traces
```json
// In config.json
"multi_agent": {
    "langsmith": {
        "enabled": true,
        "api_key": "YOUR_API_KEY"
    }
}
```
Then visit: https://smith.langchain.com

### Check Checkpoints
```python
# View PostgreSQL directly
SELECT * FROM langgraph_checkpoints;
```

---

## Configuration

### In config.json

```json
{
    "api_keys": {
        "anthropic_api_key": "sk-ant-..."
    },
    "models": {
        "llm_model": "claude-sonnet-4-5-20250929"
    },
    "agent": {
        "enable_prompt_caching": true
    },
    "multi_agent": {
        "orchestrator": {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 2048,
            "temperature": 0.3,
            "enable_prompt_caching": true
        },
        "agents": {
            "extractor": {
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 2048
            },
            "classifier": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024
            }
        },
        "routing": {
            "complexity_threshold_low": 30,
            "complexity_threshold_high": 70
        },
        "checkpointing": {
            "enabled": true,
            "postgres_url": "postgresql://user:pass@localhost/langgraph"
        },
        "caching": {
            "enable_prompt_cache": true,
            "enable_semantic_cache": true,
            "enable_system_cache": true
        },
        "langsmith": {
            "enabled": false,
            "api_key": null,
            "project_name": "sujbot2-multi-agent"
        }
    }
}
```

---

## Performance Tips

1. **Reduce Complexity** - Use keyword-based routing (ComplexityAnalyzer) as fallback
2. **Enable Caching** - Prompt caching reduces costs 90% for repeated queries
3. **Use Appropriate Models** - Claude Haiku for simple agents, Sonnet for complex
4. **Batch Queries** - Process multiple queries in interactive mode
5. **Monitor Costs** - Check `state.total_cost_cents` for cost tracking

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Agent not found" | Agent not registered | Add `@register_agent()` decorator |
| "Prompt file not found" | Missing prompt file | Create `prompts/agents/<name>.txt` |
| "Tool execution failed" | Tool not available | Ensure tool exists in `src.agent.tools` |
| "Connection failed" | PostgreSQL not running | Disable checkpointing or start PostgreSQL |
| "LangSmith not working" | API key missing | Set `LANGSMITH_API_KEY` env var |
| "JSON parsing failed" | Invalid LLM response | Check orchestrator system prompt |

---

## Integration with RAG Pipeline

The multi-agent system uses the existing RAG pipeline without modifications:

```
Multi-Agent Agents
    ↓
ToolAdapter (bridges to existing tools)
    ↓
Existing Tool Registry (src.agent.tools)
    ↓
RAG Components:
- FAISS Vector Stores (L1, L2, L3)
- BM25 Index
- Knowledge Graph
- Document Corpus
- Caching Layer
```

All 15 tools are available to agents automatically (filtered_search and similarity_search were unified into search).

---

## Research Papers

The architecture is based on:
1. **L-MARS** - Legal Multi-Agent RAG System
2. **PAKTON** - Performance-Aware Knowledge Transfer
3. **MASLegalBench** - Multi-agent evaluation benchmarks
4. **Harvey AI** - Production patterns
5. **Definely** - Document analysis architecture

See `PIPELINE.md` for full research details.

---

## Further Reading

- `docs/multi_agent_architecture.md` - Full architecture documentation
- `CLAUDE.md` - Project constraints and guidelines
- `PIPELINE.md` - 7-phase indexing pipeline
- `README.md` - User guide and installation

---

**Last Updated:** 2025-11-11  
**Version:** 2.0.0 (Multi-Agent System)
