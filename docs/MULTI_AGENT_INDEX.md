# SUJBOT2 Multi-Agent System Documentation Index

Welcome to the comprehensive documentation for the SUJBOT2 multi-agent system. This guide will help you understand, navigate, and work with the architecture.

## Quick Navigation

### For New Users
Start here if you're new to the multi-agent system:
1. **[Quick Reference Guide](multi_agent_quick_reference.md)** - Get started in 5 minutes
2. **[Architecture Overview](multi_agent_architecture.md)** - Understand how it works (20 min read)
3. **[Component Interactions](multi_agent_components.md)** - Deep dive into components

### For Developers
If you're implementing features or debugging:
1. **[Component Interactions](multi_agent_components.md)** - See detailed execution flows
2. **[Architecture Overview - Section 18](multi_agent_architecture.md#18-key-design-decisions)** - Design patterns
3. **[Quick Reference - Common Patterns](multi_agent_quick_reference.md#common-patterns)** - Code examples

### For System Architects
If you're designing or extending the system:
1. **[Architecture Overview](multi_agent_architecture.md)** - Complete specification
2. **[Component Interactions - Summary](multi_agent_components.md#summary)** - System design
3. **[Architecture Overview - Section 18](multi_agent_architecture.md#18-key-design-decisions)** - Design decisions

---

## Documentation Files

### 1. Quick Reference Guide (335 lines)
**File:** `multi_agent_quick_reference.md`

**Contents:**
- Quick start commands
- Architecture overview
- The 8 agents
- Key files and locations
- State flow
- Common patterns (adding agents, using tools)
- Debugging tips
- Configuration examples
- Performance tips
- Troubleshooting table
- Integration summary

**Best for:** Quick lookups, code snippets, debugging

---

### 2. Full Architecture Overview (1273 lines)
**File:** `multi_agent_architecture.md`

**Sections:**
1. Executive Summary
2. Architecture Overview (high-level flow, tech stack)
3. Core Components (runner, state, agent base, registry)
4. Agent Architecture (8 agents in detail)
5. Routing & Workflow Management (complexity analyzer, workflow builder)
6. Tool Integration (adapter pattern)
7. State Persistence & Checkpointing
8. Caching System (3-level caching)
9. Observability & LangSmith Integration
10. Prompt Management
11. Execution Flow (complete pipeline)
12. Configuration
13. Error Handling & Recovery
14. Integration with Existing RAG Pipeline
15. Performance Characteristics
16. Testing & Debugging
17. Architecture Diagrams
18. Key Design Decisions
19. Future Enhancements
20. Troubleshooting

**Best for:** Understanding the complete system, research-backed decisions, detailed flows

---

### 3. Component Interactions (558 lines)
**File:** `multi_agent_components.md`

**Sections:**
1. Directory Structure (full file tree)
2. Component Dependency Graph
3. Execution Flow - 5 Phases:
   - Phase 1: Initialization
   - Phase 2: Query Routing (Orchestrator)
   - Phase 3: Workflow Construction
   - Phase 4: Workflow Execution (Agent Processing)
   - Phase 5: State Persistence & Result Return
4. State Transitions
5. Tool Execution Flow
6. Registry Operations
7. Error Handling Flow
8. Summary

**Best for:** Understanding how components interact, tracing execution flows, debugging

---

## Architecture Highlights

### The 8 Agents

| Agent | Role | Responsibility |
|-------|------|-----------------|
| **Orchestrator** | Root coordinator | Analyze complexity, route query |
| **Extractor** | Document retriever | Hybrid search, context expansion |
| **Classifier** | Content categorizer | Classify documents, detect domain |
| **Compliance** | Regulatory checker | GDPR/CCPA/HIPAA/SOX verification |
| **Risk Verifier** | Risk assessor | Identify hazards, assess impact |
| **Citation Auditor** | Citation validator | Validate sources, verify provenance |
| **Gap Synthesizer** | Gap analyzer | Find knowledge gaps, synthesis |
| **Report Generator** | Report compiler | Create final Markdown report |

### Execution Patterns

```
Simple (complexity < 30):
  Extractor → Report Generator

Standard (complexity 30-70):
  Extractor → Classifier → Domain Agent → Report Generator

Complex (complexity > 70):
  All 8 agents in sequence
```

### Technology Stack

- **Framework:** LangGraph
- **LLM:** Anthropic Claude
- **Persistence:** PostgreSQL + LangGraph Checkpointer
- **Caching:** 3-level (Prompt, Semantic, System)
- **Observability:** LangSmith
- **Tools:** 15 specialized tools (existing RAG infrastructure, filtered_search and similarity_search removed)

---

## Key Concepts

### State Management (MultiAgentState)
Central state object passed through LangGraph workflow. Contains:
- Input query
- Routing decisions (complexity, type, sequence)
- Agent outputs
- Documents and citations
- Final answer
- Cost tracking
- Error log

### Agent Template Pattern
```python
class BaseAgent(ABC):
    async def execute(state):          # Public interface
        # Timing, error handling, cost tracking
        return await self.execute_impl(state)
    
    async def execute_impl(state):     # Implement in subclass
        # Your agent logic
        # Tool execution, state updates
```

### Tool Adapter Pattern
Bridges LangGraph agents to existing 15-tool infrastructure:
```
Agent → ToolAdapter → Existing Tool Registry → RAG Infrastructure
```

### Complexity-Based Routing
```
Query → Orchestrator → Complexity Score (0-100) → Agent Sequence
          ├─ Analysis via LLM
          └─ Fallback to keyword matching
```

---

## Common Tasks

### Running the System
```bash
# Single query
uv run python -m src.multi_agent.runner --query "your question"

# Interactive
uv run python -m src.multi_agent.runner --interactive

# Debug mode
uv run python -m src.multi_agent.runner --query "q" --debug
```

### Adding a New Agent
See: [Quick Reference - Adding a New Agent](multi_agent_quick_reference.md#adding-a-new-agent)

### Using Tools in an Agent
See: [Quick Reference - Using Tools](multi_agent_quick_reference.md#using-tools)

### Debugging Issues
See: [Quick Reference - Debugging](multi_agent_quick_reference.md#debugging)

---

## File Organization

```
docs/
├── MULTI_AGENT_INDEX.md (this file)
├── multi_agent_quick_reference.md (quick lookup)
├── multi_agent_architecture.md (complete spec)
└── multi_agent_components.md (component details)

src/multi_agent/
├── runner.py (entry point)
├── core/
│   ├── agent_base.py (BaseAgent with update_state_output helper)
│   ├── agent_initializer.py (SSOT for agent initialization)
│   ├── agent_registry.py (agent registration)
│   └── state.py (MultiAgentState schema)
├── agents/ (8 specialized agents)
├── routing/ (complexity, workflow builder)
├── tools/ (tool adapter)
├── checkpointing/ (PostgreSQL persistence)
├── caching/ (3-level caching)
├── prompts/ (prompt loading)
└── observability/ (LangSmith)

src/
├── exceptions.py (typed exception hierarchy)
└── utils/
    └── cache.py (LRUCache + TTLCache abstractions)

prompts/agents/
├── orchestrator.txt
├── extractor.txt
├── classifier.txt
├── compliance.txt
├── risk_verifier.txt
├── citation_auditor.txt
├── gap_synthesizer.txt
└── report_generator.txt
```

---

## Research Foundation

The architecture is based on peer-reviewed research and industry patterns:

1. **L-MARS** (Legal Multi-Agent RAG System) - Agent specialization
2. **PAKTON** (Performance-Aware Knowledge Transfer) - Tool distribution
3. **MASLegalBench** - Multi-agent evaluation benchmarks
4. **Harvey AI** - Production legal AI patterns
5. **Definely** - Document analysis architecture

See `PIPELINE.md` for complete research details.

---

## Related Documentation

- `README.md` - User guide and installation
- `PIPELINE.md` - 7-phase indexing pipeline specification
- `CLAUDE.md` - Project constraints and guidelines
- `INSTALL.md` - Platform-specific setup
- `config.json.example` - Configuration reference

---

## Quick Links by Topic

### Architecture & Design
- [Architecture Overview](multi_agent_architecture.md)
- [Component Interactions](multi_agent_components.md#component-dependency-graph)
- [Design Decisions](multi_agent_architecture.md#18-key-design-decisions)

### The Agents
- [All 8 Agents Table](multi_agent_architecture.md#agent-characteristics-table)
- [Orchestrator](multi_agent_architecture.md#41-orchestrator-agent)
- [Extractor](multi_agent_architecture.md#42-extractor-agent)
- [Classifier](multi_agent_architecture.md#43-classifier-agent)
- [Compliance](multi_agent_architecture.md#44-compliance-agent)

### Workflow & Execution
- [Complete Flow Diagram](multi_agent_architecture.md#11-execution-flow-step-by-step)
- [State Transitions](multi_agent_components.md#state-transitions)
- [Initialization Flow](multi_agent_components.md#phase-1-initialization)
- [Query Routing](multi_agent_components.md#phase-2-query-routing-orchestrator)

### Tools & Integration
- [Tool Adapter Pattern](multi_agent_architecture.md#61-tool-adapter-tooldapy)
- [Tool Execution Flow](multi_agent_components.md#tool-execution-flow)
- [RAG Integration](multi_agent_architecture.md#14-integration-with-existing-rag-pipeline)
- [All 15 Tools List](multi_agent_architecture.md#61-tool-adapter-tooldapy)

### Configuration & Deployment
- [Configuration](multi_agent_architecture.md#12-configuration)
- [Running the System](multi_agent_quick_reference.md#quick-start)
- [Debug Mode](multi_agent_quick_reference.md#debugging)
- [Performance Tips](multi_agent_quick_reference.md#performance-tips)

### Troubleshooting
- [Troubleshooting Table](multi_agent_quick_reference.md#troubleshooting)
- [Error Handling](multi_agent_architecture.md#13-error-handling--recovery)
- [Debugging Tips](multi_agent_quick_reference.md#debugging)

---

## Development Workflow

### 1. Understanding the System
```
Read: Quick Reference → Architecture Overview → Component Interactions
Time: 30 minutes
```

### 2. Making Changes
```
Identify: Which component? → Read relevant section → Check dependencies
Implement: Add/modify code → Test → Check error handling
Document: Update prompts/docstrings as needed
```

### 3. Adding a New Agent
```
Read: Quick Reference - Adding a New Agent
Create: Agent file + Prompt file
Register: @register_agent() decorator
Test: Run query that uses new agent
```

### 4. Debugging Issues
```
Enable: --debug flag
Check: Logs in debug output
Trace: Use LangSmith for workflow visualization
Review: Component interactions diagram
```

---

## Performance Characteristics

| Component | Time | Cost |
|-----------|------|------|
| Orchestrator | 200-500ms | 200-500 tokens |
| Extractor | 500-800ms | ~500 tokens |
| Classifier | 300-400ms | ~300 tokens |
| Compliance | 800-1200ms | ~800 tokens |
| Report Generator | 1-2s | ~1000 tokens |
| **Total (complex)** | **3-8s** | **3000-3500 tokens** |
| **With prompt cache** | Same | **90% reduction** |

---

## Glossary

- **Agent** - Specialized LLM-based component with distinct role
- **State** - MultiAgentState object passed through workflow
- **Tool** - Atomic action that agents can execute (search, verify, etc.)
- **Workflow** - LangGraph state graph connecting agents
- **Checkpointing** - State persistence in PostgreSQL
- **Prompt Caching** - Anthropic API feature for cost reduction (90%)
- **Complexity Score** - 0-100 scoring of query difficulty
- **Tool Adapter** - Bridge between agents and tool infrastructure

---

## Getting Help

1. **First issue?** Check [Quick Reference - Troubleshooting](multi_agent_quick_reference.md#troubleshooting)
2. **Architectural question?** Read [Architecture Overview](multi_agent_architecture.md)
3. **How does X work?** Check [Component Interactions](multi_agent_components.md)
4. **Code example?** See [Quick Reference - Common Patterns](multi_agent_quick_reference.md#common-patterns)

---

## Version Information

- **System Version:** 2.1.0 (Multi-Agent + SSOT Refactoring)
- **Last Updated:** 2025-11-26
- **Status:** Production Ready
- **Branch:** main

---

## Next Steps

1. **Read:** [Quick Reference Guide](multi_agent_quick_reference.md) (5 min)
2. **Run:** `uv run python -m src.multi_agent.runner --query "test query"`
3. **Explore:** [Architecture Overview](multi_agent_architecture.md) (20 min)
4. **Develop:** Check [Component Interactions](multi_agent_components.md) as needed

Happy exploring!
