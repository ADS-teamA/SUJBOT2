# Multi-Agent System - Component Interactions

## Directory Structure

```
src/multi_agent/
├── __init__.py                          # Package initialization
├── runner.py                            # Main entry point (345 lines)
│
├── core/
│   ├── __init__.py
│   ├── state.py                         # State schema (190 lines)
│   ├── agent_base.py                    # Base agent class (279 lines)
│   └── agent_registry.py                # Agent registry (235 lines)
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py                  # Orchestrator agent (306 lines)
│   ├── extractor.py                     # Extractor agent (11.5 KB)
│   ├── classifier.py                    # Classifier agent (~100 lines)
│   ├── compliance.py                    # Compliance agent (~100 lines)
│   ├── risk_verifier.py                 # Risk verifier agent (7.3 KB)
│   ├── citation_auditor.py              # Citation auditor agent (9.0 KB)
│   ├── gap_synthesizer.py               # Gap synthesizer agent (8.7 KB)
│   └── report_generator.py              # Report generator agent (9.6 KB)
│
├── routing/
│   ├── __init__.py
│   ├── complexity_analyzer.py           # Complexity analysis (351 lines)
│   └── workflow_builder.py              # Workflow construction (277 lines)
│
├── tools/
│   ├── __init__.py
│   └── adapter.py                       # Tool integration adapter (200+ lines)
│
├── checkpointing/
│   ├── __init__.py
│   ├── postgres_checkpointer.py         # PostgreSQL persistence (10.6 KB)
│   └── state_manager.py                 # State operations (4.8 KB)
│
├── caching/
│   ├── __init__.py
│   ├── cache_manager.py                 # Multi-level caching (7.3 KB)
│   ├── contract_cache.py                # Contract analysis cache (3.7 KB)
│   ├── regulatory_cache.py              # Regulatory findings cache (3.9 KB)
│   └── system_cache.py                  # Common patterns cache (3.5 KB)
│
├── prompts/
│   ├── __init__.py
│   └── loader.py                        # Prompt file loading (246 lines)
│
├── observability/
│   ├── __init__.py
│   └── langsmith_integration.py         # LangSmith tracing (80+ lines)
│
├── config/
│   └── (Configuration management - to be implemented)
│
├── cli/
│   └── (CLI interface - to be implemented)
│
└── integrations/
    └── (External integrations - to be implemented)

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

## Component Dependency Graph

```
MultiAgentRunner (runner.py)
    ├── depends on: LangSmithIntegration
    ├── depends on: PostgresCheckpointer
    ├── depends on: StateManager
    ├── depends on: CacheManager
    ├── depends on: AgentRegistry
    ├── depends on: ComplexityAnalyzer
    └── depends on: WorkflowBuilder
           ├── depends on: AgentRegistry
           └── depends on: PostgresCheckpointer

AgentRegistry
    ├── owns: BaseAgent instances (8 total)
    ├── owns: AgentConfig (per agent)
    ├── manages: OrchestratorAgent
    ├── manages: ExtractorAgent
    ├── manages: ClassifierAgent
    ├── manages: ComplianceAgent
    ├── manages: RiskVerifierAgent
    ├── manages: CitationAuditorAgent
    ├── manages: GapSynthesizerAgent
    └── manages: ReportGeneratorAgent

BaseAgent (Abstract)
    ├── uses: PromptLoader
    ├── uses: ToolAdapter
    ├── tracks: execution time & cost
    ├── maintains: error handling
    └── provides: execute() template method

ToolAdapter
    ├── bridges to: src.agent.tools.registry
    ├── provides: async execute()
    ├── returns: ToolResult (success, data, citations, metadata)
    └── tracks: ToolExecution in state

MultiAgentState
    ├── carries: query
    ├── carries: complexity_score
    ├── carries: agent_sequence
    ├── carries: agent_outputs
    ├── carries: documents
    ├── carries: final_answer
    ├── carries: citations
    ├── carries: total_cost_cents
    └── carries: errors

LangGraph Workflow
    ├── contains: Agent nodes (8 total)
    ├── connects: Sequential edges
    ├── handles: State transitions
    ├── integrates: PostgresCheckpointer
    └── returns: Final state with results
```

## Execution Flow - Detailed Component Interactions

### Phase 1: Initialization

```
main()
  ├─ Create MultiAgentRunner(config)
  │   └─ store config
  │
  └─ await runner.initialize()
      ├─ setup_langsmith(config)
      │   ├─ Set LANGCHAIN_TRACING_V2 env var
      │   ├─ Set LANGCHAIN_API_KEY env var
      │   └─ Set LANGCHAIN_PROJECT env var
      │
      ├─ create_checkpointer(config)
      │   └─ return PostgresCheckpointer instance
      │
      ├─ StateManager(checkpointer)
      │   └─ store checkpointer reference
      │
      ├─ create_cache_manager(config)
      │   └─ Initialize 3-level caching:
      │       ├─ Prompt cache (Anthropic API)
      │       ├─ Semantic cache (Regulatory/Contract)
      │       └─ System cache (Common patterns)
      │
      ├─ AgentRegistry()
      │   └─ Create empty registry
      │
      ├─ _register_agents()
      │   ├─ Get orchestrator config
      │   ├─ Build AgentConfig for orchestrator
      │   ├─ Create OrchestratorAgent(config)
      │   ├─ registry.register(orchestrator)
      │   │
      │   └─ For each agent in agents_config:
      │       ├─ Build AgentConfig
      │       ├─ Create Agent instance (e.g., ExtractorAgent)
      │       └─ registry.register(agent)
      │
      ├─ ComplexityAnalyzer(routing_config)
      │   └─ store routing thresholds
      │
      └─ WorkflowBuilder(agent_registry, checkpointer)
          ├─ store agent registry reference
          └─ store optional checkpointer
```

### Phase 2: Query Routing (Orchestrator)

```
runner.run_query(query)
  ├─ Create MultiAgentState(query, phase=ROUTING)
  │
  ├─ Get orchestrator agent from registry
  │   └─ registry.get_agent("orchestrator") → OrchestratorAgent instance
  │
  ├─ await orchestrator.execute(state)
  │   └─ BaseAgent.execute() template method:
  │       ├─ [TIMING] start_time = time.now()
  │       │
  │       ├─ await execute_impl(state):
  │       │   ├─ Load system prompt via PromptLoader
  │       │   ├─ Call LLM with prompt caching:
  │       │   │   ├─ api_params = {model, max_tokens, temperature, system, messages}
  │       │   │   ├─ if enable_prompt_caching:
  │       │   │   │   └─ system = [{type: "text", text: prompt, cache_control: ephemeral}]
  │       │   │   └─ response = client.messages.create(**api_params)
  │       │   │
  │       │   ├─ Extract response text
  │       │   ├─ Parse JSON response (routing decision)
  │       │   ├─ Validate routing decision
  │       │   ├─ Update state:
  │       │   │   ├─ state.complexity_score = 65 (0-100)
  │       │   │   ├─ state.query_type = "compliance"
  │       │   │   └─ state.agent_sequence = ["extractor", "classifier", "compliance", "report_generator"]
  │       │   │
  │       │   └─ Return updated state
  │       │
  │       ├─ [TIMING] elapsed_ms = (time.now() - start_time) * 1000
  │       ├─ [TRACKING] total_time_ms += elapsed_ms
  │       │
  │       └─ Return state
  │
  └─ state.agent_sequence now contains ["extractor", "classifier", "compliance", "report_generator"]
```

### Phase 3: Workflow Construction

```
WorkflowBuilder.build_workflow(agent_sequence)
  ├─ Create StateGraph(MultiAgentState)
  │   └─ graph = StateGraph(state_schema=MultiAgentState)
  │
  ├─ For each agent_name in agent_sequence:
  │   ├─ _add_agent_node(graph, agent_name):
  │   │   ├─ Get agent from registry:
  │   │   │   └─ agent = registry.get_agent("extractor")
  │   │   │
  │   │   ├─ Define async agent_node function:
  │   │   │   ├─ state["execution_phase"] = AGENT_EXECUTION
  │   │   │   ├─ state["current_agent"] = agent_name
  │   │   │   ├─ Call agent.execute(state):
  │   │   │   │   └─ (See Phase 4 below)
  │   │   │   └─ Return updated_state
  │   │   │
  │   │   └─ graph.add_node(agent_name, agent_node)
  │   │
  │   └─ [Repeat for each agent]
  │
  ├─ _add_workflow_edges(graph, agent_sequence):
  │   ├─ Connect sequential edges:
  │   │   ├─ graph.add_edge("extractor", "classifier")
  │   │   ├─ graph.add_edge("classifier", "compliance")
  │   │   ├─ graph.add_edge("compliance", "report_generator")
  │   │   └─ graph.add_edge("report_generator", END)
  │   │
  │   └─ [Sequential execution by default]
  │
  ├─ Set entry point:
  │   └─ graph.set_entry_point("extractor")
  │
  ├─ Compile with optional checkpointer:
  │   └─ if checkpointer:
  │       └─ compiled = graph.compile(checkpointer=checkpointer)
  │       else:
  │           └─ compiled = graph.compile()
  │
  └─ Return compiled workflow
```

### Phase 4: Workflow Execution (Agent Processing)

```
workflow.ainvoke(state, config={"thread_id": thread_id})
  │
  └─ For each node in the graph (sequential):
      │
      ├─ AGENT 1: EXTRACTOR
      │   └─ extractor.execute(state):
      │       ├─ Load system prompt for extractor
      │       ├─ Determine k from complexity_score
      │       ├─ await tool_adapter.execute("hybrid_search", ...):
      │       │   ├─ Get tool from tool registry
      │       │   ├─ Execute tool.execute(**inputs)
      │       │   ├─ Measure duration_ms
      │       │   ├─ Create ToolExecution record
      │       │   └─ Return {success, data, citations, metadata, error}
      │       │
      │       ├─ await tool_adapter.execute("expand_context", ...):
      │       │   └─ Similar execution
      │       │
      │       ├─ await tool_adapter.execute("get_document_metadata", ...):
      │       │   └─ Similar execution
      │       │
      │       ├─ Update state:
      │       │   ├─ state.documents = [DocumentMetadata(...), ...]
      │       │   ├─ state.retrieved_text = "text content"
      │       │   ├─ state.citations = ["citation1", "citation2", ...]
      │       │   ├─ state.agent_outputs["extractor"] = {...}
      │       │   ├─ state.tool_executions.append(ToolExecution(...))
      │       │   └─ state.total_cost_cents += calculated_cost
      │       │
      │       └─ Return updated state → flows to next node
      │
      ├─ AGENT 2: CLASSIFIER
      │   └─ classifier.execute(state):
      │       ├─ Get extractor_output from state.agent_outputs["extractor"]
      │       ├─ await tool_adapter.execute("classify_document", ...):
      │       │   └─ Execute classification
      │       │
      │       ├─ Update state:
      │       │   ├─ state.agent_outputs["classifier"] = {...}
      │       │   └─ state.total_cost_cents += cost
      │       │
      │       └─ Return updated state → flows to next node
      │
      ├─ AGENT 3: COMPLIANCE
      │   └─ compliance.execute(state):
      │       ├─ Identify framework from query
      │       ├─ await tool_adapter.execute("search_regulatory_graph", ...):
      │       │   └─ Search regulatory knowledge graph
      │       │
      │       ├─ await tool_adapter.execute("verify_compliance", ...):
      │       │   └─ Verify compliance
      │       │
      │       ├─ await tool_adapter.execute("assess_confidence", ...):
      │       │   └─ Get confidence score
      │       │
      │       ├─ Update state:
      │       │   ├─ state.agent_outputs["compliance"] = {violations, gaps, confidence}
      │       │   └─ state.total_cost_cents += cost
      │       │
      │       └─ Return updated state → flows to next node
      │
      └─ AGENT 4: REPORT GENERATOR
          └─ report_generator.execute(state):
              ├─ Collect all agent_outputs
              ├─ await tool_adapter.execute("format_report", ...):
              │   ├─ Compile findings
              │   ├─ Create executive summary
              │   ├─ Generate compliance matrix
              │   ├─ Summarize risks
              │   ├─ Consolidate citations
              │   ├─ Prioritize recommendations
              │   └─ Create appendix with metadata
              │
              ├─ Update state:
              │   ├─ state.final_answer = "# Markdown Report\n..."
              │   ├─ state.execution_phase = COMPLETE
              │   └─ state.total_cost_cents += cost
              │
              └─ Return final state → END node
```

### Phase 5: State Persistence & Result Return

```
Final state from workflow
  │
  ├─ Save checkpoint (if checkpointing enabled):
  │   └─ checkpoint_saver.put(config, state, metadata)
  │       ├─ Serialize state
  │       ├─ Store in PostgreSQL
  │       └─ Return checkpoint ID
  │
  ├─ Extract results:
  │   ├─ final_answer = state.final_answer
  │   ├─ complexity = state.complexity_score
  │   ├─ agent_sequence = state.agent_sequence
  │   ├─ documents = state.documents
  │   ├─ citations = state.citations
  │   ├─ cost = state.total_cost_cents
  │   └─ errors = state.errors
  │
  └─ Return to user:
      └─ {
          success: true,
          final_answer: "# Report...",
          complexity_score: 65,
          query_type: "compliance",
          agent_sequence: [...],
          documents: [...],
          citations: [...],
          total_cost_cents: 12.5,
          errors: []
      }
```

## State Transitions

```
MultiAgentState lifecycle:
│
├─ Initial: ExecutionPhase.ROUTING
│   ├─ query: "..."
│   ├─ complexity_score: 0 (default)
│   ├─ agent_sequence: [] (empty)
│
├─ After Orchestrator: ExecutionPhase.AGENT_EXECUTION
│   ├─ complexity_score: 65 (updated)
│   ├─ query_type: "compliance" (updated)
│   ├─ agent_sequence: ["extractor", ...] (updated)
│   ├─ agent_outputs["orchestrator"] = {...}
│
├─ After Extractor: ExecutionPhase.AGENT_EXECUTION
│   ├─ current_agent: "extractor"
│   ├─ documents: [DocumentMetadata(...), ...]
│   ├─ agent_outputs["extractor"] = {...}
│   ├─ tool_executions: [ToolExecution(...), ...]
│   ├─ total_cost_cents: 2.5 (increased)
│
├─ After Classifier: ExecutionPhase.AGENT_EXECUTION
│   ├─ current_agent: "classifier"
│   ├─ agent_outputs["classifier"] = {...}
│   ├─ total_cost_cents: 3.8 (increased)
│
├─ ... (similar for other agents)
│
└─ After Report Generator: ExecutionPhase.COMPLETE
    ├─ current_agent: "report_generator"
    ├─ final_answer: "# Markdown Report\n..."
    ├─ agent_outputs["report_generator"] = {...}
    ├─ total_cost_cents: 12.5 (final)
    └─ errors: [] (or [...] if any occurred)
```

## Tool Execution Flow

```
Agent.execute_impl()
  │
  └─ await tool_adapter.execute(tool_name, inputs, agent_name):
      │
      ├─ [LOOKUP] Get tool from existing registry:
      │   └─ tool = self.registry.get_tool(tool_name)
      │       └─ returns: Tool instance from src.agent.tools
      │
      ├─ [TIMING] start_time = now()
      │
      ├─ [EXECUTION] Execute tool:
      │   └─ result: ToolResult = tool.execute(**inputs)
      │       ├─ Input validation (Pydantic)
      │       ├─ Tool logic execution
      │       ├─ Error handling
      │       └─ Result formatting
      │
      ├─ [TRACKING] duration_ms = (now() - start_time) * 1000
      │
      ├─ [RECORDING] Create ToolExecution:
      │   └─ execution = ToolExecution(
      │       tool_name=tool_name,
      │       agent_name=agent_name,
      │       timestamp=start_time,
      │       duration_ms=duration_ms,
      │       input_tokens=...,
      │       output_tokens=...,
      │       success=result.success,
      │       error=result.error,
      │       result_summary=result.data[:200]
      │   )
      │
      ├─ [UPDATING STATE] state.tool_executions.append(execution)
      │
      ├─ [COST TRACKING] Add to total_cost:
      │   └─ state.total_cost_cents += (input_tokens + output_tokens) * 0.000003
      │
      └─ Return formatted result:
          └─ {
              success: result.success,
              data: result.data,
              citations: result.citations,
              metadata: result.metadata,
              error: result.error
          }
```

## Registry Operations

```
AgentRegistry Lifecycle:
│
├─ Import time:
│   └─ @register_agent("orchestrator")
│       └─ _registry.register_agent_class("orchestrator", OrchestratorAgent)
│
├─ runner.initialize():
│   ├─ For each agent:
│   │   ├─ registry.register_config(agent_name, config)
│   │   └─ _configs[agent_name] = AgentConfig(...)
│   │
│   └─ registry._agent_classes and registry._configs now populated
│
└─ workflow.build_workflow():
    └─ For each agent_name in sequence:
        └─ agent = registry.get_agent(agent_name):
            ├─ Check if already instantiated:
            │   └─ if agent_name in _agent_instances:
            │       └─ return cached instance
            │
            ├─ Otherwise, instantiate:
            │   ├─ agent_class = _agent_classes[agent_name]
            │   ├─ config = _configs[agent_name]
            │   ├─ instance = agent_class(config)
            │   ├─ _agent_instances[agent_name] = instance
            │   └─ return instance
            │
            └─ Agent instance ready for use
```

## Error Handling Flow

```
Agent.execute() (Template Method)
  │
  ├─ try:
  │   └─ await execute_impl(state)
  │
  └─ except Exception as e:
      └─ await handle_error(e, state):
          │
          ├─ [ERROR RECORDING]
          │   ├─ error_message = f"{agent_name}: {error_type}: {error}"
          │   ├─ state.errors.append(error_message)
          │   └─ self.error_count += 1
          │
          ├─ [STATE MARKING]
          │   └─ state.execution_phase = "error"
          │
          ├─ [LOGGING]
          │   └─ logger.error(f"Error in agent {agent_name}: {error}")
          │
          └─ Return state (non-blocking error)

Workflow level:
  │
  ├─ WorkflowBuilder._add_agent_node()
  │   └─ try-except around agent.execute():
  │       └─ Continue to next agent even if one fails
  │
  └─ Runner level:
      └─ run_query()
          ├─ try-except around workflow execution
          └─ Return error in result dict
```

---

## Summary

The multi-agent system is built on a modular architecture where:

1. **MultiAgentRunner** orchestrates initialization and query execution
2. **AgentRegistry** manages agent lifecycle and discovery
3. **ComplexityAnalyzer** determines routing based on query analysis
4. **WorkflowBuilder** constructs LangGraph workflows dynamically
5. **BaseAgent** provides template method pattern for execution
6. **ToolAdapter** bridges to existing 17-tool infrastructure
7. **MultiAgentState** carries data through the workflow
8. **PostgresCheckpointer** persists state for recovery
9. **CacheManager** optimizes costs via multi-level caching
10. **LangSmithIntegration** provides full observability

All components work together to provide a flexible, scalable, and observable multi-agent RAG system.
