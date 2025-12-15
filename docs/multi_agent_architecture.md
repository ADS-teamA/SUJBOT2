# SUJBOT2 Multi-Agent System Architecture Overview

**Last Updated:** 2025-12-15
**Status:** Production Ready (Phase 7 + SSOT Refactoring)
**Entry Point:** `src/multi_agent/runner.py`

---

## Executive Summary

The SUJBOT2 multi-agent system is a LangGraph-based orchestration framework that coordinates 8 specialized agents to process complex legal/technical queries. The system implements research-backed patterns from Harvey AI, Definely, and academic papers (L-MARS, PAKTON, MASLegalBench).

### Key Characteristics:
- **8 Specialized Agents** with distinct responsibilities and toolsets
- **Adaptive Routing** based on query complexity (0-100 score)
- **Per-Agent Configuration** for cost optimization
- **PostgreSQL Checkpointing** for conversation persistence
- **3-Level Prompt Caching** (90% cost savings)
- **LangSmith Integration** for full observability
- **Zero-Change Integration** with existing 15-tool RAG infrastructure

---

## 1. Architecture Overview

### High-Level Flow

```
User Query
    ↓
[ORCHESTRATOR AGENT]
    ├─ Analyze complexity (0-100)
    ├─ Determine query type
    ├─ Build agent sequence
    └─ Output: routing_decision
         ├─ complexity_score
         ├─ query_type
         └─ agent_sequence
    ↓
[COMPLEXITY ANALYZER]
    └─ Validates LLM routing decision
    └─ Fallback to keyword-based routing if needed
    ↓
[WORKFLOW BUILDER]
    └─ Constructs LangGraph StateGraph
    └─ Creates nodes for each agent
    └─ Sets up edges (sequential or conditional)
    ↓
[WORKFLOW EXECUTION] (LangGraph)
    ├─ Extractor Agent → Retrieves documents
    ├─ Classifier Agent → Categorizes content
    ├─ Domain Agents → Compliance/Risk/Gap analysis
    ├─ Citation Auditor → Validates sources
    └─ Report Generator → Creates final output
    ↓
[STATE PERSISTENCE] (PostgreSQL Checkpointer)
    └─ Saves state after each agent
    └─ Enables recovery & conversation continuity
    ↓
Final Answer (Markdown Report)
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Graph Framework** | LangGraph | Workflow orchestration & state management |
| **LLM Provider** | Anthropic Claude | Agent reasoning & decision-making |
| **State Persistence** | PostgreSQL + LangGraph Checkpointer | Conversation continuity |
| **Prompt Caching** | Anthropic API (ephemeral) | 90% cost reduction |
| **Observability** | LangSmith | Tracing, debugging, cost monitoring |
| **Prompts** | File-based (prompts/agents/*.txt) | Agent system prompts |
| **Tools** | Existing src.agent.tools infrastructure | 15 specialized tools |

### Related Documentation

- **Detailed Execution Flows**: [`multi_agent_components.md`](multi_agent_components.md) - Component interactions, state transitions, code-level execution diagrams
- **Quick Reference**: [`multi_agent_quick_reference.md`](multi_agent_quick_reference.md) - Commands, common patterns, troubleshooting

---

## 2. Core Components

### 2.1 Main Entry Point: `runner.py`

**File:** `/src/multi_agent/runner.py` (345 lines)

The `MultiAgentRunner` class orchestrates the entire system:

```python
class MultiAgentRunner:
    async def initialize() -> bool:
        # 1. Setup LangSmith (observability)
        # 2. Create PostgreSQL checkpointer
        # 3. Initialize state manager
        # 4. Initialize caching layer
        # 5. Register all 8 agents
        # 6. Create complexity analyzer
        # 7. Build workflow builder

    async def run_query(query: str) -> Dict:
        # 1. Orchestrator analyzes complexity
        # 2. Workflow builder constructs graph
        # 3. Execute workflow via LangGraph
        # 4. Extract final answer
        # 5. Return results with metadata
```

**CLI Usage:**
```bash
# Single query mode
uv run python -m src.multi_agent.runner --query "your query"

# Interactive mode
uv run python -m src.multi_agent.runner --interactive

# Debug mode
uv run python -m src.multi_agent.runner --query "query" --debug
```

**Initialization Steps:**
1. Load configuration from `config.json`
2. Setup observability (LangSmith)
3. Create checkpointing system (PostgreSQL)
4. Initialize caching (prompt + semantic caches)
5. Register all agents via decorator system
6. Create routing components
7. Build workflow templates

### 2.2 State Management: `core/state.py`

**File:** `/src/multi_agent/core/state.py` (190 lines)

Defines the complete state schema passed through LangGraph:

#### Key Enums:
```python
class QueryType(str, Enum):
    SIMPLE_SEARCH = "simple_search"
    CROSS_DOC_ANALYSIS = "cross_doc"
    COMPLIANCE_CHECK = "compliance"
    RISK_ASSESSMENT = "risk"
    SYNTHESIS = "synthesis"
    REPORTING = "reporting"
    UNKNOWN = "unknown"

class ExecutionPhase(str, Enum):
    ROUTING = "routing"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    REPORTING = "reporting"
    COMPLETE = "complete"
    ERROR = "error"
```

#### MultiAgentState (Main State Object):
```python
class MultiAgentState(BaseModel):
    # INPUT
    query: str
    
    # ROUTING
    query_type: QueryType
    complexity_score: int (0-100)
    execution_phase: ExecutionPhase
    agent_sequence: List[str]
    
    # EXECUTION
    current_agent: Optional[str]
    agent_outputs: Dict[str, Any]
    tool_executions: List[ToolExecution]
    
    # RETRIEVAL
    documents: List[DocumentMetadata]
    retrieved_text: str
    
    # CONTEXT
    shared_context: Dict[str, Any]
    
    # RESULTS
    final_answer: Optional[str]
    structured_output: Dict[str, Any]
    citations: List[str]
    confidence_score: Optional[float]
    
    # COST TRACKING
    total_cost_cents: float
    cost_breakdown: Dict[str, float]
    
    # ERROR HANDLING
    errors: List[str]
    
    # METADATA
    session_id: str
    user_id: str
    created_at: datetime
    checkpoints: List[str]
```

---

## 3. Agent Architecture

### 3.1 Base Agent Framework: `core/agent_base.py`

**File:** `/src/multi_agent/core/agent_base.py` (279 lines)

All 8 agents inherit from `BaseAgent` abstract class:

#### AgentConfig:
```python
@dataclass
class AgentConfig:
    name: str                           # e.g., 'extractor'
    role: AgentRole                     # orchestrate, extract, classify, verify, audit, synthesize, report
    tier: AgentTier                     # orchestrator, specialist, worker
    model: str                          # Per-agent LLM model
    max_tokens: int = 4096
    temperature: float = 0.3
    tools: Set[str] = field(default_factory=set)  # 3-5 tools per agent
    timeout_seconds: int = 30
    retry_count: int = 2
    enable_prompt_caching: bool = True
    enable_cost_tracking: bool = True
```

#### BaseAgent Template Method Pattern:
```python
class BaseAgent(ABC):
    async def execute(state: Dict) -> Dict:
        # PUBLIC INTERFACE (Template Method)
        # - Timing measurement
        # - Error handling
        # - Cost tracking
        # - Calls execute_impl()
        
    @abstractmethod
    async def execute_impl(state: Dict) -> Dict:
        # IMPLEMENT IN SUBCLASS
        # - Core agent logic
        # - Tool execution
        # - State updates
```

#### Key Methods:
- `execute()` - Public interface with error handling & tracking
- `execute_impl()` - Implementation in subclass
- `handle_error()` - Graceful error handling with fallbacks
- `get_stats()` - Return execution statistics
- `validate_tools()` - Verify required tools exist

### 3.2 Agent Registry: `core/agent_registry.py`

**File:** `/src/multi_agent/core/agent_registry.py` (235 lines)

Central registry for agent discovery and lifecycle:

```python
class AgentRegistry:
    def register_agent_class(agent_name: str, agent_class: Type[BaseAgent])
    def register_config(agent_name: str, config: AgentConfig)
    def get_agent(agent_name: str) -> Optional[BaseAgent]  # Lazy instantiation
    def get_all_agents() -> List[BaseAgent]
    def get_agents_by_role(role: AgentRole) -> List[BaseAgent]
    def get_agents_by_tier(tier: AgentTier) -> List[BaseAgent]
    def validate_all_agents(available_tools: Set[str]) -> bool
    def get_stats() -> Dict
```

**Registration Pattern (Decorator):**
```python
@register_agent("extractor")
class ExtractorAgent(BaseAgent):
    ...
```

Agents automatically register on import via `@register_agent()` decorator.

---

## 4. The 8 Specialized Agents

### Agent Characteristics Table

| # | Agent | Role | Tier | Responsibility | Key Tools | Model |
|---|-------|------|------|-----------------|-----------|-------|
| 1 | **Orchestrator** | ORCHESTRATE | ORCHESTRATOR | Route queries, analyze complexity | LLM reasoning | claude-sonnet-4-5 |
| 2 | **Extractor** | EXTRACT | SPECIALIST | Retrieve docs, hybrid search | search_documents, get_context | claude-sonnet-4-5 |
| 3 | **Classifier** | CLASSIFY | SPECIALIST | Categorize content, detect domains | classify_document, detect_language | claude-3.5-sonnet |
| 4 | **Compliance** | VERIFY | SPECIALIST | GDPR/CCPA/HIPAA/SOX checking | search_regulations, verify_compliance | claude-sonnet-4-5 |
| 5 | **Risk Verifier** | VERIFY | SPECIALIST | Risk assessment & verification | assess_risk, identify_hazards | claude-3.5-sonnet |
| 6 | **Citation Auditor** | AUDIT | SPECIALIST | Validate citations & sources | validate_citations, check_provenance | claude-3.5-sonnet |
| 7 | **Gap Synthesizer** | SYNTHESIZE | SPECIALIST | Find knowledge gaps, coverage analysis | analyze_gaps, find_missing | claude-3.5-sonnet |
| 8 | **Report Generator** | REPORT | SPECIALIST | Compile final report | format_report, consolidate_findings | claude-sonnet-4-5 |

### 4.1 Orchestrator Agent

**File:** `/src/multi_agent/agents/orchestrator.py` (306 lines)

**Responsibilities:**
1. Analyze query complexity (0-100)
2. Classify query type (compliance, risk, synthesis, search, reporting)
3. Determine agent sequence
4. Select workflow pattern (simple, standard, complex)

**Key Methods:**
```python
async def execute_impl(state: Dict) -> Dict:
    # 1. Call LLM with complexity analysis prompt
    # 2. Parse JSON response with routing decision
    # 3. Validate routing decision
    # 4. Return updated state with:
    #    - complexity_score (0-100)
    #    - query_type (QueryType enum)
    #    - agent_sequence (List[str])

async def _analyze_and_route(query: str) -> Dict:
    # Uses Anthropic API with prompt caching
    # Returns routing decision with reasoning

def _parse_routing_response(response_text: str) -> Dict:
    # Extract JSON from markdown or plain text

def _validate_routing_decision(decision: Dict) -> None:
    # Validate required fields and ranges
```

**Routing Decision Example:**
```json
{
    "complexity_score": 65,
    "query_type": "compliance",
    "agent_sequence": ["extractor", "classifier", "compliance", "report_generator"],
    "reasoning": "Medium complexity - compliance/regulatory focus detected"
}
```

### 4.2 Extractor Agent

**File:** `/src/multi_agent/agents/extractor.py` (11.5 KB)

**Responsibilities:**
1. Hybrid search (BM25 + Dense + RRF fusion)
2. Context expansion around chunks
3. Document metadata retrieval
4. Citation preservation

**Workflow:**
```python
async def execute_impl(state: Dict) -> Dict:
    # 1. Determine k (retrieval count) from complexity
    # 2. Hybrid search (BM25 + Dense + RRF)
    # 3. Expand context around top chunks
    # 4. Get document summaries
    # 5. Extract citations
    # 6. Format output for next agents
```

**Key Parameters:**
- `default_k = 6` - Default chunks to retrieve
- `max_k = 15` - Maximum for complex queries
- Uses multi-layer FAISS indexes (L1, L2, L3)

### 4.3 Classifier Agent

**File:** `/src/multi_agent/agents/classifier.py` (100+ lines)

**Responsibilities:**
1. Document type classification (Contract, Policy, Report, etc.)
2. Domain identification (Legal, Technical, Financial)
3. Complexity assessment
4. Language detection & sensitivity classification

### 4.4 Compliance Agent

**File:** `/src/multi_agent/agents/compliance.py` (100+ lines)

**Responsibilities:**
1. GDPR, CCPA, HIPAA, SOX compliance verification
2. Bidirectional checking (Contract → Law, Law → Contract)
3. Violation identification
4. Gap analysis

**Key Method:**
```python
async def execute_impl(state: Dict) -> Dict:
    # 1. Identify relevant framework from query
    # 2. Search regulatory knowledge graph
    # 3. Verify compliance using LLM
    # 4. Assess confidence in findings
    # 5. Return violations, gaps, recommendations
```

### 4.5 Risk Verifier Agent

**File:** `/src/multi_agent/agents/risk_verifier.py` (7.3 KB)

**Responsibilities:**
1. Risk assessment & verification
2. Identify hazards & liabilities
3. Impact analysis
4. Risk mitigation recommendations

### 4.6 Citation Auditor Agent

**File:** `/src/multi_agent/agents/citation_auditor.py` (9.0 KB)

**Responsibilities:**
1. Validate citations & sources
2. Check source provenance
3. Verify citation accuracy
4. Flag missing or invalid citations

### 4.7 Gap Synthesizer Agent

**File:** `/src/multi_agent/agents/gap_synthesizer.py` (8.7 KB)

**Responsibilities:**
1. Analyze knowledge gaps
2. Find missing coverage
3. Comprehensive coverage assessment
4. Synthesis of findings

### 4.8 Report Generator Agent

**File:** `/src/multi_agent/agents/report_generator.py` (9.6 KB)

**Responsibilities:**
1. Executive summary creation
2. Detailed findings compilation
3. Compliance matrix generation
4. Risk assessment summary
5. Citations consolidation
6. Recommendations prioritization
7. Appendix with metadata

**Output Format:** Comprehensive Markdown report with:
- Executive summary
- Detailed findings
- Compliance matrix
- Risk assessment
- Citations
- Recommendations
- Execution metadata

---

## 5. Routing & Workflow Management

### 5.1 Complexity Analyzer: `routing/complexity_analyzer.py`

**File:** `/src/multi_agent/routing/complexity_analyzer.py` (351 lines)

Analyzes query characteristics to determine optimal routing:

#### Complexity Scoring (0-100):
```python
def _calculate_complexity_score(query: str) -> int:
    # Base score from query length (0-30 points)
    # Score from query structure (0-20 points)
    #   - Multiple clauses/sentences
    # Keyword bonuses:
    #   - Compliance keywords: +25
    #   - Risk keywords: +20
    #   - Synthesis keywords: +15
    #   - Simple keywords: -10
    # Multiple questions: +10
    # Final: normalize to 0-100
```

#### Workflow Patterns:
```python
class WorkflowPattern(Enum):
    SIMPLE = "simple"        # < 30: Extractor → Report Generator
    STANDARD = "standard"    # 30-70: Extractor → Classifier → Domain Agent → Report
    COMPLEX = "complex"      # > 70: Full agent pipeline
```

#### Agent Sequence Building:
```python
def _build_agent_sequence(complexity_score, query_type, pattern) -> List[str]:
    # SIMPLE (< 30):
    #   ["extractor", "report_generator"]
    # STANDARD (30-70):
    #   ["extractor", "classifier", domain_agent, "report_generator"]
    # COMPLEX (> 70):
    #   ["extractor", "classifier", "compliance", "risk_verifier",
    #    "citation_auditor", "gap_synthesizer", "report_generator"]
```

### 5.2 Workflow Builder: `routing/workflow_builder.py`

**File:** `/src/multi_agent/routing/workflow_builder.py` (277 lines)

Constructs LangGraph workflows dynamically:

```python
class WorkflowBuilder:
    def build_workflow(agent_sequence: List[str], enable_parallel: bool = False) -> StateGraph:
        # 1. Create StateGraph(MultiAgentState)
        # 2. Add agent node for each agent in sequence
        # 3. Connect agents sequentially (edges)
        # 4. Set entry point (first agent)
        # 5. Set exit point (last agent → END)
        # 6. Compile with optional PostgreSQL checkpointer
        # Return: compiled LangGraph for execution

    def _add_agent_node(workflow, agent_name):
        # Get agent from registry
        # Create async node function that:
        #   - Updates execution phase
        #   - Calls agent.execute()
        #   - Handles errors
        # Add node to workflow

    def _add_workflow_edges(workflow, agent_sequence, enable_parallel):
        # Connect agents in sequence:
        #   agent[0] → agent[1] → ... → agent[n] → END
        # (Parallel execution not yet implemented)
```

#### Conditional Routing (Advanced):
```python
def build_conditional_workflow(complexity_score) -> StateGraph:
    # After extractor:
    #   if complexity < 30 → report_generator
    #   else → classifier
    # After classifier:
    #   if query_type == compliance → compliance
    #   if query_type == risk → risk_verifier
    #   else → gap_synthesizer
    # Then: citation_auditor → gap_synthesizer → report_generator → END
```

---

## 6. Tool Integration

### 6.1 Tool Adapter: `tools/adapter.py`

**File:** `/src/multi_agent/tools/adapter.py` (200+ lines)

Bridges LangGraph agents with existing 15-tool infrastructure:

```python
class ToolAdapter:
    # Provides zero-change integration with existing tools
    
    async def execute(
        tool_name: str,
        inputs: Dict,
        agent_name: str
    ) -> Dict:
        # 1. Lookup tool in existing registry (src.agent.tools.registry)
        # 2. Execute tool with existing infrastructure
        # 3. Track execution time & tokens
        # 4. Record ToolExecution in state
        # 5. Return result dict with:
        #    - success: bool
        #    - data: Any
        #    - citations: List[str]
        #    - metadata: Dict
        #    - error: Optional[str]
```

**Tool Availability:** All 15 RAG tools available to agents (filtered_search and similarity_search removed, unified into search):

**Core Retrieval:**
1. `search` - Unified hybrid search with expansion, HyDE, graph boost
2. `graph_search` - Entity-centric search (requires KG)
3. `filtered_search` - Advanced search with filters
4. `similarity_search` - Semantic similarity
5. `expand_context` - Context expansion
6. `cluster_search` - Cluster-based retrieval

**Analysis:**
7. `multi_doc_synthesizer` - Multi-document synthesis
8. `contextual_chunk_enricher` - Contextual enrichment
9. `explain_search_results` - Explain retrieval
10. `assess_retrieval_confidence` - Confidence assessment
11. `browse_entities` - Browse KG entities
12. `get_stats` - Corpus statistics
13. `definition_aligner` - Legal definition alignment

**Metadata:**
14. `get_tool_help` - Tool documentation
15. `list_available_tools` - List all tools
16. `get_document_list` - List documents
17. `get_document_info` - Document metadata

---

## 7. State Persistence & Checkpointing

### 7.1 PostgreSQL Checkpointer: `checkpointing/postgres_checkpointer.py`

**File:** `/src/multi_agent/checkpointing/postgres_checkpointer.py` (10.6 KB)

Enables conversation continuity and recovery:

```python
class PostgresCheckpointer:
    # Features:
    # - Persist state after each agent
    # - Checkpoint versioning
    # - Recovery from checkpoints
    # - Automatic cleanup
    # - Thread-safe operations
    
    async def put(
        config: RunnableConfig,
        values: Dict,
        metadata: Optional[Dict] = None
    ) -> RunnableConfig:
        # Save state checkpoint to PostgreSQL
        # Return config with checkpoint ID
    
    async def get(checkpoint_id: str) -> Optional[Dict]:
        # Retrieve state from checkpoint
    
    async def list(
        config: Optional[RunnableConfig] = None,
        filter: Optional[Dict] = None,
        before: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Iterator[Dict]:
        # List checkpoints with filtering
```

### 7.2 State Manager: `checkpointing/state_manager.py`

**File:** `/src/multi_agent/checkpointing/state_manager.py` (4.8 KB)

High-level state operations:

```python
class StateManager:
    def create_thread_id() -> str:
        # Generate UUID-based thread ID
    
    def should_snapshot() -> bool:
        # Determine snapshot interval
    
    def save_state_snapshot(
        thread_id: str,
        checkpoint_id: str,
        query: str,
        state: Dict
    ) -> None:
        # Save snapshot if checkpointer available
    
    def recover_from_checkpoint(
        checkpoint_id: str
    ) -> Optional[Dict]:
        # Recover state from checkpoint ID
```

---

## 8. Caching System

### 8.1 Cache Manager: `caching/cache_manager.py`

**File:** `/src/multi_agent/caching/cache_manager.py` (7.3 KB)

Three-level caching for cost optimization:

```python
class CacheManager:
    # 1. PROMPT CACHE (Anthropic API)
    #    - System prompts cached at API level
    #    - 90% cost reduction on repeated queries
    #    - Ephemeral cache (within 5-minute window)
    
    # 2. SEMANTIC CACHE (Regulatory/Contract specific)
    #    - Cached regulatory findings
    #    - Cached contract analysis results
    #    - Fast lookup via semantic similarity
    
    # 3. SYSTEM CACHE (Common patterns)
    #    - Cached compliance matrices
    #    - Cached risk templates
    #    - Cached report sections
```

**Cache Types:**
- `RegulatoryCache` - Compliance findings cache
- `ContractCache` - Contract analysis cache
- `SystemCache` - Common patterns cache

---

## 9. Observability & LangSmith Integration

### 9.1 LangSmith Integration: `observability/langsmith_integration.py`

**File:** `/src/multi_agent/observability/langsmith_integration.py` (80+ lines)

Full tracing for debugging and monitoring:

```python
class LangSmithIntegration:
    def setup() -> bool:
        # Enable LangSmith tracing
        # Set environment variables
        # Configure sampling rate
    
    @contextmanager
    def trace_workflow(workflow_name: str):
        # Trace context for workflow execution
        # Automatic logging of all agent calls
        # Cost tracking
```

**Benefits:**
- Full workflow visualization
- Agent execution timing
- Tool call tracing
- Cost monitoring
- Error debugging
- Performance analysis

---

## 10. Prompt Management

### 10.1 Prompt Loader: `prompts/loader.py`

**File:** `/src/multi_agent/prompts/loader.py` (246 lines)

Loads agent system prompts from file system:

```python
class PromptLoader:
    def __init__(prompts_dir: Optional[Path] = None):
        # Default: prompts/agents/*.txt
        # Load all prompts on startup
        # Cache in memory for performance
    
    def get_prompt(agent_name: str, reload: bool = False) -> str:
        # Get prompt from cache
        # Reload from disk if requested (development)
        # Fallback to generic prompt if not found
    
    def format_prompt(
        agent_name: str,
        context: Optional[Dict[str, str]] = None
    ) -> str:
        # Format prompt with context variables
        # Support Python string formatting syntax
```

**Prompt Files (Expected):**
- `prompts/agents/orchestrator.txt`
- `prompts/agents/extractor.txt`
- `prompts/agents/classifier.txt`
- `prompts/agents/compliance.txt`
- `prompts/agents/risk_verifier.txt`
- `prompts/agents/citation_auditor.txt`
- `prompts/agents/gap_synthesizer.txt`
- `prompts/agents/report_generator.txt`

---

## 11. Execution Flow: Step-by-Step

### Complete Query Processing Pipeline

```
1. USER QUERY
   └─ Input: "What are GDPR compliance requirements in this contract?"

2. RUNNER INITIALIZATION
   ├─ Load config.json
   ├─ Initialize LangSmith (observability)
   ├─ Create PostgreSQL checkpointer
   ├─ Register all 8 agents
   └─ Output: Initialized MultiAgentRunner

3. CREATE STATE
   └─ MultiAgentState(query="What are GDPR...", execution_phase=ROUTING)

4. ORCHESTRATOR ANALYSIS
   ├─ Input: query + system prompt
   ├─ LLM Call: "Analyze complexity and route"
   ├─ Parse Response: {complexity_score: 75, query_type: "compliance", ...}
   └─ Output: state.agent_sequence = ["extractor", "classifier", "compliance", 
                                        "citation_auditor", "report_generator"]

5. COMPLEXITY VALIDATION
   ├─ Check complexity_score (0-100)
   ├─ Validate agent_sequence non-empty
   └─ Fallback to keyword-based routing if LLM parsing fails

6. WORKFLOW CONSTRUCTION
   ├─ Create LangGraph StateGraph
   ├─ Add node for each agent:
   │  ├─ extractor_node(state) → Extractor.execute()
   │  ├─ classifier_node(state) → Classifier.execute()
   │  ├─ compliance_node(state) → Compliance.execute()
   │  ├─ citation_auditor_node(state) → CitationAuditor.execute()
   │  └─ report_generator_node(state) → ReportGenerator.execute()
   ├─ Connect nodes: extractor → classifier → compliance → citation_auditor → report_generator → END
   └─ Compile with PostgreSQL checkpointer

7. WORKFLOW EXECUTION (LangGraph)
   ├─ Initial state flows through graph
   ├─ Each agent execution:
   │  ├─ [TIMING] Start execution
   │  ├─ [EXECUTION] Execute agent:
   │  │  ├─ Load system prompt
   │  │  ├─ Prepare input context
   │  │  ├─ Call LLM with tools
   │  │  ├─ Execute tools (hybrid search, verify, etc.)
   │  │  ├─ Collect results
   │  │  └─ Update state
   │  ├─ [CHECKPOINTING] Save state checkpoint
   │  ├─ [LOGGING] Log execution metrics
   │  └─ [ERROR HANDLING] Handle failures gracefully
   │
   ├─ AGENT 1: EXTRACTOR
   │  ├─ Tool: hybrid_search(query, k=10)
   │  │  ├─ BM25 search
   │  │  ├─ Dense vector search (Layer 1, 2, 3)
   │  │  ├─ RRF fusion (k=60)
   │  │  └─ Return top 10 chunks
   │  ├─ Tool: expand_context(chunks)
   │  │  └─ Add surrounding context
   │  ├─ Tool: get_document_metadata(docs)
   │  │  └─ Retrieve summaries & metadata
   │  └─ Output: state.documents, state.citations, state.agent_outputs["extractor"]
   │
   ├─ AGENT 2: CLASSIFIER
   │  ├─ Tool: classify_document(extractor_output)
   │  │  ├─ Document type (Contract, Policy, Report, etc.)
   │  │  ├─ Domain (Legal, Technical, Financial)
   │  │  ├─ Complexity assessment
   │  │  └─ Sensitivity classification
   │  └─ Output: state.agent_outputs["classifier"]
   │
   ├─ AGENT 3: COMPLIANCE
   │  ├─ Tool: search_regulatory_graph("GDPR")
   │  │  └─ Find regulatory entities in knowledge graph
   │  ├─ Tool: verify_compliance(extractor_output, regulatory_data)
   │  │  ├─ Check for violations
   │  │  ├─ Identify gaps
   │  │  └─ Generate compliance matrix
   │  ├─ Tool: assess_confidence(findings)
   │  │  └─ Confidence score (0-100%)
   │  └─ Output: state.agent_outputs["compliance"]
   │
   ├─ AGENT 4: CITATION AUDITOR
   │  ├─ Tool: validate_citations(compliance_findings)
   │  │  ├─ Verify sources
   │  │  ├─ Check provenance
   │  │  └─ Flag invalid citations
   │  ├─ Tool: check_provenance(citations)
   │  │  └─ Verify citation accuracy
   │  └─ Output: state.agent_outputs["citation_auditor"]
   │
   └─ AGENT 5: REPORT GENERATOR
      ├─ Tool: format_report(all_agent_outputs)
      │  ├─ Executive summary
      │  ├─ Detailed findings
      │  ├─ Compliance matrix
      │  ├─ Risk assessment
      │  ├─ Citations
      │  ├─ Recommendations
      │  └─ Execution metadata
      └─ Output: state.final_answer (Markdown report)

8. STATE PERSISTENCE
   └─ Save final state to PostgreSQL

9. RESULT FORMATTING
   ├─ Extract final_answer
   ├─ Include metadata:
   │  ├─ complexity_score
   │  ├─ query_type
   │  ├─ agent_sequence
   │  ├─ documents_retrieved
   │  ├─ citations
   │  ├─ total_cost_cents
   │  └─ errors (if any)
   └─ Return to user

10. OUTPUT
    └─ Comprehensive Markdown report with all findings
```

---

## 12. Configuration

### Configuration Structure (config.json)

The multi-agent system is configured in the main `config.json`. While specific multi-agent configuration sections are being developed, the system currently uses:

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
  "agent_tools": {
    "default_k": 6,
    "enable_graph_boost": true
  }
}
```

**Per-Agent Model Configuration (in runner.py):**
```python
def _build_agent_config(agent_name, config):
    return AgentConfig(
        name=agent_name,
        model=config.get("model", "claude-sonnet-4-5-20250929"),
        max_tokens=config.get("max_tokens", 2048),
        temperature=config.get("temperature", 0.3),
        enable_prompt_caching=config.get("enable_prompt_caching", True),
    )
```

---

## 13. Error Handling & Recovery

### Error Handling Strategy

**Hierarchical Error Handling:**
```python
# Level 1: Agent-level error handling
BaseAgent.handle_error(error, state):
    - Add error to state.errors
    - Set execution_phase = ERROR
    - Return state (non-blocking)

# Level 2: Workflow-level error handling
WorkflowBuilder._add_agent_node():
    - Try-catch around agent.execute()
    - Catch errors, update state
    - Continue to next agent if possible

# Level 3: Runner-level error handling
MultiAgentRunner.run_query():
    - Catch exceptions
    - Return error in result dict
    - Log for debugging
```

**Fallback Mechanisms:**
1. Orchestrator fails → Use keyword-based routing (ComplexityAnalyzer)
2. Tool fails → Graceful degradation (agent continues with limited data)
3. LLM call fails → Retry with exponential backoff
4. Workflow fails → Return partial results with error details

---

## 14. Integration with Existing RAG Pipeline

### Zero-Change Integration

The multi-agent system **does not modify** the existing RAG pipeline:

**Preserved Components:**
- **Hierarchical Document Summaries** - Generated in Phase 3B (section → document)
- **Token-Aware Chunking** - 512 tokens (HybridChunker)
- **Generic Summaries** - 150 characters (research-backed)
- **Multi-Layer Embeddings** - 3 FAISS indexes (Layer 1, 2, 3)
- **Hybrid Search** - BM25 + Dense + RRF fusion (k=60)
- **All 15 Tools** - Available via ToolAdapter (filtered_search and similarity_search removed)

**How Integration Works:**
```
Multi-Agent System
    ↓
ToolAdapter (src/multi_agent/tools/adapter.py)
    ↓
Existing Tool Registry (src/agent/tools/registry.py)
    ↓
RAG Infrastructure:
    ├─ FAISS Vector Stores (L1, L2, L3)
    ├─ BM25 Index
    ├─ Knowledge Graph
    ├─ Document Metadata
    └─ Caching Layer
```

**Result:** Agents automatically use:
- Latest vector stores
- Current hybrid search configuration
- Existing document corpus
- All indexed knowledge graphs

---

## 15. Performance Characteristics

### Token & Time Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| **Query Processing** | Time per query | 2-5 seconds |
| **Orchestrator** | Token cost | 200-500 tokens |
| **Extractor** | Tool time | 500-800ms (hybrid search) |
| **Classifier** | Tool time | 300-400ms |
| **Compliance** | Tool time | 800-1200ms (graph search) |
| **Report Generator** | Generation time | 1-2 seconds |
| **Total** | E2E time | 3-8 seconds |
| **Prompt Cache** | Cost reduction | 90% (repeated queries) |

### Cost Optimization

**Three-Level Caching:**
1. **Prompt Caching** (API-level) - System prompts cached for 5 minutes
2. **Semantic Caching** - Regulatory/contract findings cached
3. **System Cache** - Common patterns pre-computed

**Expected Cost Reduction:**
- First query: Baseline cost
- Subsequent queries (same session): 90% reduction via prompt caching
- Bulk processing: Additional 40% reduction via semantic caching

---

## 16. Testing & Debugging

### Running the Multi-Agent System

```bash
# Single query
uv run python -m src.multi_agent.runner \
  --config config.json \
  --query "What GDPR requirements apply?"

# Interactive mode
uv run python -m src.multi_agent.runner \
  --config config.json \
  --interactive

# Debug mode (verbose logging)
uv run python -m src.multi_agent.runner \
  --config config.json \
  --query "query" \
  --debug
```

### Debugging Tools

**LangSmith Tracing:**
- Set `multi_agent.langsmith.enabled = true` in config
- View all agent executions, tool calls, and costs
- Trace errors and performance bottlenecks

**State Checkpointing:**
- Recovery from any agent failure
- Replay workflows from checkpoint
- Inspect intermediate states

**Logging:**
- All components log to `src.multi_agent.*` loggers
- Use `--debug` flag for DEBUG level
- Review logs in `logs/pipeline.log`

---

## 17. Architecture Diagrams

### System Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   MULTI-AGENT RUNNER                         │
│  (src/multi_agent/runner.py)                                │
│  ├─ Initialize all systems                                  │
│  ├─ Load configuration                                      │
│  └─ Orchestrate query execution                             │
└────────────────┬────────────────────────────────────────────┘
                 │
       ┌─────────┼─────────┬──────────────┬───────────┐
       │         │         │              │           │
┌──────▼──┐┌─────▼──┐┌────▼─┐┌──────────┐│  ┌────────▼─┐
│LangSmith││Postgres││Config││Prompt    ││  │Tool      │
│Integration Checkpoint Manager  │Loader     ││  │Adapter   │
└──────────┘└────────┘└───────┘└──────────┘  └──────────┘
       │                                           │
       └──────────────────┬──────────────────────┘
                          │
       ┌──────────────────▼──────────────────┐
       │  AGENT REGISTRY & ROUTING            │
       │  ├─ Agent Discovery                 │
       │  ├─ Complexity Analysis             │
       │  └─ Workflow Building               │
       └──────────────┬───────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼────┐    ┌──────▼──────┐    ┌────▼────┐
│SIMPLE  │    │  STANDARD   │    │COMPLEX  │
│Pattern │    │  Pattern    │    │Pattern  │
│        │    │             │    │         │
│Extract │    │ Extract →   │    │ All 8   │
│ Report │    │ Classify →  │    │ Agents  │
└────────┘    │ Domain Agent│    │         │
              │ Report      │    └────────┘
              └─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   LANGGRAPH WORKFLOW       │
    │   (State Processing)       │
    │                            │
    │  ┌─────────────────────┐  │
    │  │   AGENT NODES       │  │
    │  │ (Per-agent execute) │  │
    │  ├─ Orchestrator       │  │
    │  ├─ Extractor         │  │
    │  ├─ Classifier        │  │
    │  ├─ Compliance        │  │
    │  ├─ Risk Verifier     │  │
    │  ├─ Citation Auditor  │  │
    │  ├─ Gap Synthesizer   │  │
    │  └─ Report Generator  │  │
    │  └─────────────────────┘  │
    │                            │
    │  ┌─────────────────────┐  │
    │  │   EDGES             │  │
    │  │ (Sequential routing)│  │
    │  └─────────────────────┘  │
    └────────────┬───────────────┘
                 │
    ┌────────────▼──────────────┐
    │  RAG INFRASTRUCTURE        │
    │  ├─ Vector Stores (L1/L2/L3)
    │  ├─ BM25 Index             │
    │  ├─ Knowledge Graph        │
    │  ├─ Tools (17 total)       │
    │  └─ Caching               │
    └────────────────────────────┘
```

### Agent Execution Flow Diagram

```
State: MultiAgentState
       (query, phase, complexity, ...)
         │
         ▼
    ┌─────────────────────┐
    │   AGENT EXECUTION   │
    ├─────────────────────┤
    │ 1. Load system prompt
    │ 2. Prepare input context
    │ 3. Call LLM
    │    ├─ With prompt caching
    │    ├─ With tools available
    │    └─ With cost tracking
    │ 4. Process tool calls
    │    ├─ Via ToolAdapter
    │    ├─ Execute in parallel
    │    └─ Aggregate results
    │ 5. Update state
    │    ├─ agent_outputs[agent_name]
    │    ├─ tool_executions[]
    │    ├─ errors[] (if any)
    │    └─ total_cost_cents
    │ 6. Save checkpoint (optional)
    │ 7. Return updated state
    └─────────────────────┘
         │
         ▼
    Updated State → Next Agent
```

---

## 18. Key Design Decisions

### Why This Architecture?

| Decision | Rationale |
|----------|-----------|
| **8 Agents** | Specialization improves accuracy; reduces hallucination |
| **Complexity-Based Routing** | Avoids over-processing simple queries; cost optimization |
| **Per-Agent Config** | Different tasks need different models/temperatures |
| **PostgreSQL Checkpointing** | Conversation continuity; recovery from failures |
| **Prompt Caching** | 90% cost reduction for repeated queries |
| **ToolAdapter Pattern** | Zero changes to existing 15-tool infrastructure |
| **LangGraph Foundation** | Industry-standard; proven for multi-agent orchestration |
| **Sequential by Default** | Simpler mental model; easier debugging; explicit state flow |

### Research Backing

- **L-MARS** (Legal Multi-Agent RAG System) - Agent specialization
- **PAKTON** (Performance-Aware Knowledge Transfer) - Tool distribution
- **MASLegalBench** - Multi-agent evaluation benchmarks
- **Harvey AI** - Production legal AI patterns
- **Definely** - Legal document analysis architecture

---

## 19. Future Enhancements

### Planned Features

1. **Parallel Execution** - Execute independent agents in parallel
2. **Dynamic Agent Selection** - Select subset of agents per query
3. **Agent Communication** - Direct agent-to-agent messaging
4. **Learning from Feedback** - Improve routing over time
5. **Cost Optimization** - Automatic model downgrade for simple tasks
6. **Custom Agent Framework** - Plugin architecture for new agents

### Research Integration

- Integration with latest multi-agent papers
- Continuous improvement of routing heuristics
- Benchmark against MASLegalBench
- Cost optimization via model selection

---

## 20. Troubleshooting

### Common Issues

#### Issue: "Agent not found in registry"
**Solution:** Ensure agent class has `@register_agent("name")` decorator and is imported before use.

#### Issue: "Tool execution failed"
**Solution:** Check if tool is available in `src.agent.tools.registry`. Use `--debug` for detailed error.

#### Issue: "Prompt file not found"
**Solution:** Create prompt files in `prompts/agents/<agent_name>.txt` or use fallback prompt.

#### Issue: "PostgreSQL connection failed"
**Solution:** Set `checkpointing.enabled = false` in config to disable checkpointing, or ensure PostgreSQL is running.

#### Issue: "LangSmith tracing not working"
**Solution:** Set `langsmith.enabled = true` and provide `langsmith.api_key` in config.

---

## Summary

The SUJBOT2 multi-agent system is a sophisticated, research-backed orchestration framework that:

1. **Routes queries intelligently** based on complexity analysis
2. **Orchestrates 8 specialized agents** with distinct roles
3. **Manages state effectively** using LangGraph + PostgreSQL
4. **Integrates seamlessly** with existing RAG infrastructure
5. **Optimizes costs** via prompt caching and semantic caching
6. **Provides full observability** via LangSmith
7. **Handles errors gracefully** with fallbacks and recovery

The architecture is production-ready and designed for scalability, reliability, and cost-effectiveness in legal/technical document analysis.

---

**Key Files to Review:**
- `/src/multi_agent/runner.py` - Main entry point
- `/src/multi_agent/core/state.py` - State schema
- `/src/multi_agent/core/agent_base.py` - Agent framework
- `/src/multi_agent/core/agent_registry.py` - Agent registration
- `/src/multi_agent/agents/orchestrator.py` - Routing logic
- `/src/multi_agent/routing/workflow_builder.py` - Workflow construction
- `/src/multi_agent/tools/adapter.py` - Tool integration

**Related Documentation:**
- `README.md` - User guide
- `PIPELINE.md` - Full pipeline specification
- `CLAUDE.md` - Project instructions
- `config.json.example` - Configuration reference
