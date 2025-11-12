# CLAUDE.md - Navigation Guide

**SUJBOT2**: Production RAG system for legal/technical docs. 7-phase pipeline + multi-agent orchestration + Human-in-the-Loop clarifications.

**Status:** MULTI-AGENT SYSTEM + HITL COMPLETE ✅ (2025-11-11)

---

## 🚀 Quick Start

**Read these files for detailed information:**
- [`README.md`](README.md) - User guide, installation, quick start
- [`PIPELINE.md`](PIPELINE.md) - Complete pipeline specification with research papers
- [`INSTALL.md`](INSTALL.md) - Platform-specific setup (Windows/macOS/Linux)
- [`docs/agent/README.md`](docs/agent/README.md) - Agent CLI documentation
- [`docs/HITL_IMPLEMENTATION_SUMMARY.md`](docs/HITL_IMPLEMENTATION_SUMMARY.md) - Human-in-the-Loop clarification system
- Visual docs: [`indexing_pipeline.html`](indexing_pipeline.html), [`user_search_pipeline.html`](user_search_pipeline.html)

**Common commands:**
```bash
# Index documents
uv run python run_pipeline.py data/document.pdf

# Launch multi-agent system
uv run python -m src.multi_agent.runner --query "your query"
uv run python -m src.multi_agent.runner --interactive

# Launch web interface
./start_web.sh

# Run tests
uv run pytest tests/ -v
```

---

## ⚠️ CRITICAL CONSTRAINTS (NEVER CHANGE)

These are research-backed decisions. **DO NOT modify** without explicit approval:

### 0. AUTONOMOUS AGENTIC ARCHITECTURE (MANDATORY) 🤖

**CRITICAL: Agents MUST be autonomous LLM-driven, NOT hardcoded workflows!**

```
❌ WRONG (Hardcoded):
class ComplianceAgent:
    def execute():
        step1 = call_tool_a()  # Hardcoded sequence
        step2 = call_tool_b()
        return synthesize(step1, step2)

✅ CORRECT (Autonomous):
class ComplianceAgent:
    def execute():
        # LLM autonomously decides which tools to call and when
        return llm.run(
            system_prompt="You are compliance expert...",
            tools=[search, graph_search, assess_confidence, ...],
            messages=[user_query]
        )
```

**Principles:**
1. **LLM decides tool calling** - Agent provides system prompt + tools, LLM autonomously calls them
2. **No hardcoded flows** - No predefined "step 1, step 2, step 3" logic
3. **Tools are capabilities** - Agent defines WHAT tools are available, LLM decides HOW to use them
4. **Orchestrator exception** - ONLY Orchestrator has hardcoded logic (routing), all other agents are autonomous
5. **System prompts guide behavior** - Control agent behavior via prompts, NOT code

**Technical Implementation:**

All agents inherit from `BaseAgent` which provides the autonomous tool calling loop:

```python
# src/multi_agent/core/agent_base.py
async def _run_autonomous_tool_loop(
    self,
    system_prompt: str,
    state: Dict[str, Any],
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Core autonomous agentic behavior:
    1. LLM sees state/query + available tools
    2. LLM decides to call tools or provide final answer
    3. Tool results fed back to LLM
    4. Loop continues until LLM provides final answer
    """
```

**How it works:**
1. Agent calls `_run_autonomous_tool_loop()` with system prompt and state
2. Loop builds context from state (query + previous agent outputs)
3. LLM receives: system prompt + context + tool schemas
4. LLM decides: call tools (→ execute tools → feed results back) OR provide final answer
5. Loop continues until LLM returns final answer or max_iterations reached
6. Result contains: final_answer, tool_calls history, iterations count

**Tool Schema Conversion:**
- `ToolAdapter.get_tool_schema()` converts Pydantic schemas to LLM format (Anthropic/OpenAI)
- Agents automatically get schemas for their configured tools from `config.json`
- LLM sees tool name, description, and input schema for each available tool

**Prompt Design Pattern:**
Every agent prompt (in `prompts/agents/`) follows this structure:
1. **ROLE** - Agent's responsibility
2. **DOMAIN KNOWLEDGE** - Risk categories, compliance frameworks, etc.
3. **AVAILABLE TOOLS (use autonomously as needed)** - Explicit tool listing
4. **AUTONOMOUS WORKFLOW** - "Typical approach" guidance (NOT prescription)
5. **IMPORTANT** - Reinforces autonomy: "YOU decide which tools to use"
6. **FINAL ANSWER FORMAT** - Expected output structure

**Benefits vs Hardcoded:**
- ✅ ~70% code reduction per agent (~200 lines → ~60 lines)
- ✅ LLM adapts to query complexity (simple → calls 1-2 tools, complex → calls 5+ tools)
- ✅ Emergent reasoning (LLM discovers tool combinations we didn't explicitly program)
- ✅ Behavior changes via prompts (no code changes needed)
- ✅ Single implementation in BaseAgent (changes propagate automatically)
- ❌ Requires proper tool schemas and clear prompts
- ❌ LLM cost per agent execution (tool calling loop)

**Files:**
- `src/multi_agent/core/agent_base.py` - Autonomous agent base class with `_run_autonomous_tool_loop()`
- `src/multi_agent/tools/adapter.py` - Tool schema conversion (`get_tool_schema()`)
- `src/multi_agent/agents/*.py` - All 7 agents use autonomous pattern (extractor, classifier, compliance, risk_verifier, citation_auditor, gap_synthesizer, report_generator)
- `prompts/agents/*.txt` - System prompts guide autonomous behavior

**Testing:**
```bash
# Test single autonomous agent
uv run python test_autonomous.py

# Test full multi-agent workflow
uv run python -m src.multi_agent.runner --query "Your query here"
```

**Why:** True agentic system enables emergent behavior, better reasoning, and flexibility without code changes. LLM can discover optimal tool calling strategies we didn't explicitly program.

### 1. Hierarchical Document Summary (MANDATORY)
```
Flow: Sections → Section Summaries (PHASE 3B) → Document Summary
```
- **NEVER pass full document text to LLM** for document summary
- **ALWAYS generate from section summaries** (hierarchical aggregation)
- Exception: Fallback `"(Document summary unavailable)"` if section summaries fail
- Files: `src/docling_extractor_v2.py`, `src/summary_generator.py`

### 2. Token-Aware Chunking (HybridChunker)
- **Max tokens:** 512 (optimal for legal docs)
- **Tokenizer:** tiktoken (OpenAI text-embedding-3-large)
- **Overlap:** 0 (hierarchical chunking handles naturally)
- **Research basis:** 512 tokens ≈ 500 chars (LegalBench-RAG constraint preserved)
- **Why tokens not chars:** Guarantees embedding model compatibility, handles Czech diacritics correctly
- Changing this invalidates ALL vector stores

### 3. Generic Summaries (Reuter et al., counterintuitive!)
- Length: **150 chars**
- Style: **GENERIC** (NOT expert terminology)

### 4. Summary-Augmented Chunking (SAC)
- Prepend document summary during embedding
- Strip summaries during retrieval
- **-58% context drift** (proven by research)

### 5. Multi-Layer Embeddings (Lima 2024)
- **3 separate FAISS indexes** (NOT merged)
- 2.3x essential chunks vs single-layer

### 6. No Cohere Reranking
- Cohere performs WORSE on legal docs
- Use: `ms-marco`, `bge-reranker` instead

### 7. Hybrid Search (Industry 2025)
- BM25 + Dense + RRF fusion
- **+23% precision** vs dense-only
- RRF k=60 (optimal)

### 8. **AUTONOMOUS AGENT RESPONSES (CRITICAL - NO HARDCODED TEMPLATES)**
- **NEVER use rule-based conditional responses** (if greeting → template)
- **ALL communication generated by LLM agents** - no hardcoded strings
- **Orchestrator returns `final_answer` directly** when no specialized agents needed (greetings, chitchat, meta queries)
- **Why this matters:**
  - Enables contextual awareness and natural conversation flow
  - Eliminates brittle template logic that fails on edge cases
  - Allows agent to adapt responses based on conversation history
  - Modern LLM-based architecture principle
- **Implementation:**
  - Orchestrator: Returns `{"agent_sequence": [], "final_answer": "<LLM-generated response>"}` for greetings
  - Runner: Checks for `final_answer` in orchestrator output, returns directly without building workflow
  - NO if/else conditional logic for response generation anywhere in codebase
- **Files:** `src/multi_agent/runner.py`, `src/multi_agent/agents/orchestrator.py`, `prompts/agents/orchestrator.txt`

---

## 🔢 Token vs Character Equivalence

**Research Constraint (LegalBench-RAG):**
- Optimal chunk size: **500 characters** for legal/technical documents

**Token-Aware Implementation:**
- Uses **max_tokens=512** (token-based limit)
- **Equivalence:** 512 tokens ≈ 500-640 characters for Czech/English mixed text
- **Calculation:**
  - English: ~4 chars/token → 500 chars ≈ 125 tokens
  - Czech (diacritics): ~5 chars/token → 500 chars ≈ 100 tokens
  - Safety margin: 512 tokens accommodates worst-case Czech text

**Why This Preserves Research Intent:**
1. Same semantic granularity (small chunks for legal precision)
2. Better reliability (guarantees embedding model compatibility)
3. Czech optimization (accounts for UTF-8 multi-byte encoding)
4. Hierarchical structure preserved (HybridChunker respects DoclingDocument hierarchy)

---

## 📂 Key File Locations

**Pipeline Core:**
- `run_pipeline.py` - CLI entry point
- `src/indexing_pipeline.py` - Main orchestrator (PHASE 1-6)
- `src/config.py` - Shared configs

**7 Phases:**
1. `src/unstructured_extractor.py` - Multi-format extraction with Unstructured.io (Layout model: **yolox** - best results from testing on legal documents)
2. `src/summary_generator.py` - Document summaries (hierarchical from section summaries)
3. `src/multi_layer_chunker.py` - Token-aware chunking + SAC + section summaries
4. `src/embedding_generator.py`, `src/faiss_vector_store.py` - Embeddings + FAISS
5. `src/hybrid_search.py`, `src/graph/`, `src/reranker.py` - Advanced retrieval
6. `src/context_assembly.py` - Context prep
7. `src/agent/` - RAG agent (16 tools)

**Agent Tools:**
- `src/agent/tools/tier1_basic.py` - 5 fast tools (100-300ms)
- `src/agent/tools/tier2_advanced.py` - 10 quality tools (500-1000ms)
  - **NEW (2025-01):** `multi_doc_synthesizer` - Multi-document synthesis (replaces broken `compare_documents`)
  - **NEW (2025-01):** `contextual_chunk_enricher` - Anthropic Contextual Retrieval (-58% context drift)
- `src/agent/tools/tier3_analysis.py` - 1 analysis tool (1-3s)

**Tests:**
- `tests/test_phase*.py` - Pipeline tests
- `tests/agent/` - Agent tests (49 tests)
- `tests/graph/` - Knowledge graph tests
- `tests/multi_agent/integration/` - HITL integration tests

**Human-in-the-Loop (HITL) System:**
- `src/multi_agent/hitl/` - HITL components (4 files)
  - `config.py` - Configuration with quality thresholds
  - `quality_detector.py` - Multi-metric quality detection
  - `clarification_generator.py` - LLM-based question generation
  - `context_enricher.py` - Query enrichment with user response
- `backend/main.py` - `/chat/clarify` endpoint for clarification submission
- `backend/agent_adapter.py` - `resume_clarification()` method
- `frontend/src/components/chat/ClarificationModal.tsx` - React modal component
- **Enabled by default** - Configure in `config_multi_agent_extension.json` under `clarification`
- **See:** [`docs/HITL_IMPLEMENTATION_SUMMARY.md`](docs/HITL_IMPLEMENTATION_SUMMARY.md) for complete documentation

---

## 🛠️ Common Development Tasks

### Adding New Tool
1. Create class in `src/agent/tools/tier{1,2,3}_{basic,advanced,analysis}.py`
2. Define `ToolInput` schema (Pydantic)
3. Implement `execute_impl()` method
4. Register with `@register_tool` decorator
5. Add tests in `tests/agent/tools/`

Example skeleton:
```python
class MyToolInput(ToolInput):
    query: str = Field(..., description="User query")

@register_tool
class MyTool(BaseTool):
    name = "my_tool"
    description = "What this tool does"
    tier = 1
    input_schema = MyToolInput

    def execute_impl(self, query: str) -> ToolResult:
        results = self.vector_store.search(query, k=10)
        return ToolResult(success=True, data=results)
```

### New Tools (2025-01)

#### **multi_doc_synthesizer** (Tier 2)
Synthesizes information from multiple documents using LLM. Replaces broken `compare_documents` tool.

**Use cases:**
```python
# Compare documents
tool.execute(
    document_ids=["doc1", "doc2", "doc3"],
    synthesis_query="Compare privacy policies",
    synthesis_mode="compare"
)

# Unified summary
tool.execute(
    document_ids=["standard1", "standard2"],
    synthesis_query="Data retention requirements",
    synthesis_mode="summarize"
)
```

**Key features:**
- Synthesis modes: `compare`, `summarize`, `analyze`
- Uses public API only (hierarchical_search with document filters)
- 2-10 documents supported
- Cites all source documents

#### **contextual_chunk_enricher** (Tier 2)
Implements Anthropic Contextual Retrieval technique (-58% context drift).

**Use cases:**
```python
# Auto mode (intelligent selection)
tool.execute(
    chunk_ids=["doc1:sec1:0", "doc1:sec2:1"],
    enrichment_mode="auto"
)

# Maximum context
tool.execute(
    chunk_ids=["doc1:sec1:0"],
    enrichment_mode="both",  # document + section summaries
    include_metadata=True
)
```

**Enrichment modes:**
- `auto`: Selects best mode (section > document)
- `document_summary`: Prepend document context
- `section_summary`: Prepend section context
- `both`: Maximum context (document + section)

**Research basis:** Anthropic (2024) - Contextual Retrieval

### Debugging Retrieval Issues
```bash
# Enable debug mode
uv run python -m src.agent.cli --debug

# Use /debug-optimize slash command
/debug-optimize
[Paste error description]
```

### GPT-5 and O-Series Model Compatibility

**RECOMMENDED MODELS (production-ready):**
- **Production:** `claude-sonnet-4-5` (highest quality, best for complex queries)
- **Development:** `gpt-4o-mini` (best cost/performance, $0.15/$0.60 per 1M tokens)
- **Budget:** `claude-haiku-4-5` (fastest, cheapest Claude model)

**GPT-5 Support (✅ IMPLEMENTED, EXPERIMENTAL):**

GPT-5 and O-series models (`gpt-5`, `gpt-5-mini`, `o1`, `o3`, `o4-mini`) are **now supported** but require special parameter handling:

```python
# GPT-5/o-series require different parameters
if model.startswith(("gpt-5", "o1", "o3", "o4")):
    params = {
        "model": model,
        "max_completion_tokens": 300,  # NOT max_tokens
        "temperature": 1.0,  # ONLY 1.0 supported (default)
        "reasoning_effort": "minimal"  # Controls reasoning depth
    }
else:
    # GPT-4 and earlier
    params = {
        "model": model,
        "max_tokens": 300,
        "temperature": 0.7
    }
```

**`reasoning_effort` parameter:**
- `"minimal"` - Fastest, used for simple tasks (summarization, context generation)
- `"low"` - Light reasoning
- `"medium"` - Default reasoning (if not specified)
- `"high"` - Deep reasoning for complex tasks

**Why GPT-5/o-series may not be recommended:**
- ⚠️ API parameter differences can cause confusion (`max_completion_tokens` vs `max_tokens`)
- ⚠️ Temperature is fixed at 1.0 (no customization)
- ⚠️ May be more expensive than GPT-4o-mini for simple tasks
- ⚠️ `reasoning_effort` behavior can be unpredictable for certain prompts

**Files with GPT-5 support:**
- ✅ `src/summary_generator.py` - Document/section summaries
- ✅ `src/contextual_retrieval.py` - Context generation
- ⚠️ `src/agent/query_expander.py` - Not yet updated (uses gpt-4o-mini)

**Testing recommendation:** If you want to use GPT-5 models, test thoroughly with your specific use case before deploying to production. Fall back to `gpt-4o-mini` if you encounter issues.

---

## 🎯 Best Practices

### Agent Development
- **Tool tier selection:** Always start with TIER 1 tools (fast), escalate to TIER 2/3 only when needed
- **Query expansion:** Use `num_expands=0` (default) for speed, `num_expands=1-2` for recall-critical queries
- **Graph boost:** Enable only for entity-focused queries (organizations, standards, regulations)
- **Prompt caching:** Enable via `ENABLE_PROMPT_CACHING=true` for 90% cost savings on repeated queries
- **Context pruning:** Keep conversation history under 50K tokens to prevent quadratic growth

### Pipeline Indexing
- **Speed modes:** Use `SPEED_MODE=fast` for development, `SPEED_MODE=eco` for overnight bulk processing
- **Batch processing:** Index directories instead of individual files for better throughput
- **Knowledge graph:** Set `KG_BACKEND=neo4j` for production, `simple` for testing
- **Entity deduplication:** Use Layer 1 + Layer 3 (production balanced mode) for legal docs
- **Validation:** Always run `pytest tests/` before committing pipeline changes

### Code Quality
- **Type hints:** Required for all public APIs (use `mypy src/` to verify)
- **Error handling:** Use graceful degradation (e.g., reranker unavailable → fall back to RRF)
- **Logging:** Use appropriate levels (debug/info/warning/error) - avoid print statements
- **Testing:** Write tests BEFORE implementing new features (TDD approach)
- **Documentation:** Update PIPELINE.md if research constraints change
- **Model selection:** ALWAYS use `gpt-4o-mini` (NOT gpt-5-nano) for stability and cost savings

### Performance
- **Embedding cache:** Monitor hit rate with `embedder.get_cache_stats()` (target >80%)
- **FAISS indexes:** Keep layer separation (DO NOT merge L1/L2/L3)
- **Reranker loading:** Lazy load to reduce startup time (~2s savings)
- **Token limits:** Use `max_total_tokens` parameter to prevent context overflow

### Debugging
- **Debug mode:** Use `--debug` flag to see tool execution details
- **Cost tracking:** Call `reset_global_tracker()` at operation start, `get_summary()` at end
- **Vector store stats:** Use `store.get_stats()` to diagnose retrieval issues
- **Multi-agent debug:** Use `/debug-optimize` for complex issues (auto-applies fixes)

---

## 🔧 Configuration

**SSOT (Single Source of Truth): `config.json.example`** (UPDATED 2025-11-10)

All configuration lives in [`config.json.example`](config.json.example) - DO NOT duplicate config in CLAUDE.md!

**Migration from .env to config.json:**
- Strict validation - NO fallbacks, NO defaults
- If any required parameter is missing, application exits with error
- Hierarchical JSON structure for better organization

**Setup:**
```bash
cp config.json.example config.json
# Edit config.json with your API keys and settings
# ALL required fields must be filled in
```

**Key decisions** (see `config.json.example` for all options):
- **Required:** `api_keys.anthropic_api_key` or `api_keys.openai_api_key`
- **Embedding:** `models.embedding_model="bge-m3"` (macOS M1/M2/M3, free) or `"text-embedding-3-large"` (Windows, cloud)
- **Knowledge Graph:** `knowledge_graph.backend="neo4j"` (production) or `"simple"` (dev/testing)
- **Speed:** `summarization.speed_mode="fast"` (default) or `"eco"` (50% cheaper, overnight jobs)

**For detailed config docs, read `config.json.example` inline comments.**

---

## 📚 Code Style

**Formatting:**
```bash
uv run black src/ tests/ --line-length 100
uv run isort src/ tests/ --profile black
```

**Conventions:**
- Classes: `PascalCase`
- Functions/vars: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Docstrings: Google style
- Type hints: Required for public APIs

---

## 📖 Research Papers (DO NOT CONTRADICT)

1. **LegalBench-RAG** (Pipitone & Alami, 2024) - RCTS, reranking
2. **Summary-Augmented Chunking** (Reuter et al., 2024) - SAC, generic summaries
3. **Multi-Layer Embeddings** (Lima, 2024) - 3-layer indexing
4. **Contextual Retrieval** (Anthropic, 2024) - Context prepending
5. **HybridRAG** (2024) - Graph boosting (+8% factual correctness)

---

## 🐛 Debug System

**`/debug-optimize` slash command** - Multi-agent debugging:
- 5 specialized agents (cost-optimizer, rag-debugger, validation-expert, pipeline-expert, agent-expert)
- Auto-applies fixes (max 20 per run)
- Respects research constraints
- Git commits if tests pass

**When to use:**
- Agent errors, tool failures
- High API costs, cache misses
- Pipeline failures, validation errors

---

**Last Updated:** 2025-11-11
**Version:** PHASE 1-7 COMPLETE + Multi-Agent System + HITL Clarifications (16 tools, 8 agents)

**Note:** `vector_db/` is tracked in git (contains merged vector stores) - DO NOT add to `.gitignore`
