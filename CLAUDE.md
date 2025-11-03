# CLAUDE.md - Navigation Guide

**SUJBOT2**: Production RAG system for legal/technical docs. 7-phase pipeline + 17-tool AI agent.

**Status:** PHASE 1-7 COMPLETE ✅ (2025-11-03)

---

## 🚀 Quick Start

**Read these files for detailed information:**
- [`README.md`](README.md) - User guide, installation, quick start
- [`PIPELINE.md`](PIPELINE.md) - Complete pipeline specification with research papers
- [`INSTALL.md`](INSTALL.md) - Platform-specific setup (Windows/macOS/Linux)
- [`docs/agent/README.md`](docs/agent/README.md) - Agent CLI documentation
- Visual docs: [`indexing_pipeline.html`](indexing_pipeline.html), [`user_search_pipeline.html`](user_search_pipeline.html)

**Common commands:**
```bash
# Index documents
uv run python run_pipeline.py data/document.pdf

# Launch agent
uv run python -m src.agent.cli

# Run tests
uv run pytest tests/ -v

# Debug issues
/debug-optimize
```

---

## ⚠️ CRITICAL CONSTRAINTS (NEVER CHANGE)

These are research-backed decisions. **DO NOT modify** without explicit approval:

### 1. Hierarchical Document Summary (MANDATORY)
```
Flow: Sections → Section Summaries (PHASE 3B) → Document Summary
```
- **NEVER pass full document text to LLM** for document summary
- **ALWAYS generate from section summaries** (hierarchical aggregation)
- Exception: Fallback `"(Document summary unavailable)"` if section summaries fail
- Files: `src/docling_extractor_v2.py`, `src/summary_generator.py`

### 2. RCTS Chunking (LegalBench-RAG)
- Chunk size: **500 chars** (optimal for legal docs)
- Overlap: **0** (RCTS handles via hierarchy)
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

---

## 📂 Key File Locations

**Pipeline Core:**
- `run_pipeline.py` - CLI entry point
- `src/indexing_pipeline.py` - Main orchestrator (PHASE 1-6)
- `src/config.py` - Shared configs

**7 Phases:**
1. `src/docling_extractor_v2.py` - Hierarchy extraction
2. `src/summary_generator.py` - Document summaries
3. `src/multi_layer_chunker.py` - Chunking + SAC + section summaries
4. `src/embedding_generator.py`, `src/faiss_vector_store.py` - Embeddings + FAISS
5. `src/hybrid_search.py`, `src/graph/`, `src/reranker.py` - Advanced retrieval
6. `src/context_assembly.py` - Context prep
7. `src/agent/` - RAG agent (17 tools)

**Agent Tools:**
- `src/agent/tools/tier1_basic.py` - 6 fast tools (100-300ms)
- `src/agent/tools/tier2_advanced.py` - 8 quality tools (500-1000ms)
- `src/agent/tools/tier3_analysis.py` - 3 analysis tools (1-3s)

**Tests:**
- `tests/test_phase*.py` - Pipeline tests
- `tests/agent/` - Agent tests (49 tests)
- `tests/graph/` - Knowledge graph tests

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

**SSOT (Single Source of Truth): `.env.example`**

All configuration lives in [`.env.example`](.env.example) - DO NOT duplicate config in CLAUDE.md!

**Setup:**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

**Key decisions** (see `.env.example` for all options):
- **Required:** `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- **Embedding:** `EMBEDDING_MODEL=bge-m3` (macOS M1/M2/M3, free) or `text-embedding-3-large` (Windows, cloud)
- **Knowledge Graph:** `KG_BACKEND=neo4j` (production) or `simple` (dev/testing)
- **Speed:** `SPEED_MODE=fast` (default) or `eco` (50% cheaper, overnight jobs)

**For detailed config docs, read `.env.example` inline comments.**

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

**Last Updated:** 2025-11-03
**Version:** PHASE 1-7 COMPLETE + Hierarchical Summaries + Query Expansion + RAG Confidence Scoring

**Note:** `vector_db/` is tracked in git (contains merged vector stores) - DO NOT add to `.gitignore`
