# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

**SUJBOT2**: Production RAG system for legal/technical documents with multi-agent orchestration.

## Debugging with LangSmith

**IMPORTANT:** When debugging conversations/traces, ALWAYS use **LangSmith MCP tools** (`mcp__langsmith__*`), NOT Python scripts.

```
# Available MCP tools for debugging:
mcp__langsmith__list_projects      # List projects
mcp__langsmith__fetch_runs         # Fetch runs with filters
mcp__langsmith__list_experiments   # List experiments
```

## Common Commands

```bash
# Development
uv sync                                    # Install dependencies
uv run python run_pipeline.py data/doc.pdf # Index single document
uv run python run_pipeline.py data/        # Index directory

# Testing
uv run pytest tests/ -v                                    # All tests
uv run pytest tests/test_phase4_indexing.py -v             # Single file
uv run pytest tests/agent/test_tool_registry.py::test_name -v  # Single test
uv run pytest tests/ --cov=src --cov-report=html           # With coverage

# Linting & Type Checking
uv run black src/ tests/ --line-length 100                 # Format code
uv run isort src/ tests/ --profile black                   # Sort imports
uv run mypy src/                                           # Type check

# Docker (full stack)
docker compose up -d                       # Start all services
docker compose logs -f backend             # Watch backend logs
docker compose exec backend uv run pytest  # Run tests in container

# Agent CLI
uv run python -m src.agent.cli             # Interactive mode
uv run python -m src.agent.cli --debug     # Debug mode
```

## Architecture Overview

```
User Query
    ↓
Orchestrator (routing + synthesis)
    ↓
Specialized Agents (extractor, classifier, compliance, risk_verifier, etc.)
    ↓
RAG Tools (search, graph_search, multi_doc_synthesizer, etc.)
    ↓
Retrieval (HyDE + Expansion Fusion → PostgreSQL pgvector)
    ↓
Storage (PostgreSQL: vectors + graph + checkpoints)
```

**Key directories:**
- `src/agent/` - Agent CLI and tools (`tools/` has individual tool files)
- `src/multi_agent/` - LangGraph-based multi-agent system (orchestrator, 7 specialized agents)
- `src/retrieval/` - HyDE + Expansion Fusion retrieval pipeline
- `src/graph/` - Knowledge graph extraction and storage
- `backend/` - FastAPI web backend with auth, routes, middleware
- `frontend/` - React + Vite web UI

## Critical Constraints (DO NOT CHANGE)

These are research-backed decisions from published papers. **DO NOT modify** without explicit approval.

### 1. SINGLE SOURCE OF TRUTH (SSOT)

**Principles:**
- **One canonical implementation** - Each feature has EXACTLY ONE authoritative version
- **No duplicate code** - Delete obsolete implementations immediately
- **No legacy code** - Remove unused/deprecated code
- **API keys in `.env` ONLY** - NEVER in config.json, NEVER in code

**Why:** Prevents confusion, reduces maintenance burden, protects secrets.

### 2. Autonomous Agents (NOT Hardcoded)

**CRITICAL: Agents MUST be LLM-driven, NOT hardcoded workflows!**

```python
# ❌ WRONG (Hardcoded)
def execute():
    step1 = call_tool_a()  # Predefined sequence
    step2 = call_tool_b()
    return synthesize(step1, step2)

# ✅ CORRECT (Autonomous)
def execute():
    return llm.run(
        system_prompt="You are an expert...",
        tools=[search, analyze, verify],  # LLM decides sequence
        messages=[user_query]
    )
```

**Principles:**
- LLM decides tool calling sequence autonomously
- No "step 1, step 2, step 3" logic in code
- System prompts guide behavior (NOT code)
- Exception: Orchestrator has routing logic

Agents inherit from `BaseAgent.run_autonomous_tool_loop()`. LLM decides tool calling order.

### 3. Hierarchical Document Summaries

**NEVER pass full document text to LLM for summarization!**

```
Flow: Sections → Section Summaries → Document Summary
```

- **Why:** Prevents context overflow, handles 100+ page documents
- **Implementation:** PHASE 2 generates section summaries, then aggregates
- **Length:** 100-1000 chars (adaptive based on document complexity)
- **Fallback:** `"(Document summary unavailable)"` if section summaries fail

### 4. Token-Aware Chunking

- **Max tokens:** 512 (tiktoken, text-embedding-3-large tokenizer)
- **Research:** LegalBench-RAG optimal for legal documents
- **Why tokens not chars:** Guarantees embedding model compatibility, handles Czech diacritics
- **Warning:** Changing this invalidates ALL vector stores!

### 5. Summary-Augmented Chunking (SAC)

- Prepend document summary during embedding
- Strip summaries during retrieval
- **Result:** -58% context drift (Anthropic, 2024)

### 6. Multi-Layer Embeddings

- **3 separate indexes** (document/section/chunk) - NOT merged
- **Result:** 2.3x essential chunks vs single-layer (Lima, 2024)

### 6.1 Chunk JSON Format (phase3_chunks.json)

**IMPORTANT:** Chunks serialized to JSON use this format. Do NOT use `content` field!

```json
{
  "chunk_id": "doc_L3_c1_sec_1",
  "context": "SAC context summary (what chunk is about)",
  "raw_content": "Actual text content from the document",
  "embedding_text": "[breadcrumb]\n\ncontext\n\nraw_content",
  "metadata": { "chunk_id": "...", "layer": 3, "document_id": "..." }
}
```

| Field | Purpose |
|-------|---------|
| `context` | SAC context summary - LLM-generated description of what the chunk contains |
| `raw_content` | Actual document text (used for LLM generation, NOT just titles) |
| `embedding_text` | `[breadcrumb]\n\ncontext\n\nraw_content` - full text for embedding |

**PhaseLoaders**: When loading chunks from JSON, use `embedding_text` as the Chunk's `content` field.

### 6.2 Chunked PDF Extraction Deduplication

When Gemini extracts large PDFs in chunks (TOC pages first, content pages later), duplicate sections appear:
- **TOC sections** (`c1_sec_*`): contain only section titles as content
- **Content sections** (`c2_sec_*`): contain actual text

**Solution**: `GeminiKGExtractor._deduplicate_sections_by_path()` merges sections with the same hierarchical path, keeping the one with the longest content. This happens automatically during chunked extraction.

### 7. No Cohere Reranking

Cohere performs WORSE on legal docs. Use `ms-marco` or `bge-reranker` instead.

### 8. Generic Summaries (Counterintuitive!)

- **Style:** GENERIC (NOT expert terminology)
- **Research:** Reuter et al. (2024) - generic summaries improve retrieval

### 9. Autonomous Agent Responses

**NEVER use hardcoded template responses!**

```python
# ❌ WRONG
if is_greeting(query):
    return "Hello! How can I help?"

# ✅ CORRECT
orchestrator.run(query)  # LLM generates contextual response
```

## Configuration

**Two-file system:**

1. **`.env`** - Secrets (gitignored)
   - API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`
   - Database: `DATABASE_URL`, `POSTGRES_PASSWORD`
   - Auth: `AUTH_SECRET_KEY`

2. **`config.json`** - Settings (version-controlled)
   - Models, retrieval method, agent config, pipeline params
   - **NO secrets allowed!**

**Key config.json settings:**
```json
{
  "retrieval": {
    "method": "hyde_expansion_fusion",
    "hyde_weight": 0.6,
    "expansion_weight": 0.4
  },
  "storage": {
    "backend": "postgresql"
  },
  "agent": {
    "model": "claude-haiku-4-5"
  }
}
```

**Extraction Backend Selection:**
```bash
# Environment variable (default: "auto")
EXTRACTION_BACKEND=auto     # Use Gemini if GOOGLE_API_KEY available, else Unstructured
EXTRACTION_BACKEND=gemini   # Force Gemini (requires GOOGLE_API_KEY)
EXTRACTION_BACKEND=unstructured  # Force Unstructured
```

## Best Practices

### SSOT (Single Source of Truth)

- **One implementation per feature** - delete obsolete code immediately
- **No duplicate helpers** - use `src/utils/` for shared functions
- **API keys in `.env` ONLY** - never in config.json or code

### Code Quality

- Type hints required for public APIs
- Google-style docstrings
- Graceful degradation (e.g., reranker unavailable → fall back)
- TDD: Write tests BEFORE implementing features
- Narrow exception catches - catch specific exceptions, not bare `Exception`

### Error Handling

- Catch specific exceptions (e.g., `ValueError`, `RuntimeError`, `json.JSONDecodeError`)
- Log errors with context (file name, operation, exception type)
- Use `exc_info=True` for unexpected errors to capture traceback
- Provide actionable error messages for user-fixable issues

### Git Workflow

- Use `gh` CLI for PRs: `gh pr create --title "..." --body "..."`
- Update CLAUDE.md when making major architectural changes

### Model Selection

- **Production:** `claude-sonnet-4-5`
- **Development:** `gpt-4o-mini` (best cost/performance)
- **Budget:** `claude-haiku-4-5` (fastest)

### LangSmith Observability

**Configuration:**
```bash
# .env (EU endpoint for EU workspaces)
LANGSMITH_API_KEY=lsv2_pt_xxx
LANGSMITH_PROJECT_NAME=sujbot2-multi-agent
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com  # US: https://api.smith.langchain.com
```

**Accessing Traces:**
```bash
# List projects
curl -s "https://eu.api.smith.langchain.com/api/v1/sessions" \
  -H "X-API-Key: $LANGSMITH_API_KEY"

# Query runs (requires session ID as list)
curl -s "https://eu.api.smith.langchain.com/api/v1/runs/query" \
  -H "X-API-Key: $LANGSMITH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"session": ["SESSION_ID"], "limit": 10}'

# Query specific trace
curl -s "https://eu.api.smith.langchain.com/api/v1/runs/query" \
  -H "X-API-Key: $LANGSMITH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"trace": "TRACE_ID", "limit": 100}'
```

**Key Metrics to Monitor:**
- **Latency per agent**: extractor, orchestrator_synthesis, compliance
- **Token usage**: prompt_tokens, completion_tokens (track overflow)
- **Tool calls**: which tools called, how many iterations
- **Error rate**: failed runs, timeout patterns

**Common Issues:**
- `403 Forbidden`: Wrong endpoint (EU vs US) or invalid API key
- `0 tokens`: Token counting may not propagate in LangGraph chains
- Double agent execution: Check workflow routing logic

## Research Papers (DO NOT CONTRADICT)

1. **LegalBench-RAG** (Pipitone & Alami, 2024) - RCTS, 500-char chunks
2. **Summary-Augmented Chunking** (Reuter et al., 2024) - SAC, generic summaries
3. **Multi-Layer Embeddings** (Lima, 2024) - 3-layer indexing
4. **Contextual Retrieval** (Anthropic, 2024) - Context prepending
5. **HybridRAG** (2024) - Graph boosting
6. **HyDE** (Gao et al., 2022) - Hypothetical Document Embeddings

## Documentation

- [`README.md`](README.md) - Installation, quick start
- [`docs/DOCKER_SETUP.md`](docs/DOCKER_SETUP.md) - Docker configuration
- [`docs/HITL_IMPLEMENTATION_SUMMARY.md`](docs/HITL_IMPLEMENTATION_SUMMARY.md) - Human-in-the-Loop
- [`docs/WEB_INTERFACE.md`](docs/WEB_INTERFACE.md) - Web UI features

---

**Last Updated:** 2025-11-25
**Version:** PHASE 1-7 + HyDE Expansion Fusion + Multi-Agent + Gemini Extractor + Docker
