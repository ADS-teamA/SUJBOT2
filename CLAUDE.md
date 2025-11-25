# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

**SUJBOT2**: Production RAG system for legal/technical documents with multi-agent orchestration.

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

These are research-backed decisions from published papers.

### 1. Autonomous Agents (NOT Hardcoded)

```python
# ❌ WRONG
def execute():
    step1 = call_tool_a()  # Predefined sequence
    step2 = call_tool_b()
    return synthesize(step1, step2)

# ✅ CORRECT
def execute():
    return llm.run(
        system_prompt="You are an expert...",
        tools=[search, analyze, verify],  # LLM decides sequence
        messages=[user_query]
    )
```

Agents inherit from `BaseAgent.run_autonomous_tool_loop()`. LLM decides tool calling order.

### 2. Hierarchical Document Summaries

**NEVER pass full document text to LLM for summarization!**

```
Flow: Sections → Section Summaries → Document Summary
```

PHASE 2 generates section summaries first, then aggregates. Prevents context overflow on 100+ page docs.

### 3. Token-Aware Chunking

- **Max tokens:** 512 (tiktoken, text-embedding-3-large tokenizer)
- **Research:** LegalBench-RAG optimal for legal documents
- **Warning:** Changing this invalidates ALL vector stores!

### 4. Summary-Augmented Chunking (SAC)

- Prepend document summary during embedding
- Strip summaries during retrieval
- **Result:** -58% context drift (Anthropic, 2024)

### 5. Multi-Layer Embeddings

- **3 separate indexes** (document/section/chunk) - NOT merged
- **Result:** 2.3x essential chunks vs single-layer (Lima, 2024)

### 6. No Cohere Reranking

Cohere performs WORSE on legal docs. Use `ms-marco` or `bge-reranker` instead.

### 7. Generic Summaries (Counterintuitive!)

- **Style:** GENERIC (NOT expert terminology)
- **Research:** Reuter et al. (2024) - generic summaries improve retrieval

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
    "method": "hyde_expansion_fusion",  // Current default
    "hyde_weight": 0.6,
    "expansion_weight": 0.4
  },
  "storage": {
    "backend": "postgresql"  // or "faiss" for dev
  },
  "agent": {
    "model": "claude-haiku-4-5"
  }
}
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

### Git Workflow

- Use `gh` CLI for PRs: `gh pr create --title "..." --body "..."`
- Update CLAUDE.md when making major architectural changes

### Model Selection

- **Production:** `claude-sonnet-4-5`
- **Development:** `gpt-4o-mini` (best cost/performance)
- **Budget:** `claude-haiku-4-5` (fastest)

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

**Last Updated:** 2025-11-24
**Version:** PHASE 1-7 + HyDE Expansion Fusion + Multi-Agent + Docker
