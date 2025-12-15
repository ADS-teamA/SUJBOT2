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
mcp__langsmith__list_datasets      # List evaluation datasets
```

## LangSmith Evaluation

Run QA evaluation on indexed documents using LLM-as-judge.

**When user says "evaluuj v langsmith", run this command:**
```bash
uv run python scripts/langsmith_eval.py \
    --dataset-path dataset/dataset_exp_ver_2.json \
    --dataset-name "sujbot2-eval-qa-40" \
    --replace-dataset \
    --experiment-prefix "sujbot2-qa-40-optimized" \
    --judge-model anthropic:claude-sonnet-4-5
```

**Other commands:**
```bash
# Quick test (5 examples)
uv run python scripts/langsmith_eval.py --limit 5

# Upload dataset only
uv run python scripts/langsmith_eval.py --upload-only

# With cheaper judge model
uv run python scripts/langsmith_eval.py --judge-model openai:gpt-4o-mini
```

**Metrics evaluated:**
- `semantic_correctness` - Does answer match reference meaning?
- `factual_accuracy` - Are numbers/names/dates correct?
- `completeness` - Are all key points covered?

**Dataset:** `dataset/dataset_exp_ver_2.json` (40 Czech legal/nuclear QA pairs)
**LangSmith Dataset:** `sujbot2-eval-qa-40`

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

# Frontend development (HOT RELOAD)
# Frontend runs with Vite hot reload - NO rebuild needed for .tsx/.css changes!
# Just edit files and changes appear instantly in browser.

# Agent CLI
uv run python -m src.agent.cli             # Interactive mode
uv run python -m src.agent.cli --debug     # Debug mode

# Evaluation
uv run python scripts/langsmith_eval.py             # Full QA evaluation
uv run python scripts/langsmith_eval.py --limit 5   # Quick test
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
- `src/graph/` - Graphiti temporal knowledge graph (Neo4j + PostgreSQL)
- `backend/` - FastAPI web backend with auth, routes, middleware
- `frontend/` - React + Vite web UI

**Key utility modules:**
- `src/exceptions.py` - Typed exception hierarchy (SSOT for error handling)
- `src/utils/cache.py` - Unified `LRUCache` + `TTLCache` abstractions
- `src/multi_agent/core/agent_initializer.py` - SSOT for agent initialization
- `src/multi_agent/prompts/loader.py` - SSOT for loading system prompts from `prompts/`
- `src/agent/providers/factory.py` - Provider creation + `detect_provider_from_model()`

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

### 6.3 PostgreSQL Vector Schema

**IMPORTANT:** Vectors are stored in the `vectors` schema (NOT `public`). Always use `vectors.layer{n}` when querying.

```sql
-- Schema: vectors
-- Tables: layer1 (documents), layer2 (sections), layer3 (chunks)

-- Primary table for RAG retrieval: vectors.layer3
-- Columns:
--   id (serial)           - Primary key
--   chunk_id (text)       - Unique chunk identifier (e.g., "BZ_VR1_L3_266")
--   document_id (text)    - Source document (e.g., "BZ_VR1")
--   section_id (text)     - Section identifier (e.g., "sec_383")
--   section_title (text)  - Section heading
--   section_path (text)   - Hierarchical breadcrumb path
--   embedding (vector)    - 4096-dimension vector (Qwen3-Embedding-8B)
--   content (text)        - SAC-formatted text: "[breadcrumb]\n\ncontext\n\nraw_content"
--   content_tsv (tsvector)- Full-text search index
--   metadata (jsonb)      - Additional chunk metadata
--   created_at (timestamp)
```

**Key implementation details:**
- Embeddings: 4096 dimensions via `Qwen/Qwen3-Embedding-8B` (DeepInfra)
- Content field: Contains SAC context prepended (breadcrumb format for retrieval)
- Cosine similarity: `1 - (embedding <=> query_vector)`

**Query example:**
```sql
SELECT chunk_id, content, 1 - (embedding <=> $1::vector) AS similarity
FROM vectors.layer3
WHERE document_id = 'BZ_VR1'
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

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

**SSOT Modules (use these, don't duplicate):**
```python
# Agent initialization (all 8 agents use this)
from src.multi_agent.core.agent_initializer import initialize_agent
components = initialize_agent(config, "agent_name")

# Caching (thread-safe, with hit rate tracking)
from src.utils.cache import LRUCache, TTLCache
cache = LRUCache[str](max_size=500, name="my_cache")
cache.set("key", "value")
result = cache.get("key")  # None if not found

# Provider detection (don't inline this logic!)
from src.agent.providers.factory import detect_provider_from_model
provider = detect_provider_from_model("claude-sonnet-4")  # → "anthropic"

# Load prompts (SSOT: prompts/ directory)
from src.multi_agent.prompts.loader import load_prompt, get_prompt_loader
system_prompt = load_prompt("extractor")  # Loads prompts/agents/extractor.txt
```

### System Prompts (SSOT: `prompts/` directory)

**CRITICAL: ALL LLM system prompts MUST be loaded from `prompts/` directory!**

```
prompts/
├── agents/                    # Multi-agent system prompts
│   ├── orchestrator.txt       # Orchestrator routing/synthesis
│   ├── extractor.txt          # Extractor agent
│   ├── classifier.txt         # Classifier agent
│   ├── compliance.txt         # Compliance agent
│   ├── risk_verifier.txt      # Risk verifier agent
│   ├── requirement_extractor.txt
│   ├── citation_auditor.txt
│   └── gap_synthesizer.txt
├── document_summary.txt       # Document summary generation
├── section_summary.txt        # Section summary generation
├── hyde_expansion.txt         # HyDE query expansion
├── entity_extraction.txt      # Entity extraction (legacy)
└── relationship_extraction.txt # Relationship extraction (legacy)
```

**How to load prompts:**
```python
# For multi-agent system (agents in prompts/agents/)
from src.multi_agent.prompts.loader import load_prompt, get_prompt_loader
system_prompt = load_prompt("extractor")  # → prompts/agents/extractor.txt

# For pipeline components (prompts in prompts/)
from pathlib import Path
prompts_dir = Path(__file__).parent.parent / "prompts"
prompt = (prompts_dir / "document_summary.txt").read_text()
```

**Rules:**
- ❌ **NEVER** hardcode system prompts in Python code
- ❌ **NEVER** use inline f-strings for complex system prompts
- ✅ **ALWAYS** load from `prompts/` directory
- ✅ **ALWAYS** use `PromptLoader` for agent prompts (supports caching, hot-reload)

**Why:** Prompts are easier to iterate on, version control, and review when externalized.

### Code Quality

- Type hints required for public APIs
- Google-style docstrings
- Graceful degradation (e.g., reranker unavailable → fall back)
- TDD: Write tests BEFORE implementing features
- Narrow exception catches - catch specific exceptions, not bare `Exception`

### Error Handling

**Use typed exceptions from `src/exceptions.py`:**
```python
from src.exceptions import (
    ExtractionError, ValidationError, ProviderError,
    APIKeyError, ToolExecutionError, AgentInitializationError
)

# ✅ CORRECT - typed exception
raise APIKeyError(
    "Missing API key for model",
    details={"model": model_name},
    cause=original_exception
)

# ❌ WRONG - generic exception
raise ValueError("Missing API key")
```

**Exception hierarchy:**
- `SujbotError` (base) → `ExtractionError`, `ValidationError`, `ProviderError`, `ToolExecutionError`, `AgentError`, `StorageError`, `RetrievalError`
- Each has `message`, `details` dict, and optional `cause` for chaining

**Best practices:**
- Catch specific exceptions (e.g., `APIKeyError`, not bare `Exception`)
- Log errors with context (file name, operation, exception type)
- Use `exc_info=True` for unexpected errors to capture traceback
- Use `wrap_exception()` helper to convert generic exceptions

### Internationalization (i18n)

Frontend supports CZ/EN language switching via react-i18next.

**Rules:**
- **Always maintain translations**: When adding/changing UI text, update BOTH files:
  - `/frontend/src/i18n/locales/cs.json` - Czech
  - `/frontend/src/i18n/locales/en.json` - English
- **Use useTranslation hook**: `const { t } = useTranslation()`
- **Hierarchical keys**: `section.subsection.key` (e.g., `login.signIn`)
- **No hardcoded strings**: All user-visible text must use `t('key')`

**Translation files structure:**
```json
{
  "header": { "tagline": "...", "signOut": "..." },
  "login": { "email": "...", "password": "...", "signIn": "..." },
  "sidebar": { "newChat": "...", "noConversations": "..." },
  "chat": { "placeholder": "...", "processing": "..." },
  "welcome": { "suggestedQuestions": "..." },
  "common": { "loading": "...", "verifyingSession": "..." }
}
```

### Git Workflow

- Use `gh` CLI for PRs: `gh pr create --title "..." --body "..."`
- Update CLAUDE.md when making major architectural changes

### Adding New Backend API Routes

**IMPORTANT:** When adding new backend API endpoints, you MUST update nginx routing!

The nginx reverse proxy (`docker/nginx/reverse-proxy.conf`) routes requests based on URL patterns.
New backend routes must be added to the location regex in the "Direct backend endpoints" section:

```nginx
# In reverse-proxy.conf, find "# Direct backend endpoints" section
location ~ ^/(health|docs|openapi.json|chat|models|clarify|auth|conversations|settings|documents) {
    proxy_pass http://backend;
    ...
}
```

**Checklist for new backend routes:**
1. Create route file in `backend/routes/` with `APIRouter(prefix="/newroute", tags=["newroute"])`
2. Register router in `backend/main.py`:
   - Add import: `from backend.routes.newroute import router as newroute_router`
   - Add registration: `app.include_router(newroute_router)`
3. (If needed) Add dependency setter function and call it in `main.py` lifespan handler
4. **Add route to nginx regex** in `docker/nginx/reverse-proxy.conf`
5. Reload nginx: `docker compose exec nginx nginx -s reload`

**Alternative:** Routes prefixed with `/api/` are automatically proxied to backend (rewritten to remove `/api/` prefix). No nginx changes needed for `/api/*` routes.

**Note:** Admin routes (`/admin/*`) have separate location blocks in nginx config (lines 149-174).

**Symptom if forgotten:** Frontend receives HTML (`<!doctype...`) instead of JSON - nginx falls through to frontend catch-all and returns `index.html`.

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

**Last Updated:** 2025-12-15
**Version:** PHASE 1-7 + Multi-Agent + Graphiti KG + Gemini Extractor + Exception Hierarchy + SSOT Refactoring + LangSmith Evaluation + Prompts SSOT
