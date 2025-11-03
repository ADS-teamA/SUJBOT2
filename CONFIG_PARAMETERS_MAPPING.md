# Configuration Parameters Mapping - MY_SUJBOT RAG System

**Last Updated:** 2025-11-03
**Purpose:** Complete mapping of all configuration parameters across all Config classes for creating comprehensive `.env.example`

## Table of Contents

1. [ModelConfig](#modelconfig) - Core model selection
2. [ExtractionConfig](#extractionconfig) - PHASE 1 document extraction
3. [SummarizationConfig](#summarizationconfig) - PHASE 2 summarization
4. [ContextGenerationConfig](#contextgenerationconfig) - PHASE 3A contextual retrieval
5. [ChunkingConfig](#chunkingconfig) - PHASE 3 chunking
6. [EmbeddingConfig](#embeddingconfig) - PHASE 4 embeddings
7. [PipelineConfig](#pipelineconfig) - General pipeline settings
8. [RAGConfig](#ragconfig) - Unified RAG configuration
9. [ClusteringConfig](#clusteringconfig) - PHASE 4.5 semantic clustering
10. [AgentConfig](#agentconfig) - PHASE 7 RAG agent
11. [ToolConfig](#toolconfig) - Agent tool settings
12. [CLIConfig](#cliconfig) - CLI-specific settings
13. [KnowledgeGraphConfig](#knowledgegraphconfig) - Knowledge graph pipeline
14. [EntityExtractionConfig](#entityextractionconfig) - Entity extraction settings
15. [RelationshipExtractionConfig](#relationshipextractionconfig) - Relationship extraction settings
16. [Neo4jConfig](#neo4jconfig) - Neo4j database settings
17. [EntityDeduplicationConfig](#entitydeduplicationconfig) - KG entity deduplication
18. [GraphStorageConfig](#graphstorageconfig) - Graph storage configuration

---

## ModelConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 124-212)

**Purpose:** Central model configuration loaded from environment variables.

### Parameters from Environment

```
LLM_MODEL: "claude-sonnet-4-5-20250929" (str) - LLM model name
  [FROM .env] Currently: gpt-5-nano
  Description: Model for summaries and RAG agent (Claude, OpenAI, Google Gemini)
  Examples: "claude-sonnet-4-5-20250929", "gpt-4o-mini", "gpt-5-nano", "gemini-2.5-flash"

LLM_PROVIDER: auto-detected from LLM_MODEL (str, optional) - LLM provider override
  [FROM .env] Optional - auto-detected if not set
  Description: "claude", "openai", or "google" (auto-detected from model name)
  Auto-detection: "claude-*" → "claude", "gpt-*" → "openai", "gemini-*" → "google"

EMBEDDING_MODEL: "bge-m3" (str) - Embedding model name
  [FROM .env] Currently: text-embedding-3-large
  Description: Embedding model (HuggingFace local, OpenAI, or Voyage AI)
  Examples: "bge-m3" (local), "text-embedding-3-large", "voyage-3-large"

EMBEDDING_PROVIDER: auto-detected from EMBEDDING_MODEL (str, optional) - Embedding provider override
  [FROM .env] Optional - auto-detected if not set
  Description: "huggingface", "openai", or "voyage" (auto-detected from model name)
  Auto-detection: "bge-*" → "huggingface", "text-embedding-*" → "openai", "voyage-*" → "voyage"

ANTHROPIC_API_KEY: (str, required if LLM_PROVIDER=claude) - Anthropic API key
  [FROM .env] Required for Claude models
  Description: API key for Anthropic Claude models
  Get from: https://console.anthropic.com/

OPENAI_API_KEY: (str, required if LLM_PROVIDER=openai or EMBEDDING_PROVIDER=openai) - OpenAI API key
  [FROM .env] Required for OpenAI models
  Description: API key for OpenAI GPT/embeddings
  Get from: https://platform.openai.com/api-keys

VOYAGE_API_KEY: (str, required if EMBEDDING_PROVIDER=voyage) - Voyage AI API key
  [FROM .env] Required for Voyage embeddings
  Description: API key for Voyage AI embeddings
  Get from: https://www.voyageai.com/

GOOGLE_API_KEY: (str, required if LLM_MODEL=gemini-*) - Google Gemini API key
  [FROM .env] Required for Gemini models
  Description: API key for Google Gemini
  Get from: https://aistudio.google.com/apikey
```

### Hardcoded Defaults (No .env Mapping)

```
anthropic_api_key: None (Optional[str]) - From ANTHROPIC_API_KEY or None
openai_api_key: None (Optional[str]) - From OPENAI_API_KEY or None
voyage_api_key: None (Optional[str]) - From VOYAGE_API_KEY or None
```

---

## ExtractionConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 243-308)

**Purpose:** Configuration for Docling extraction (PHASE 1).

### Parameters from Environment

```
OCR_LANGUAGE: "ces,eng" (str, comma-separated) - OCR languages
  [FROM .env] Yes, via from_env() method
  Description: Comma-separated language codes (default: "ces,eng" for Czech+English)
  Options: "ces" (Czech), "eng" (English), "deu" (German), "fra" (French), etc.
  Auto-detection: Use ["auto"] for automatic detection

ENABLE_SMART_HIERARCHY: "true" (str, boolean) - Font-size based hierarchy
  [FROM .env] Yes, via from_env() method
  Description: Enable font-size based hierarchy detection (CRITICAL for hierarchical chunking)
  Values: "true" or "false"
```

### Hardcoded Defaults (Should Consider for .env)

```
enable_ocr: True (bool) - Enable OCR processing
  SHOULD BE IN .env: YES - as ENABLE_OCR

ocr_engine: "tesseract" (str) - OCR engine selection
  SHOULD BE IN .env: YES - as OCR_ENGINE
  Options: "tesseract" (best Czech support, slower), "rapidocr" (3-5x faster, requires: pip install rapidocr_onnxruntime)

ocr_recognition: "accurate" (str) - OCR recognition mode
  SHOULD BE IN .env: YES - as OCR_RECOGNITION (deprecated parameter for Tesseract)
  Options: "accurate" or "fast"

extract_tables: True (bool) - Extract tables from documents
  SHOULD BE IN .env: YES - as EXTRACT_TABLES

table_mode: "ACCURATE" (str) - Table extraction mode
  SHOULD BE IN .env: YES - as TABLE_MODE
  Options: "ACCURATE" (recommended)

extract_hierarchy: True (bool) - Extract document hierarchy
  SHOULD BE IN .env: YES - as EXTRACT_HIERARCHY
  CRITICAL: Required for hierarchical chunking

hierarchy_tolerance: 0.8 (float) - BBox height clustering tolerance (pixels)
  SHOULD BE IN .env: YES - as HIERARCHY_TOLERANCE
  Notes: Lower = stricter clustering, ~0.8 is optimal

generate_summaries: True (bool) - Generate document/section summaries
  SHOULD BE IN .env: YES - as GENERATE_SUMMARIES

summary_model: None (Optional[str]) - Override summary model
  SHOULD BE IN .env: YES - as SUMMARY_MODEL (uses SummarizationConfig default when None)

summary_max_chars: 150 (int) - Generic summary target length
  SHOULD BE IN .env: YES - as SUMMARY_MAX_CHARS
  Notes: Research-optimal value (do not change without testing)

summary_style: "generic" (str) - Summary style
  SHOULD BE IN .env: YES - as SUMMARY_STYLE (must match SummarizationConfig default)
  Options: "generic" (proven better than "expert" - counterintuitive!)

use_batch_api: True (bool) - Use OpenAI Batch API for summaries
  SHOULD BE IN .env: YES - as USE_BATCH_API (mirrors SummarizationConfig behavior)
  Notes: 50% cheaper, async processing

batch_api_poll_interval: 5 (int) - Seconds between batch status checks
  SHOULD BE IN .env: YES - as BATCH_API_POLL_INTERVAL

batch_api_timeout: 43200 (int) - Max wait for batch completion in seconds
  SHOULD BE IN .env: YES - as BATCH_API_TIMEOUT (12 hours default)

generate_markdown: True (bool) - Generate markdown output
  SHOULD BE IN .env: YES - as GENERATE_MARKDOWN

generate_json: True (bool) - Generate JSON output
  SHOULD BE IN .env: YES - as GENERATE_JSON

layout_model: "EGRET_XLARGE" (str) - Docling layout model
  SHOULD BE IN .env: YES - as LAYOUT_MODEL
  Options: "HERON", "EGRET_LARGE", "EGRET_XLARGE" (recommended)
```

---

## SummarizationConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 311-371)

**Purpose:** Configuration for summarization (PHASE 2).

### Parameters from Environment

```
SPEED_MODE: "fast" (str) - Pipeline speed mode
  [FROM .env] Yes, via from_env() method
  Description: Controls use_batch_api behavior
  Values: "fast" (immediate API calls) or "eco" (OpenAI Batch API, 50% cheaper)
  Notes: Affects use_batch_api: True when "eco", False when "fast"

LLM_PROVIDER: (from ModelConfig) - LLM provider for summaries
LLM_MODEL: (from ModelConfig) - Model name for summaries
```

### Hardcoded Defaults (Should Consider for .env)

```
max_chars: 150 (int) - Summary length target
  CURRENTLY: Hardcoded (research-optimal from LegalBench-RAG)
  SHOULD BE IN .env: NO - critical research constraint, don't expose

tolerance: 20 (int) - Length tolerance for summaries
  SHOULD BE IN .env: NO - derived from max_chars

style: "generic" (str) - Summary style
  CURRENTLY: Hardcoded to "generic"
  SHOULD BE IN .env: NO - research finding: generic > expert (counterintuitive!)
  Do not change without extensive testing

temperature: 0.3 (float) - LLM temperature for consistency
  SHOULD BE IN .env: YES - as SUMMARY_TEMPERATURE (research-backed, but user might want to override)

max_tokens: 100 (int) - Max LLM output tokens
  SHOULD BE IN .env: YES - as SUMMARY_MAX_TOKENS (optimized: 150 chars ≈ 40-60 tokens)

retry_on_exceed: True (bool) - Retry if exceeds max_chars
  SHOULD BE IN .env: YES - as SUMMARY_RETRY_ON_EXCEED

max_retries: 3 (int) - Max retry attempts
  SHOULD BE IN .env: YES - as SUMMARY_MAX_RETRIES

max_workers: 20 (int) - Parallel summary generation threads
  SHOULD BE IN .env: YES - as SUMMARY_MAX_WORKERS (OPTIMIZED: 2x faster)

min_text_length: 50 (int) - Minimum text length for summarization
  SHOULD BE IN .env: YES - as SUMMARY_MIN_TEXT_LENGTH

enable_prompt_batching: False (bool) - Batch multiple sections in one API call
  SHOULD BE IN .env: YES - as SUMMARY_ENABLE_BATCHING (DISABLED - JSON overhead slower)

batch_size: 8 (int) - Number of sections per API call (if batching enabled)
  SHOULD BE IN .env: YES - as SUMMARY_BATCH_SIZE

use_batch_api: True (bool) - Use OpenAI Batch API for summaries
  SHOULD BE IN .env: YES - as SUMMARY_USE_BATCH_API (controlled by SPEED_MODE)
  Notes: 50% cheaper, async processing, 12h max wait

batch_api_poll_interval: 5 (int) - Seconds between status checks
  SHOULD BE IN .env: YES - as SUMMARY_BATCH_API_POLL_INTERVAL

batch_api_timeout: 43200 (int) - Max wait for batch completion (seconds)
  SHOULD BE IN .env: YES - as SUMMARY_BATCH_API_TIMEOUT (12 hours default)

provider: (auto from ModelConfig) - Loaded from ModelConfig
model: (auto from ModelConfig) - Loaded from ModelConfig
```

---

## ContextGenerationConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 374-463)

**Purpose:** Configuration for Contextual Retrieval (PHASE 3A) - generates LLM-based context for chunks.

### Parameters from Environment

```
SPEED_MODE: "fast" (str) - Pipeline speed mode (affects use_batch_api)
  [FROM .env] Yes, via from_env() method
  Values: "fast" or "eco" (OpenAI Batch API mode)

LLM_PROVIDER: (from ModelConfig) - LLM provider
LLM_MODEL: (from ModelConfig) - Model name
```

### Hardcoded Defaults (Should Consider for .env)

```
enable_contextual: True (bool) - Enable contextual retrieval
  SHOULD BE IN .env: YES - as ENABLE_CONTEXTUAL
  Notes: Results in 67% reduction in retrieval failures (Anthropic research)

temperature: 0.3 (float) - LLM temperature for consistency
  SHOULD BE IN .env: YES - as CONTEXT_GENERATION_TEMPERATURE

max_tokens: 150 (int) - Max context output tokens (50-100 words target)
  SHOULD BE IN .env: YES - as CONTEXT_GENERATION_MAX_TOKENS

include_surrounding_chunks: True (bool) - Include neighboring chunks for context
  SHOULD BE IN .env: YES - as CONTEXT_INCLUDE_SURROUNDING

num_surrounding_chunks: 1 (int) - Number of chunks before/after to include
  SHOULD BE IN .env: YES - as CONTEXT_NUM_SURROUNDING_CHUNKS

fallback_to_basic: True (bool) - Fallback to basic chunking if context generation fails
  SHOULD BE IN .env: YES - as CONTEXT_FALLBACK_TO_BASIC

batch_size: 20 (int) - Generate contexts in batches
  SHOULD BE IN .env: YES - as CONTEXT_BATCH_SIZE (OPTIMIZED: 2x faster)

max_workers: 10 (int) - Parallel context generation threads
  SHOULD BE IN .env: YES - as CONTEXT_MAX_WORKERS

use_batch_api: True (bool) - Use OpenAI Batch API for context generation
  SHOULD BE IN .env: YES - as CONTEXT_USE_BATCH_API (controlled by SPEED_MODE, 50% cheaper)

batch_api_poll_interval: 5 (int) - Seconds between status checks
  SHOULD BE IN .env: YES - as CONTEXT_BATCH_API_POLL_INTERVAL

batch_api_timeout: 43200 (int) - Max wait for batch completion (12h default)
  SHOULD BE IN .env: YES - as CONTEXT_BATCH_API_TIMEOUT

provider: (auto from ModelConfig) - Loaded from ModelConfig
model: (auto from ModelConfig) - Loaded from ModelConfig
api_key: (auto from ModelConfig) - Loaded from ModelConfig
```

---

## ChunkingConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 466-513)

**Purpose:** Configuration for chunking (PHASE 3).

### Parameters from Environment

```
CHUNK_SIZE: "500" (str, int) - Chunk size in characters
  [FROM .env] Yes, via from_env() method
  Default: "500"
  Notes: Research-optimal value from RCTS (LegalBench-RAG), DO NOT CHANGE without testing

ENABLE_SAC: "true" (str, boolean) - Summary-Augmented Chunking
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"
  Notes: SAC reduces DRM by 58% (Reuter et al.)

SPEED_MODE: "fast" (str) - Pipeline speed mode
  [FROM .env] Yes (passed to ContextGenerationConfig)
  Values: "fast" or "eco"
```

### Hardcoded Defaults (No .env Mapping Needed)

```
method: "RecursiveCharacterTextSplitter" (str) - Chunking algorithm
  SHOULD BE IN .env: NO - fixed architecture choice

chunk_overlap: 0 (int) - Chunk overlap
  SHOULD BE IN .env: NO - RCTS handles naturally via hierarchy, no overlap needed

enable_multi_layer: True (bool) - Enable multi-layer chunking
  SHOULD BE IN .env: NO - core architecture requirement

separators: ["\n\n", "\n", ". ", "; ", ", ", " ", ""] (List[str]) - Text separators
  SHOULD BE IN .env: NO - fixed architecture choice

context_config: ContextGenerationConfig (Optional) - Context generation config
  Auto-initialized from environment via from_env()
```

---

## EmbeddingConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 516-596)

**Purpose:** Unified configuration for embedding generation (PHASE 4).

### Parameters from Environment

```
EMBEDDING_PROVIDER: (from ModelConfig) - Embedding provider
  [FROM .env] Yes (auto-detected or explicit)
  Values: "voyage", "openai", or "huggingface"

EMBEDDING_MODEL: (from ModelConfig) - Embedding model name
  [FROM .env] Yes
  Examples: "text-embedding-3-large" (OpenAI), "bge-m3" (HuggingFace), "voyage-3-large" (Voyage)

EMBEDDING_BATCH_SIZE: "64" (str, int) - Batch size for embedding generation
  [FROM .env] Yes, via from_env() method
  Default: "64"
  Notes: Optimized for performance

EMBEDDING_CACHE_ENABLED: "true" (str, boolean) - Enable embedding cache
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"
  Notes: 40-80% hit rate

EMBEDDING_CACHE_SIZE: "1000" (str, int) - Max embedding cache entries
  [FROM .env] Yes, via from_env() method
  Default: "1000"
```

### Hardcoded Defaults (No .env Mapping Needed)

```
normalize: True (bool) - Normalize embeddings for cosine similarity
  SHOULD BE IN .env: NO - always True for FAISS IndexFlatIP

enable_multi_layer: True (bool) - Multi-layer indexing (document, section, chunk)
  SHOULD BE IN .env: NO - core architecture requirement

dimensions: None (Optional[int]) - Auto-detected from model
  SHOULD BE IN .env: NO - auto-derived
```

---

## PipelineConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 599-605)

**Purpose:** General pipeline configuration.

### Hardcoded Defaults (Should Consider for .env)

```
log_level: "INFO" (str) - Logging level
  SHOULD BE IN .env: YES - as LOG_LEVEL
  Values: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s" (str) - Log format
  SHOULD BE IN .env: NO - standard Python logging format

log_file: "logs/pipeline.log" (str) - Log file path
  SHOULD BE IN .env: YES - as LOG_FILE
```

---

## RAGConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 608-660)

**Purpose:** Unified RAG pipeline configuration combining all sub-configs.

### Sub-Configurations (See Individual Sections)

```
extraction: ExtractionConfig - PHASE 1 settings
summarization: SummarizationConfig - PHASE 2 settings
chunking: ChunkingConfig - PHASE 3 settings
embedding: EmbeddingConfig - PHASE 4 settings
pipeline: PipelineConfig - Pipeline settings
models: ModelConfig - Model selection
```

### No Direct Parameters

This class aggregates all other configs via sub-configs.

---

## ClusteringConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/config.py` (lines 722-784)

**Purpose:** Configuration for semantic clustering (PHASE 4.5).

### Parameters from Environment

```
CLUSTERING_ALGORITHM: "hdbscan" (str) - Clustering algorithm
  [FROM .env] Yes, via from_env() method
  Values: "hdbscan" (automatic cluster count, density-based) or "agglomerative" (hierarchical)

CLUSTERING_N_CLUSTERS: (optional int) - Number of clusters for agglomerative
  [FROM .env] Yes, via from_env() method (parsed as int if present)
  Default: None (auto-detection)
  Notes: Agglomerative only, ignored for HDBSCAN

CLUSTERING_MIN_SIZE: "5" (str, int) - Minimum chunks per cluster (HDBSCAN only)
  [FROM .env] Yes, via from_env() method
  Default: "5"

CLUSTERING_ENABLE_LABELS: "true" (str, boolean) - Generate semantic labels using LLM
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"

CLUSTERING_ENABLE_VIZ: "false" (str, boolean) - Generate UMAP visualization
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"

CLUSTERING_VIZ_DIR: "output/clusters" (str) - Visualization output directory
  [FROM .env] Yes, via from_env() method
  Default: "output/clusters"
```

### Hardcoded Defaults (No .env Mapping Needed)

```
max_clusters: 50 (int) - Maximum clusters for auto-detection (agglomerative)
  SHOULD BE IN .env: YES - as CLUSTERING_MAX_CLUSTERS

min_clusters: 5 (int) - Minimum clusters for auto-detection (agglomerative)
  SHOULD BE IN .env: YES - as CLUSTERING_MIN_CLUSTERS

cluster_layers: [3] (List[int]) - Which layers to cluster
  SHOULD BE IN .env: YES - as CLUSTERING_LAYERS (e.g., "3" or "1,2,3")
  Notes: Default [3] clusters chunk-level only
```

---

## AgentConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/agent/config.py` (lines 169-381)

**Purpose:** Main agent configuration for PHASE 7 RAG agent.

### Parameters from Environment

```
ANTHROPIC_API_KEY: (str) - Anthropic API key
  [FROM .env] Yes, via environment variable
  Description: Required for Claude models

OPENAI_API_KEY: (str) - OpenAI API key
  [FROM .env] Yes, via environment variable
  Description: Optional, for GPT models

GOOGLE_API_KEY: (str) - Google API key
  [FROM .env] Yes, via environment variable
  Description: Optional, for Gemini models

AGENT_MODEL: "claude-sonnet-4-5-20250929" (str) - Agent model selection
  [FROM .env] Yes, via from_env() method
  Default: "claude-sonnet-4-5-20250929"
  Examples: "claude-haiku-4-5", "gpt-4o-mini", "gemini-2.5-flash"

AGENT_MAX_TOKENS: "4096" (str, int) - Max output tokens for agent
  [FROM .env] Yes, via from_env() method
  Default: "4096"
  Notes: Gemini: up to 8192, GPT-5: up to 16384+

VECTOR_STORE_PATH: "vector_db" (str) - Path to phase4_vector_store directory
  [FROM .env] Yes, via from_env() method
  Default: "vector_db"
  CRITICAL: Must point to actual phase4_vector_store

KNOWLEDGE_GRAPH_PATH: (str, optional) - Path to knowledge graph JSON
  [FROM .env] Yes, via from_env() method
  Description: Optional, for KG-based tools

ENABLE_KNOWLEDGE_GRAPH: "false" (str, boolean) - Enable knowledge graph tools
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"

ENABLE_PROMPT_CACHING: "true" (str, boolean) - Enable prompt caching (Anthropic)
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"
  Notes: 90% cost reduction on cached tokens (system prompt, tools, init messages)

ENABLE_CONTEXT_MANAGEMENT: "true" (str, boolean) - Auto-prune old tool results
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"
  Notes: Prevents quadratic cost growth in long conversations

CONTEXT_MANAGEMENT_TRIGGER: "50000" (str, int) - Token threshold for pruning
  [FROM .env] Yes, via from_env() method
  Default: "50000"

CONTEXT_MANAGEMENT_KEEP: "3" (str, int) - Keep last N messages with full tool context
  [FROM .env] Yes, via from_env() method
  Default: "3"

QUERY_EXPANSION_MODEL: "gpt-4o-mini" (str) - Model for query expansion
  [FROM .env] Yes, via from_env() method
  Default: "gpt-4o-mini"
  Notes: Auto-detects provider from model name
```

### Hardcoded Defaults (Should Consider for .env)

```
max_tokens: 4096 (int) - Max output tokens (from environment or default)
  CURRENTLY: Loaded from AGENT_MAX_TOKENS
  SHOULD BE IN .env: YES - via AGENT_MAX_TOKENS

temperature: 0.3 (float) - LLM temperature
  SHOULD BE IN .env: YES - as AGENT_TEMPERATURE

enable_tool_validation: True (bool) - Validate tools on startup
  SHOULD BE IN .env: YES - as AGENT_ENABLE_TOOL_VALIDATION

debug_mode: False (bool) - Enable debug logging
  SHOULD BE IN .env: YES - as AGENT_DEBUG_MODE
```

---

## ToolConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/agent/config.py` (lines 84-137)

**Purpose:** Configuration for RAG tools.

### Hardcoded Defaults (Should Consider for .env)

```
default_k: 6 (int) - Default number of results to retrieve
  SHOULD BE IN .env: YES - as TOOL_DEFAULT_K

enable_reranking: True (bool) - Enable cross-encoder reranking
  SHOULD BE IN .env: YES - as TOOL_ENABLE_RERANKING

reranker_candidates: 50 (int) - Number of candidates before reranking
  SHOULD BE IN .env: YES - as TOOL_RERANKER_CANDIDATES
  Notes: Must be >= default_k

reranker_model: "bge-reranker-large" (str) - Reranker model
  SHOULD BE IN .env: YES - as TOOL_RERANKER_MODEL
  Notes: SOTA accuracy (was: ms-marco-mini)

enable_graph_boost: True (bool) - Enable graph-based result boosting
  SHOULD BE IN .env: YES - as TOOL_ENABLE_GRAPH_BOOST
  Notes: +200-500ms overhead, +8% factual correctness on entity queries

graph_boost_weight: 0.3 (float) - Weight for graph boosting
  SHOULD BE IN .env: YES - as TOOL_GRAPH_BOOST_WEIGHT
  Range: [0.0, 1.0]

max_document_compare: 3 (int) - Max documents to compare
  SHOULD BE IN .env: YES - as TOOL_MAX_DOCUMENT_COMPARE

compliance_threshold: 0.7 (float) - Legal compliance threshold
  SHOULD BE IN .env: YES - as TOOL_COMPLIANCE_THRESHOLD
  Notes: HIGH (≥0.85), MEDIUM (0.70-0.84), LOW (0.50-0.69)

context_window: 2 (int) - Chunks before/after for context expansion
  SHOULD BE IN .env: YES - as TOOL_CONTEXT_WINDOW

query_expansion_provider: "openai" (str) - Provider for query expansion
  SHOULD BE IN .env: YES - as QUERY_EXPANSION_PROVIDER
  Values: "openai" or "anthropic"
  Notes: Auto-detected from QUERY_EXPANSION_MODEL

query_expansion_model: "gpt-4o-mini" (str) - Model for query expansion
  SHOULD BE IN .env: YES - as QUERY_EXPANSION_MODEL
  Notes: Stable, fast model for expansion

lazy_load_reranker: False (bool) - Load reranker at startup
  SHOULD BE IN .env: YES - as TOOL_LAZY_LOAD_RERANKER

lazy_load_graph: True (bool) - Load graph lazily
  SHOULD BE IN .env: YES - as TOOL_LAZY_LOAD_GRAPH

cache_embeddings: True (bool) - Cache embeddings
  SHOULD BE IN .env: YES - as TOOL_CACHE_EMBEDDINGS
```

---

## CLIConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/agent/config.py` (lines 140-166)

**Purpose:** CLI-specific configuration.

### Hardcoded Defaults (Should Consider for .env)

```
show_citations: True (bool) - Display citations in responses
  SHOULD BE IN .env: YES - as CLI_SHOW_CITATIONS

citation_format: "inline" (str) - Citation format style
  SHOULD BE IN .env: YES - as CLI_CITATION_FORMAT
  Options: "inline", "footnote", "detailed", "simple"

show_tool_calls: True (bool) - Display tool calls in output
  SHOULD BE IN .env: YES - as CLI_SHOW_TOOL_CALLS

show_timing: True (bool) - Display timing information
  SHOULD BE IN .env: YES - as CLI_SHOW_TIMING

enable_streaming: True (bool) - Enable streaming responses
  SHOULD BE IN .env: YES - as CLI_ENABLE_STREAMING

save_history: True (bool) - Save conversation history
  SHOULD BE IN .env: YES - as CLI_SAVE_HISTORY

history_file: ".agent_history" (str) - History file path
  SHOULD BE IN .env: YES - as CLI_HISTORY_FILE

max_history_items: 1000 (int) - Max history items to keep
  SHOULD BE IN .env: YES - as CLI_MAX_HISTORY_ITEMS
```

---

## KnowledgeGraphConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/graph/config.py` (lines 309-400)

**Purpose:** Complete configuration for Knowledge Graph pipeline.

### Parameters from Environment

```
KG_LLM_PROVIDER: "openai" (str) - LLM provider for entity/relationship extraction
  [FROM .env] Yes, via from_env() method
  Default: "openai"

KG_LLM_MODEL: "gpt-4o-mini" (str) - Model for entity/relationship extraction
  [FROM .env] Yes, via from_env() method
  Default: "gpt-4o-mini"
  Notes: OPTIMIZED: 70% cheaper than gpt-5-mini

KG_BACKEND: "simple" (str) - Graph storage backend
  [FROM .env] Yes, via from_env() method
  Default: "simple"
  Values: "simple" (JSON), "neo4j", "networkx"
  Notes: "neo4j" recommended for production (ALL TOOLS)

KG_EXPORT_PATH: "./data/graphs/knowledge_graph.json" (str) - JSON export path
  [FROM .env] Yes, via from_env() method

KG_VERBOSE: "true" (str, boolean) - Enable verbose logging for KG extraction
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"

ANTHROPIC_API_KEY: (str) - API key if using Anthropic for KG extraction
  [FROM .env] Yes, loaded from environment

OPENAI_API_KEY: (str) - API key if using OpenAI for KG extraction
  [FROM .env] Yes, loaded from environment
```

### Hardcoded Defaults (Should Consider for .env)

```
enable_entity_extraction: True (bool) - Enable entity extraction
  SHOULD BE IN .env: YES - as ENABLE_ENTITY_EXTRACTION

enable_relationship_extraction: True (bool) - Enable relationship extraction
  SHOULD BE IN .env: YES - as ENABLE_RELATIONSHIP_EXTRACTION

enable_cross_document_relationships: False (bool) - Extract cross-doc relationships
  SHOULD BE IN .env: YES - as ENABLE_CROSS_DOCUMENT_RELATIONSHIPS
  Notes: Expensive, for multi-doc graphs

max_retries: 3 (int) - Retry failed extractions
  SHOULD BE IN .env: YES - as KG_MAX_RETRIES

retry_delay: 1.0 (float) - Seconds between retries
  SHOULD BE IN .env: YES - as KG_RETRY_DELAY

timeout: 300 (int) - Seconds per extraction batch
  SHOULD BE IN .env: YES - as KG_TIMEOUT

log_path: "./logs/kg_extraction.log" (str) - KG extraction log path
  SHOULD BE IN .env: YES - as KG_LOG_PATH
```

---

## EntityExtractionConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/graph/config.py` (lines 25-89)

**Purpose:** Configuration for entity extraction.

### Hardcoded Defaults (Should Consider for .env)

```
llm_provider: "openai" (str) - LLM provider for extraction
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_LLM_PROVIDER
  Values: "openai", "anthropic"

llm_model: "gpt-4o-mini" (str) - Model for extraction
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_LLM_MODEL
  Notes: Fast, cost-effective

temperature: 0.0 (float) - Deterministic extraction
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_TEMPERATURE
  Notes: Should remain 0.0 for consistency

min_confidence: 0.6 (float) - Minimum confidence threshold
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_MIN_CONFIDENCE

extract_definitions: True (bool) - Extract entity definitions
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_EXTRACT_DEFINITIONS

normalize_entities: True (bool) - Normalize entity values
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_NORMALIZE

batch_size: 20 (int) - Chunks per batch (OPTIMIZED: 2x faster)
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_BATCH_SIZE

max_workers: 10 (int) - Parallel extraction threads
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_MAX_WORKERS

cache_results: True (bool) - Cache extraction results
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_CACHE_RESULTS

include_examples: True (bool) - Include few-shot examples in prompt
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_INCLUDE_EXAMPLES

max_entities_per_chunk: 50 (int) - Max entities per chunk
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_MAX_PER_CHUNK

enabled_entity_types: {30 types} (Set[EntityType]) - All 30 entity types enabled
  SHOULD BE IN .env: YES - as ENTITY_EXTRACTION_ENABLED_TYPES (comma-separated)
  Core: STANDARD, ORGANIZATION, DATE, CLAUSE, TOPIC, PERSON, LOCATION, CONTRACT
  Regulatory: REGULATION, DECREE, DIRECTIVE, TREATY, LEGAL_PROVISION, REQUIREMENT
  Authorization: PERMIT, LICENSE_CONDITION
  Nuclear: REACTOR, FACILITY, SYSTEM, SAFETY_FUNCTION, FUEL_TYPE, ISOTOPE, etc.
  Events: INCIDENT, EMERGENCY_CLASSIFICATION, INSPECTION, DECOMMISSIONING_PHASE
  Liability: LIABILITY_REGIME
```

---

## RelationshipExtractionConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/graph/config.py` (lines 92-173)

**Purpose:** Configuration for relationship extraction.

### Hardcoded Defaults (Should Consider for .env)

```
llm_provider: "openai" (str) - LLM provider for extraction
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_LLM_PROVIDER

llm_model: "gpt-4o-mini" (str) - Model for extraction
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_LLM_MODEL

temperature: 0.0 (float) - Deterministic extraction
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_TEMPERATURE

min_confidence: 0.5 (float) - Minimum confidence threshold
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_MIN_CONFIDENCE

extract_evidence: True (bool) - Extract supporting evidence text
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_EXTRACT_EVIDENCE

max_evidence_length: 200 (int) - Max characters for evidence
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_MAX_EVIDENCE_LENGTH

extract_within_chunk: True (bool) - Extract relationships within single chunk
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_WITHIN_CHUNK

extract_cross_chunk: True (bool) - Extract relationships across chunks
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_CROSS_CHUNK

extract_from_metadata: True (bool) - Extract from chunk metadata
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_FROM_METADATA

batch_size: 10 (int) - Entity pairs per batch (OPTIMIZED: 2x faster)
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_BATCH_SIZE

max_workers: 10 (int) - Parallel extraction threads
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_MAX_WORKERS

cache_results: True (bool) - Cache extraction results
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_CACHE_RESULTS

max_relationships_per_entity: 100 (int) - Limit relationships per entity
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_MAX_PER_ENTITY

enabled_relationship_types: {40 types} (Set[RelationshipType]) - All 40 types enabled
  SHOULD BE IN .env: YES - as RELATIONSHIP_EXTRACTION_ENABLED_TYPES (comma-separated)
  Compliance: COMPLIES_WITH, CONTRADICTS, PARTIALLY_SATISFIES, SPECIFIES_REQUIREMENT, REQUIRES_CLAUSE
  Regulatory: IMPLEMENTS, TRANSPOSES, SUPERSEDED_BY, SUPERSEDES, AMENDS
  Structure: CONTAINS_CLAUSE, CONTAINS_PROVISION, CONTAINS, PART_OF
  Citations: REFERENCES, REFERENCED_BY, CITES_PROVISION, BASED_ON
  Authorization: ISSUED_BY, GRANTED_BY, ENFORCED_BY, SUBJECT_TO_INSPECTION, SUPERVISES
  Technical: REGULATED_BY, OPERATED_BY, HAS_SYSTEM, PERFORMS_FUNCTION, USES_FUEL, CONTAINS_ISOTOPE, PRODUCES_WASTE, HAS_DOSE_LIMIT
  Temporal: EFFECTIVE_DATE, EXPIRY_DATE, SIGNED_ON, DECOMMISSIONED_ON
  Content: COVERS_TOPIC, APPLIES_TO
  Provenance: MENTIONED_IN, DEFINED_IN, DOCUMENTED_IN
```

---

## Neo4jConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/graph/config.py` (lines 176-202)

**Purpose:** Neo4j database configuration.

### Parameters from Environment

```
NEO4J_URI: "bolt://localhost:7687" (str) - Neo4j connection URI
  [FROM .env] Yes, via from_env() method
  Default: "bolt://localhost:7687" (local)
  Examples:
    - Local: "bolt://localhost:7687"
    - Neo4j Aura: "neo4j+s://abc123.databases.neo4j.io"

NEO4J_USERNAME: "neo4j" (str) - Neo4j username
  [FROM .env] Yes, via from_env() method
  Default: "neo4j"

NEO4J_PASSWORD: "" (str) - Neo4j password
  [FROM .env] Yes, via from_env() method
  CRITICAL: Set from environment (cannot be recovered from Aura!)

NEO4J_DATABASE: "neo4j" (str) - Neo4j database name
  [FROM .env] Yes, via from_env() method
  Default: "neo4j"
```

### Hardcoded Defaults (Should Consider for .env)

```
max_connection_lifetime: 3600 (int) - Connection lifetime in seconds
  SHOULD BE IN .env: YES - as NEO4J_MAX_CONNECTION_LIFETIME

max_connection_pool_size: 50 (int) - Max concurrent connections
  SHOULD BE IN .env: YES - as NEO4J_MAX_CONNECTION_POOL_SIZE

connection_timeout: 30 (int) - Connection timeout in seconds
  SHOULD BE IN .env: YES - as NEO4J_CONNECTION_TIMEOUT

create_indexes: True (bool) - Auto-create indexes for entity types
  SHOULD BE IN .env: YES - as NEO4J_CREATE_INDEXES

create_constraints: True (bool) - Auto-create uniqueness constraints
  SHOULD BE IN .env: YES - as NEO4J_CREATE_CONSTRAINTS
```

---

## EntityDeduplicationConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/graph/config.py` (lines 205-278)

**Purpose:** Entity deduplication configuration for incremental Neo4j indexing.

### Parameters from Environment

```
KG_DEDUPLICATE_ENTITIES: "true" (str, boolean) - Master enable/disable
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"
  Notes: Entire 3-layer system controlled by this flag

KG_DEDUP_USE_EMBEDDINGS: "false" (str, boolean) - Layer 2: Semantic similarity
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"
  Notes: Requires embeddings, 50-200ms latency per entity

KG_DEDUP_SIMILARITY_THRESHOLD: "0.90" (str, float) - Cosine similarity threshold
  [FROM .env] Yes, via from_env() method
  Default: "0.90"
  Range: [0.0, 1.0]

KG_DEDUP_USE_ACRONYM_EXPANSION: "false" (str, boolean) - Layer 3: Acronym expansion
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"
  Notes: Domain-specific + fuzzy matching, 100-500ms latency

KG_DEDUP_ACRONYM_FUZZY_THRESHOLD: "0.85" (str, float) - Fuzzy match threshold
  [FROM .env] Yes, via from_env() method
  Default: "0.85"
  Range: [0.0, 1.0]

KG_DEDUP_CUSTOM_ACRONYMS: "" (str) - Custom acronym mappings
  [FROM .env] Yes, via from_env() method
  Format: "ACRO1:expansion1,ACRO2:expansion2"
  Examples: "ISO:International Organization for Standardization"

KG_DEDUP_APOC_ENABLED: "true" (str, boolean) - Try APOC, fallback to Cypher
  [FROM .env] Yes, via from_env() method
  Values: "true" or "false"
  Notes: APOC ~10-20ms per 1000 entities, pure Cypher ~20-50ms
```

### Hardcoded Defaults (No .env Mapping Needed)

```
enabled: True (bool) - Controlled by KG_DEDUPLICATE_ENTITIES

exact_match_enabled: True (bool) - Layer 1 always enabled
  SHOULD BE IN .env: NO - always true when master enabled

embedding_model: "text-embedding-3-large" (str) - Model for Layer 2
  SHOULD BE IN .env: YES - as KG_DEDUP_EMBEDDING_MODEL

embedding_batch_size: 100 (int) - Batch size for embeddings
  SHOULD BE IN .env: YES - as KG_DEDUP_EMBEDDING_BATCH_SIZE

cache_embeddings: True (bool) - Cache Layer 2 embeddings
  SHOULD BE IN .env: YES - as KG_DEDUP_CACHE_EMBEDDINGS

create_uniqueness_constraints: True (bool) - Create Neo4j constraints
  SHOULD BE IN .env: YES - as KG_DEDUP_CREATE_CONSTRAINTS
```

---

## GraphStorageConfig

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/src/graph/config.py` (lines 281-306)

**Purpose:** Configuration for graph storage.

### Hardcoded Defaults (Should Consider for .env)

```
backend: GraphBackend.SIMPLE (enum) - Storage backend
  SHOULD BE IN .env: YES - as GRAPH_STORAGE_BACKEND
  Values: "simple" (JSON), "neo4j", "networkx"
  Notes: Neo4j recommended for production

simple_store_path: "./data/graphs/simple_graph.json" (str) - JSON path
  SHOULD BE IN .env: YES - as GRAPH_SIMPLE_STORE_PATH

export_json: True (bool) - Export to JSON after construction
  SHOULD BE IN .env: YES - as GRAPH_EXPORT_JSON

export_path: "./data/graphs/knowledge_graph.json" (str) - Export path
  SHOULD BE IN .env: YES - as GRAPH_EXPORT_PATH

deduplicate_entities: True (bool) - Merge duplicate entities
  SHOULD BE IN .env: YES - as GRAPH_DEDUPLICATE_ENTITIES

merge_similar_entities: False (bool) - Merge similar entities (expensive)
  SHOULD BE IN .env: YES - as GRAPH_MERGE_SIMILAR_ENTITIES

similarity_threshold: 0.9 (float) - For entity merging
  SHOULD BE IN .env: YES - as GRAPH_SIMILARITY_THRESHOLD

track_provenance: True (bool) - Track chunk sources for entities
  SHOULD BE IN .env: YES - as GRAPH_TRACK_PROVENANCE
```

---

## Summary: Parameters Currently in .env.example

**File:** `/Users/michalprusek/PycharmProjects/MY_SUJBOT/.env.example`

### Already Documented (17 variables)

1. ANTHROPIC_API_KEY
2. OPENAI_API_KEY
3. GOOGLE_API_KEY
4. LLM_MODEL
5. EMBEDDING_PROVIDER
6. EMBEDDING_MODEL
7. VOYAGE_API_KEY
8. KG_LLM_PROVIDER
9. KG_LLM_MODEL
10. KG_BACKEND
11. KG_EXPORT_PATH
12. KG_VERBOSE
13. NEO4J_URI
14. NEO4J_USERNAME
15. NEO4J_PASSWORD
16. NEO4J_DATABASE
17. AGENT_MODEL
18. AGENT_MAX_TOKENS (labeled but no default shown)
19. VECTOR_STORE_PATH
20. KNOWLEDGE_GRAPH_PATH
21. QUERY_EXPANSION_MODEL
22. ENABLE_PROMPT_CACHING
23. ENABLE_CONTEXT_MANAGEMENT
24. CONTEXT_MANAGEMENT_TRIGGER
25. CONTEXT_MANAGEMENT_KEEP
26. DATA_DIR
27. OUTPUT_DIR
28. CHUNK_SIZE
29. ENABLE_SAC
30. ENABLE_SMART_HIERARCHY
31. SUMMARY_MAX_CHARS
32. ENABLE_HYBRID_SEARCH
33. HYBRID_FUSION_K
34. ENABLE_KNOWLEDGE_GRAPH
35. KG_MIN_ENTITY_CONFIDENCE
36. KG_MIN_RELATIONSHIP_CONFIDENCE

---

## Summary: Parameters Missing from .env.example

**Total Parameters Found:** 128
**Already in .env.example:** 36
**Missing from .env.example:** 92

### Critical Parameters Missing

**Extraction (PHASE 1):** 11 missing
- ENABLE_OCR
- OCR_ENGINE
- OCR_RECOGNITION
- EXTRACT_TABLES
- TABLE_MODE
- EXTRACT_HIERARCHY
- HIERARCHY_TOLERANCE
- GENERATE_SUMMARIES
- SUMMARY_MODEL
- SUMMARY_STYLE
- LAYOUT_MODEL

**Summarization (PHASE 2):** 9 missing
- SPEED_MODE (controls use_batch_api)
- SUMMARY_TEMPERATURE
- SUMMARY_MAX_TOKENS
- SUMMARY_RETRY_ON_EXCEED
- SUMMARY_MAX_RETRIES
- SUMMARY_MAX_WORKERS
- SUMMARY_MIN_TEXT_LENGTH
- SUMMARY_ENABLE_BATCHING
- SUMMARY_BATCH_SIZE
- SUMMARY_USE_BATCH_API (controlled by SPEED_MODE)
- SUMMARY_BATCH_API_POLL_INTERVAL
- SUMMARY_BATCH_API_TIMEOUT

**Contextual Retrieval (PHASE 3A):** 10 missing
- ENABLE_CONTEXTUAL
- CONTEXT_GENERATION_TEMPERATURE
- CONTEXT_GENERATION_MAX_TOKENS
- CONTEXT_INCLUDE_SURROUNDING
- CONTEXT_NUM_SURROUNDING_CHUNKS
- CONTEXT_FALLBACK_TO_BASIC
- CONTEXT_BATCH_SIZE
- CONTEXT_MAX_WORKERS
- CONTEXT_USE_BATCH_API
- CONTEXT_BATCH_API_POLL_INTERVAL
- CONTEXT_BATCH_API_TIMEOUT

**Embedding (PHASE 4):** 3 missing
- EMBEDDING_BATCH_SIZE
- EMBEDDING_CACHE_ENABLED
- EMBEDDING_CACHE_SIZE

**Clustering (PHASE 4.5):** 5 missing
- CLUSTERING_MAX_CLUSTERS
- CLUSTERING_MIN_CLUSTERS
- CLUSTERING_LAYERS

**Agent (PHASE 7):** 9 missing
- AGENT_TEMPERATURE
- AGENT_ENABLE_TOOL_VALIDATION
- AGENT_DEBUG_MODE

**Tools:** 11 missing
- TOOL_DEFAULT_K
- TOOL_ENABLE_RERANKING
- TOOL_RERANKER_CANDIDATES
- TOOL_RERANKER_MODEL
- TOOL_ENABLE_GRAPH_BOOST
- TOOL_GRAPH_BOOST_WEIGHT
- TOOL_MAX_DOCUMENT_COMPARE
- TOOL_COMPLIANCE_THRESHOLD
- TOOL_CONTEXT_WINDOW
- TOOL_LAZY_LOAD_RERANKER
- TOOL_LAZY_LOAD_GRAPH
- TOOL_CACHE_EMBEDDINGS

**CLI:** 8 missing
- CLI_SHOW_CITATIONS
- CLI_CITATION_FORMAT
- CLI_SHOW_TOOL_CALLS
- CLI_SHOW_TIMING
- CLI_ENABLE_STREAMING
- CLI_SAVE_HISTORY
- CLI_HISTORY_FILE
- CLI_MAX_HISTORY_ITEMS

**Pipeline:** 1 missing
- LOG_LEVEL
- LOG_FILE

**Knowledge Graph:** 8 missing
- ENABLE_ENTITY_EXTRACTION
- ENABLE_RELATIONSHIP_EXTRACTION
- ENABLE_CROSS_DOCUMENT_RELATIONSHIPS
- KG_MAX_RETRIES
- KG_RETRY_DELAY
- KG_TIMEOUT
- KG_LOG_PATH

**Entity Extraction:** 8 missing
- ENTITY_EXTRACTION_LLM_PROVIDER
- ENTITY_EXTRACTION_LLM_MODEL
- ENTITY_EXTRACTION_TEMPERATURE
- ENTITY_EXTRACTION_MIN_CONFIDENCE
- ENTITY_EXTRACTION_EXTRACT_DEFINITIONS
- ENTITY_EXTRACTION_NORMALIZE
- ENTITY_EXTRACTION_BATCH_SIZE
- ENTITY_EXTRACTION_MAX_WORKERS
- ENTITY_EXTRACTION_CACHE_RESULTS
- ENTITY_EXTRACTION_INCLUDE_EXAMPLES
- ENTITY_EXTRACTION_MAX_PER_CHUNK
- ENTITY_EXTRACTION_ENABLED_TYPES

**Relationship Extraction:** 12 missing
- RELATIONSHIP_EXTRACTION_LLM_PROVIDER
- RELATIONSHIP_EXTRACTION_LLM_MODEL
- RELATIONSHIP_EXTRACTION_TEMPERATURE
- RELATIONSHIP_EXTRACTION_MIN_CONFIDENCE
- RELATIONSHIP_EXTRACTION_EXTRACT_EVIDENCE
- RELATIONSHIP_EXTRACTION_MAX_EVIDENCE_LENGTH
- RELATIONSHIP_EXTRACTION_WITHIN_CHUNK
- RELATIONSHIP_EXTRACTION_CROSS_CHUNK
- RELATIONSHIP_EXTRACTION_FROM_METADATA
- RELATIONSHIP_EXTRACTION_BATCH_SIZE
- RELATIONSHIP_EXTRACTION_MAX_WORKERS
- RELATIONSHIP_EXTRACTION_CACHE_RESULTS
- RELATIONSHIP_EXTRACTION_MAX_PER_ENTITY
- RELATIONSHIP_EXTRACTION_ENABLED_TYPES

**Neo4j:** 5 missing
- NEO4J_MAX_CONNECTION_LIFETIME
- NEO4J_MAX_CONNECTION_POOL_SIZE
- NEO4J_CONNECTION_TIMEOUT
- NEO4J_CREATE_INDEXES
- NEO4J_CREATE_CONSTRAINTS

**Entity Deduplication:** 8 missing
- KG_DEDUP_EMBEDDING_MODEL
- KG_DEDUP_EMBEDDING_BATCH_SIZE
- KG_DEDUP_CACHE_EMBEDDINGS
- KG_DEDUP_CREATE_CONSTRAINTS

**Graph Storage:** 6 missing
- GRAPH_STORAGE_BACKEND
- GRAPH_SIMPLE_STORE_PATH
- GRAPH_EXPORT_JSON
- GRAPH_EXPORT_PATH
- GRAPH_DEDUPLICATE_ENTITIES
- GRAPH_MERGE_SIMILAR_ENTITIES
- GRAPH_SIMILARITY_THRESHOLD
- GRAPH_TRACK_PROVENANCE

---

## Recommendations for .env.example Expansion

### Priority 1: Research-Critical Parameters (DO NOT EXPOSE)

These are research constraints backed by SOTA papers. Keep hardcoded, do NOT expose to .env:

- `CHUNK_SIZE` - Already exposed (good), but mark as CRITICAL
- `SUMMARY_STYLE` - Keep hardcoded to "generic" (research finding)
- `CHUNK_OVERLAP` - Keep hardcoded to 0 (RCTS architecture)
- `NORMALIZE_EMBEDDINGS` - Keep hardcoded to True (FAISS requirement)

### Priority 2: Essential For Users (HIGH PRIORITY)

Add these immediately to .env.example:

**Categorize into sections:**

1. **Required API Keys** (already there) ✓
2. **Model Selection** (partially there)
   - Add: LLM_PROVIDER (optional, auto-detected)
   - Add: EMBEDDING_PROVIDER (optional, auto-detected)

3. **Pipeline Control** (partially there)
   - Add: SPEED_MODE (fast|eco) - controls batch API, crucial for cost
   - Add: ENABLE_OCR (true|false)
   - Add: ENABLE_SMART_HIERARCHY (already there) ✓
   - Add: ENABLE_SAC (already there) ✓
   - Add: ENABLE_CONTEXTUAL (new - controls PHASE 3A)

4. **Performance Tuning** (missing)
   - Add: SUMMARY_MAX_WORKERS
   - Add: CONTEXT_MAX_WORKERS
   - Add: ENTITY_EXTRACTION_MAX_WORKERS
   - Add: EMBEDDING_BATCH_SIZE

5. **Neo4j Setup** (partially there)
   - Add: NEO4J_MAX_CONNECTION_LIFETIME
   - Add: NEO4J_MAX_CONNECTION_POOL_SIZE
   - Add: NEO4J_CREATE_INDEXES

6. **Entity Deduplication** (mostly missing)
   - Add: KG_DEDUPLICATE_ENTITIES (master switch)
   - Add: KG_DEDUP_USE_EMBEDDINGS
   - Add: KG_DEDUP_SIMILARITY_THRESHOLD
   - Add: KG_DEDUP_USE_ACRONYM_EXPANSION
   - Add: KG_DEDUP_ACRONYM_FUZZY_THRESHOLD
   - Add: KG_DEDUP_CUSTOM_ACRONYMS

7. **Agent Tools** (mostly missing)
   - Add: TOOL_DEFAULT_K
   - Add: TOOL_ENABLE_RERANKING
   - Add: TOOL_RERANKER_MODEL
   - Add: TOOL_ENABLE_GRAPH_BOOST

### Priority 3: Advanced Configuration (MEDIUM PRIORITY)

Add with explanatory comments:

- All SUMMARY_* parameters (temperature, max_tokens, retries, etc.)
- All CONTEXT_GENERATION_* parameters
- All CLUSTERING_* parameters
- All CLI_* parameters
- All ENTITY_EXTRACTION_* parameters
- All RELATIONSHIP_EXTRACTION_* parameters

### Priority 4: Internal Tuning (LOW PRIORITY)

Keep documented but not in .env.example (too specialized):

- Individual min_confidence thresholds
- Batch sizes for specific components
- Caching parameters
- Log file paths

---

## Action Items

1. **Expand .env.example** with all Priority 1-2 parameters
2. **Add ENV-ONLY section** documenting auto-detected parameters
3. **Add COMMENTED EXAMPLES** section for each phase
4. **Create PARAMETER_DEFAULTS.md** (this document) as reference
5. **Update CLAUDE.md** with updated parameter counts
6. **Update config.py docstrings** to reference .env variables

