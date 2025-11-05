#!/usr/bin/env python3
"""
Configuration Display Script

Displays complete configuration for all pipeline tasks:
- PHASE 1: Extraction (Docling)
- PHASE 2: Summarization (LLM)
- PHASE 3: Contextual Retrieval (LLM)
- PHASE 4: Embeddings
- PHASE 5A: Knowledge Graph (LLM)
- PHASE 7: RAG Agent (LLM)

Shows:
- Model selections from .env
- API key status
- Research-backed parameters from config.py
"""

import os
import sys
from pathlib import Path

# Load .env file manually
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

# Add src to path and import config directly to avoid docling import
sys.path.insert(0, str(Path(__file__).parent))

# Import config module directly (without src.__init__ which imports docling)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "config",
    Path(__file__).parent / "src" / "config.py"
)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

# Extract classes from module
ModelConfig = config_module.ModelConfig
ExtractionConfig = config_module.ExtractionConfig
SummarizationConfig = config_module.SummarizationConfig
ContextGenerationConfig = config_module.ContextGenerationConfig
ChunkingConfig = config_module.ChunkingConfig
EmbeddingConfig = config_module.EmbeddingConfig
get_default_config = config_module.get_default_config


def check_api_key(key_name: str) -> str:
    """Check if API key is set and show status."""
    key = os.getenv(key_name)
    if not key:
        return "❌ NOT SET"
    elif len(key) < 20:
        return "⚠️  INVALID (too short)"
    else:
        # Show first 10 chars and last 4 chars
        masked = f"{key[:10]}...{key[-4:]}"
        return f"✅ {masked}"


def display_separator(title: str):
    """Display section separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def display_config():
    """Display complete configuration."""
    # Load configs
    model_config = ModelConfig.from_env()
    full_config = get_default_config()

    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MY_SUJBOT CONFIGURATION" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")

    # ========================================================================
    # API KEYS
    # ========================================================================
    display_separator("🔑 API KEYS")
    print(f"\n  ANTHROPIC_API_KEY:  {check_api_key('ANTHROPIC_API_KEY')}")
    print(f"  OPENAI_API_KEY:     {check_api_key('OPENAI_API_KEY')}")
    print(f"  VOYAGE_API_KEY:     {check_api_key('VOYAGE_API_KEY')}")

    # ========================================================================
    # PHASE 1: EXTRACTION (Docling)
    # ========================================================================
    display_separator("📄 PHASE 1: DOCUMENT EXTRACTION (Docling)")
    extract_config = full_config.extraction

    print(f"\n  Layout Model:       {extract_config.layout_model}")
    print(f"  OCR Enabled:        {extract_config.enable_ocr}")
    print(f"  OCR Languages:      {', '.join(extract_config.ocr_language)}")
    print(f"  OCR Mode:           {extract_config.ocr_recognition}")
    print(f"  Smart Hierarchy:    {extract_config.enable_smart_hierarchy}")
    print(f"  Hierarchy Tolerance: {extract_config.hierarchy_tolerance}")
    print(f"  Extract Tables:     {extract_config.extract_tables}")
    print(f"  Table Mode:         {extract_config.table_mode}")

    # ========================================================================
    # PHASE 2: SUMMARIZATION (LLM)
    # ========================================================================
    display_separator("✍️  PHASE 2: SUMMARIZATION (Generic Summaries)")
    summary_config = full_config.summarization

    print(f"\n  Provider:           {summary_config.provider}")
    print(f"  Model:              {summary_config.model}")
    print(f"  Style:              {summary_config.style} (research optimal)")
    print(f"  Max Chars:          {summary_config.max_chars} (research optimal)")
    print(f"  Temperature:        {summary_config.temperature}")
    print(f"  Max Tokens:         {summary_config.max_tokens}")
    print(f"  Max Workers:        {summary_config.max_workers} (parallel)")

    # Check if API key is available
    if summary_config.provider == "anthropic":
        key_status = check_api_key("ANTHROPIC_API_KEY")
    else:
        key_status = check_api_key("OPENAI_API_KEY")
    print(f"  API Key:            {key_status}")

    # ========================================================================
    # PHASE 3: CHUNKING + CONTEXTUAL RETRIEVAL
    # ========================================================================
    display_separator("🔪 PHASE 3: CHUNKING + CONTEXTUAL RETRIEVAL")
    chunk_config = full_config.chunking

    print(f"\n  Max Tokens:         {chunk_config.max_tokens} tokens (HybridChunker)")
    print(f"  Tokenizer Model:    {chunk_config.tokenizer_model}")
    print(f"  Enable Contextual:  {chunk_config.enable_contextual}")
    print(f"  Enable Multi-Layer: {chunk_config.enable_multi_layer}")

    if chunk_config.enable_contextual and chunk_config.context_config:
        context_config = chunk_config.context_config
        print(f"\n  Context Generation:")
        print(f"    Provider:         {context_config.provider}")
        print(f"    Model:            {context_config.model}")
        print(f"    Temperature:      {context_config.temperature}")
        print(f"    Max Tokens:       {context_config.max_tokens}")
        print(f"    Batch Size:       {context_config.batch_size}")
        print(f"    Max Workers:      {context_config.max_workers}")

        # Check API key
        if context_config.provider == "anthropic":
            key_status = check_api_key("ANTHROPIC_API_KEY")
        else:
            key_status = check_api_key("OPENAI_API_KEY")
        print(f"    API Key:          {key_status}")

    # ========================================================================
    # PHASE 4: EMBEDDINGS
    # ========================================================================
    display_separator("🔢 PHASE 4: EMBEDDINGS")
    embed_config = full_config.embedding

    print(f"\n  Provider:           {embed_config.provider}")
    print(f"  Model:              {embed_config.model}")
    print(f"  Batch Size:         {embed_config.batch_size}")
    print(f"  Enable Multi-Layer: {embed_config.enable_multi_layer}")

    # Check API key based on provider
    if embed_config.provider == "openai":
        key_status = check_api_key("OPENAI_API_KEY")
        print(f"  API Key:            {key_status}")
    elif embed_config.provider == "voyage":
        key_status = check_api_key("VOYAGE_API_KEY")
        print(f"  API Key:            {key_status}")
    elif embed_config.provider == "huggingface":
        print(f"  API Key:            ✅ NOT NEEDED (local model)")
    else:
        print(f"  API Key:            ⚠️  Unknown provider")

    # ========================================================================
    # PHASE 5A: KNOWLEDGE GRAPH (Optional)
    # ========================================================================
    display_separator("🕸️  PHASE 5A: KNOWLEDGE GRAPH (Optional)")

    kg_provider = os.getenv("KG_LLM_PROVIDER", "Not set")
    kg_model = os.getenv("KG_LLM_MODEL", "Not set")
    kg_backend = os.getenv("KG_BACKEND", "simple")
    kg_enabled = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "false").lower() == "true"

    print(f"\n  Enabled:            {kg_enabled}")
    print(f"  Provider:           {kg_provider}")
    print(f"  Model:              {kg_model}")
    print(f"  Backend:            {kg_backend}")

    if kg_provider != "Not set":
        if kg_provider == "anthropic":
            key_status = check_api_key("ANTHROPIC_API_KEY")
        elif kg_provider == "openai":
            key_status = check_api_key("OPENAI_API_KEY")
        else:
            key_status = "⚠️  Unknown provider"
        print(f"  API Key:            {key_status}")

    # ========================================================================
    # PHASE 5B: HYBRID SEARCH
    # ========================================================================
    display_separator("🔍 PHASE 5B: HYBRID SEARCH")
    hybrid_enabled = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
    hybrid_k = os.getenv("HYBRID_FUSION_K", "60")

    print(f"\n  Enabled:            {hybrid_enabled}")
    print(f"  RRF Fusion K:       {hybrid_k} (research optimal)")
    print(f"  Features:           BM25 + Dense + RRF fusion")

    # ========================================================================
    # PHASE 7: RAG AGENT (Optional)
    # ========================================================================
    display_separator("🤖 PHASE 7: RAG AGENT")

    agent_model = os.getenv("AGENT_MODEL", model_config.llm_model)
    enable_hyde = os.getenv("ENABLE_HYDE", "false").lower() == "true"
    enable_decomp = os.getenv("ENABLE_DECOMPOSITION", "false").lower() == "true"

    print(f"\n  Model:              {agent_model}")
    print(f"  Enable HyDE:        {enable_hyde}")
    print(f"  Enable Decomposition: {enable_decomp}")
    print(f"  API Key:            {check_api_key('ANTHROPIC_API_KEY')}")
    print(f"  Vector Store:       {os.getenv('VECTOR_STORE_PATH', 'output/hybrid_store')}")

    # ========================================================================
    # COST TRACKING
    # ========================================================================
    display_separator("💰 COST TRACKING")

    print(f"\n  Status:             ✅ ENABLED")
    print(f"  Tracked Operations: summary, context, embedding, kg_extraction, agent")
    print(f"  Display:            Automatic at end of indexing")
    print(f"\n  Pricing (2025):")
    print(f"    Anthropic Haiku:  $1.00/$5.00 per 1M tokens (input/output)")
    print(f"    Anthropic Sonnet: $3.00/$15.00 per 1M tokens (input/output)")
    print(f"    OpenAI GPT-5:     $1.25/$10.00 per 1M tokens (input/output)")
    print(f"    OpenAI Embeddings: $0.13 per 1M tokens")
    print(f"    Voyage AI:        $0.06 per 1M tokens")
    print(f"    HuggingFace:      $0.00 (FREE local)")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    display_separator("📊 CONFIGURATION SUMMARY")

    # Count enabled features
    features = []
    if extract_config.enable_smart_hierarchy:
        features.append("Smart Hierarchy")
    if summary_config.style == "generic":
        features.append("Generic Summaries")
    if chunk_config.enable_contextual:
        features.append("Contextual Retrieval")
    if embed_config.enable_multi_layer:
        features.append("Multi-Layer Indexing")
    if kg_enabled:
        features.append("Knowledge Graph")
    if hybrid_enabled:
        features.append("Hybrid Search")

    print(f"\n  Active Features:    {len(features)}")
    for feature in features:
        print(f"    ✓ {feature}")

    # Check if configuration is valid
    print(f"\n  Configuration:      ", end="")
    has_llm_key = (
        os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    has_embed_key = (
        embed_config.provider == "huggingface" or
        os.getenv("OPENAI_API_KEY") or
        os.getenv("VOYAGE_API_KEY")
    )

    if has_llm_key and has_embed_key:
        print("✅ VALID (ready to run)")
    else:
        print("⚠️  INCOMPLETE")
        if not has_llm_key:
            print("    ❌ Missing LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        if not has_embed_key:
            print("    ❌ Missing embedding API key or config")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        display_config()
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
