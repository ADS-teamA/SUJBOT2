#!/usr/bin/env python3
"""
Test script for HyDE and Query Expansion

Demonstrates how HyDE and Query Expander process queries.
Shows the same output that agents see internally.
Generates JSON file with exact search tool output for agent inspection.

Usage:
    python scripts/test_hyde_query_expansion.py

Requirements:
    - config.json with query_expansion_provider and query_expansion_model
    - ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable
    - Indexed documents (for search tool output generation)

Output:
    - test_hyde_output.json: Exact ToolResult that agent receives from search tool
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agent.query_expander import QueryExpander
from src.agent.hyde_generator import HyDEGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.json."""
    config_path = project_root / "config.json"

    if not config_path.exists():
        logger.error(f"config.json not found at {config_path}")
        logger.info("Copy config.json.example to config.json and add your API keys")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    return config


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """Main test function."""
    # Test query
    query = "voda v reaktoru"

    print_section_header("HyDE and Query Expansion Test")
    print(f"Original Query: '{query}'")
    print(f"Project Root: {project_root}")

    # Load config
    logger.info("Loading configuration...")
    config = load_config()

    # Get settings from config
    provider = config.get("models", {}).get("llm_provider")
    model = config.get("models", {}).get("llm_model", "gpt-4o-mini")
    hyde_num_hypotheses = config.get("agent_tools", {}).get("hyde_num_hypotheses", 3)

    # Fallback to query_expansion settings if llm_provider not set
    if not provider:
        provider = "openai"  # Default
        logger.warning(f"llm_provider not set in config, using default: {provider}")

    logger.info(f"Using provider: {provider}, model: {model}")
    logger.info(f"HyDE num_hypotheses: {hyde_num_hypotheses}")

    # Get API keys from environment
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_key and not openai_key:
        logger.error("No API keys found in environment!")
        logger.info("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        sys.exit(1)

    # ========================================================================
    # PART 1: Query Expansion
    # ========================================================================
    print_section_header("PART 1: Query Expansion")

    try:
        # Initialize QueryExpander
        logger.info("Initializing QueryExpander...")
        expander = QueryExpander(
            provider=provider,
            model=model,
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
        )

        # Expand query with 2 variations
        num_expansions = 2
        logger.info(f"Expanding query with {num_expansions} variations...")

        expansion_result = expander.expand(query, num_expansions=num_expansions)

        # Print results
        print(f"Expansion Method: {expansion_result.expansion_method}")
        print(f"Model Used: {expansion_result.model_used}")
        print(f"Number of Queries: {len(expansion_result.expanded_queries)}")
        print(f"\nExpanded Queries:")
        for idx, q in enumerate(expansion_result.expanded_queries, 1):
            marker = " (original)" if q == query else ""
            print(f"  {idx}. {q}{marker}")

    except Exception as e:
        logger.error(f"Query expansion failed: {e}", exc_info=True)
        print(f"\n❌ Query Expansion FAILED: {e}")

    # ========================================================================
    # PART 2: HyDE (Hypothetical Document Embeddings)
    # ========================================================================
    print_section_header("PART 2: HyDE (Hypothetical Document Embeddings)")

    try:
        # Initialize HyDEGenerator
        logger.info("Initializing HyDEGenerator...")
        hyde_gen = HyDEGenerator(
            provider=provider,
            model=model,
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            num_hypotheses=hyde_num_hypotheses,
        )

        # Generate hypothetical documents
        logger.info(f"Generating {hyde_num_hypotheses} hypothetical documents...")

        hyde_result = hyde_gen.generate(query, num_docs=hyde_num_hypotheses)

        # Print results
        print(f"Generation Method: {hyde_result.generation_method}")
        print(f"Model Used: {hyde_result.model_used}")
        print(f"Number of Hypothetical Docs: {len(hyde_result.hypothetical_docs)}")
        print(f"\nHypothetical Documents:")

        for idx, doc in enumerate(hyde_result.hypothetical_docs, 1):
            print(f"\n  Document {idx}:")
            print(f"  {'-'*76}")
            # Print document with indentation
            for line in doc.split('\n'):
                print(f"  {line}")

        if not hyde_result.hypothetical_docs:
            print("  (No hypothetical documents generated)")

    except Exception as e:
        logger.error(f"HyDE generation failed: {e}", exc_info=True)
        print(f"\n❌ HyDE Generation FAILED: {e}")

    # ========================================================================
    # PART 3: How Agent Uses These Results
    # ========================================================================
    print_section_header("PART 3: How Agent Uses These Results")

    print("Agent Processing Flow:")
    print("\n1. QUERY EXPANSION:")
    print("   - Original query: 'voda v reaktoru'")
    print("   - Generates 2 paraphrases with different wording")
    print("   - Result: 3 queries total (original + 2 variations)")
    print("   - All 3 queries are searched in vector store")
    print("   - Results are merged using RRF (Reciprocal Rank Fusion)")

    print("\n2. HyDE (Multi-Hypothesis Averaging):")
    print(f"   - Generates {hyde_num_hypotheses} hypothetical answer documents")
    print("   - Each document is embedded separately")
    print("   - Embeddings are averaged: np.mean(embeddings, axis=0)")
    print("   - Averaged embedding is used for ORIGINAL query only")
    print("   - Paraphrases use standard embeddings (efficiency)")

    print("\n3. SEARCH EXECUTION:")
    print("   - Query 1 (original): Uses averaged HyDE embedding")
    print("   - Query 2 (paraphrase 1): Uses standard embedding")
    print("   - Query 3 (paraphrase 2): Uses standard embedding")
    print("   - All results are merged and reranked")

    print("\n4. EXPECTED IMPROVEMENT:")
    print("   - Query Expansion: +15-25% recall improvement")
    print("   - HyDE: +15-30% recall improvement (zero-shot)")
    print("   - Combined: Significant improvement for domain-specific queries")

    # ========================================================================
    # Summary
    # ========================================================================
    print_section_header("Summary")

    print("✅ Query Expansion: Generates multiple phrasings of the same question")
    print("✅ HyDE: Generates hypothetical answers to bridge vocabulary gap")
    print("✅ Both techniques improve retrieval for legal/technical documents")
    print("\nNext Steps:")
    print("  - Run full search with: use_hyde=True, num_expands=2")
    print("  - Compare results with and without these features")
    print("  - Measure recall improvement on your specific documents")

    # ========================================================================
    # PART 4: Generate JSON Output with Search Tool Result
    # ========================================================================
    print_section_header("PART 4: Generating JSON Output (Search Tool Result)")

    async def run_search_tool():
        """Run search tool with HyDE and query expansion."""
        try:
            logger.info("Attempting to run search tool with HyDE and query expansion...")

            # Import search tool dependencies
            from src.agent.tools.tier1_basic import SearchTool
            from src.config import get_config, EmbeddingConfig
            from src.storage import load_vector_store_adapter
            from src.embedding_generator import EmbeddingGenerator
            from src.reranker import CrossEncoderReranker

            # Load full config
            full_config = get_config()

            # Get storage backend
            storage_dict = full_config.storage if hasattr(full_config.storage, '__dict__') else full_config.storage
            if isinstance(storage_dict, dict):
                backend = storage_dict.get("backend", "postgresql")
            else:
                backend = storage_dict.backend

            logger.info(f"Initializing vector store (backend: {backend})...")

            # Load vector store adapter
            if backend == "postgresql":
                # Get PostgreSQL config
                if hasattr(full_config, 'postgresql'):
                    pg_config = full_config.postgresql if isinstance(full_config.postgresql, dict) else full_config.postgresql.__dict__
                else:
                    pg_config = {"connection_string_env": "DATABASE_URL", "pool_size": 20, "dimensions": 3072}

                connection_string = os.getenv(pg_config.get("connection_string_env", "DATABASE_URL"))
                if not connection_string:
                    print("⚠️  PostgreSQL connection string not found.")
                    print("    Set DATABASE_URL in .env file.")
                    return None

                vector_store = await load_vector_store_adapter(
                    backend="postgresql",
                    connection_string=connection_string,
                    pool_size=pg_config.get("pool_size", 20),
                    dimensions=pg_config.get("dimensions", 3072),
                )
            else:  # FAISS
                vector_store_path = Path(full_config.vector_store_path)
                if not vector_store_path.exists():
                    print(f"⚠️  Vector store not found at {vector_store_path}")
                    print("    Run: uv run python run_pipeline.py <document.pdf>")
                    return None

                vector_store = await load_vector_store_adapter(
                    backend="faiss",
                    path=str(vector_store_path)
                )

            logger.info("Vector store loaded successfully")

            # Initialize embedder
            models_dict = full_config.models if isinstance(full_config.models, dict) else full_config.models.__dict__
            embedding_config = EmbeddingConfig(
                provider=models_dict.get("embedding_provider", "openai"),
                model=models_dict.get("embedding_model", "text-embedding-3-large"),
                batch_size=64,
                normalize=True,
                enable_multi_layer=True,
                cache_enabled=True,
                cache_max_size=1000
            )
            embedder = EmbeddingGenerator(embedding_config)

            # Initialize reranker (optional)
            agent_tools_dict = full_config.agent_tools if isinstance(full_config.agent_tools, dict) else full_config.agent_tools.__dict__
            reranker = None
            if agent_tools_dict.get("enable_reranking", False):
                try:
                    reranker = CrossEncoderReranker()
                    logger.info("Reranker initialized")
                except Exception as e:
                    logger.warning(f"Reranker unavailable: {e}")

            # Create SearchTool instance - wrap agent_tools in object if it's a dict
            logger.info("Creating SearchTool instance...")

            # Create config object from dict - add query expansion fields
            if isinstance(agent_tools_dict, dict):
                from types import SimpleNamespace
                # Add query_expansion fields from agent or models config
                agent_dict = full_config.agent if isinstance(full_config.agent, dict) else full_config.agent.__dict__
                config_dict = {
                    **agent_tools_dict,
                    "query_expansion_provider": models_dict.get("llm_provider", "openai"),
                    "query_expansion_model": agent_dict.get("query_expansion_model", models_dict.get("llm_model", "gpt-4o-mini")),
                }
                tool_config = SimpleNamespace(**config_dict)
            else:
                tool_config = agent_tools_dict

            search_tool = SearchTool(
                vector_store=vector_store,
                embedder=embedder,
                reranker=reranker,
                graph_retriever=None,
                knowledge_graph=None,
                context_assembler=None,
                llm_provider=None,
                config=tool_config,
            )

            # Execute search with HyDE and query expansion
            logger.info(f"Running search with HyDE + query expansion for: '{query}'")
            result = search_tool.execute(
                query=query,
                k=5,
                num_expands=2,  # Original + 2 paraphrases
                use_hyde=True,  # Enable HyDE
                enable_graph_boost=False,
            )

            # Convert ToolResult to dict
            result_dict = {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "metadata": result.metadata,
                "citations": result.citations,
                "execution_time_ms": result.execution_time_ms,
                "estimated_tokens": result.estimated_tokens,
                "api_cost_usd": result.api_cost_usd,
            }

            # Add timestamp and query info
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "tool": "search",
                "parameters": {
                    "k": 5,
                    "num_expands": 2,
                    "use_hyde": True,
                    "enable_graph_boost": False,
                },
                "tool_result": result_dict,
            }

            # Save to JSON file
            output_file = project_root / "test_hyde_output.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"✅ Search tool result saved to: {output_file}")
            print(f"\nResult Summary:")
            print(f"  - Success: {result.success}")
            print(f"  - Chunks retrieved: {len(result.data) if result.data else 0}")
            print(f"  - Execution time: {result.execution_time_ms:.0f}ms")
            print(f"  - Queries used: {result.metadata.get('queries_used', [])}")
            print(f"  - HyDE hypotheses: {result.metadata.get('hyde_metadata', {}).get('num_hypotheses', 0)}")
            print(f"  - Expansion method: {result.metadata.get('expansion_metadata', {}).get('expansion_method', 'none')}")

            # Print citations preview
            if result.citations:
                print(f"\nCitations Preview (first 3):")
                for citation in result.citations[:3]:
                    print(f"  {citation}")
                if len(result.citations) > 3:
                    print(f"  ... and {len(result.citations) - 3} more")

            return output_data

        except ImportError as e:
            logger.warning(f"Could not import search tool dependencies: {e}")
            print(f"\n⚠️  Search tool not available (missing dependencies: {e})")
            print("    Install dependencies to generate JSON output.")
            return None
        except Exception as e:
            logger.error(f"Search tool execution failed: {e}", exc_info=True)
            print(f"\n❌ Search tool execution FAILED: {e}")
            print("    Make sure you have indexed documents before running this script.")
            return None

    # Run async search tool
    try:
        asyncio.run(run_search_tool())
    except Exception as e:
        logger.error(f"Failed to run search tool: {e}", exc_info=True)
        print(f"\n❌ Could not run search tool: {e}")


if __name__ == "__main__":
    main()
