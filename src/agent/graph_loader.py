"""
Shared Knowledge Graph Loading Logic

Provides consistent Neo4j/JSON loading for both CLI and WebApp with
proper error handling and fallback behavior.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


def load_knowledge_graph(
    kg_path: Path,
    vector_store: Optional[Any] = None,
    user_print: Optional[Callable[[str], None]] = None,
) -> Tuple[Any, Optional[Any]]:
    """
    Load knowledge graph from Neo4j or JSON with automatic fallback.

    Used by both CLI and WebApp backend for consistent KG loading.

    Args:
        kg_path: Path to knowledge graph files (directory or single file)
        vector_store: Optional vector store for GraphEnhancedRetriever (JSON mode only)
        user_print: Optional function to print user-facing messages (e.g., CLI print)

    Returns:
        Tuple of (knowledge_graph, graph_retriever or None)
        - If Neo4j: (GraphAdapter, None) - GraphAdapter doesn't use GraphEnhancedRetriever
        - If JSON: (KnowledgeGraph, GraphEnhancedRetriever or None)

    Raises:
        RuntimeError: If Neo4j explicitly requested but unavailable with config errors
        FileNotFoundError: If JSON fallback fails (no KG files found)
        ImportError: If required dependencies missing and can't recover
    """
    def _print(msg: str) -> None:
        """Print to user if callback provided."""
        if user_print:
            user_print(msg)

    # Check configured backend
    kg_backend = os.getenv("KG_BACKEND", "simple").lower()
    backend_explicit = kg_backend == "neo4j"  # User explicitly requested Neo4j

    knowledge_graph = None
    graph_retriever = None

    # Attempt Neo4j if configured
    if kg_backend == "neo4j":
        _print("   Using Neo4j backend...")
        logger.info("KG_BACKEND=neo4j - attempting Neo4j connection")

        try:
            from src.graph import Neo4jConfig
            from src.agent.graph_adapter import GraphAdapter
            from src.graph.exceptions import (
                Neo4jAuthenticationError,
                Neo4jConnectionError,
                Neo4jTimeoutError,
            )

            neo4j_config = Neo4jConfig.from_env()
            knowledge_graph = GraphAdapter.from_neo4j(neo4j_config)

            # Get stats for display
            entity_count = len(knowledge_graph.entities)
            rel_count = len(knowledge_graph.relationships)

            _print(f"   ✓ Connected to Neo4j: {entity_count} entities, {rel_count} relationships")
            logger.info(f"Neo4j connection successful: {entity_count} entities, {rel_count} relationships")

        except ImportError as e:
            # Missing neo4j package - permanent error
            error_msg = (
                f"neo4j package not installed but KG_BACKEND=neo4j is set.\n"
                f"   Install with: uv pip install neo4j\n"
                f"   OR change KG_BACKEND=simple in .env to use JSON"
            )

            if backend_explicit:
                # User explicitly requested Neo4j - fail fast
                logger.error(f"Neo4j package missing: {e}")
                _print(f"   ❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            else:
                # Not explicitly requested - can fallback
                logger.warning(f"Neo4j package missing, falling back to JSON: {e}")
                _print(f"   ⚠️  Neo4j package not installed, using JSON fallback")
                kg_backend = "simple"

        except (Neo4jAuthenticationError, KeyError, ValueError) as e:
            # Configuration errors - permanent, don't retry
            error_msg = str(e)
            if isinstance(e, Neo4jAuthenticationError) or "authentication" in error_msg.lower():
                guidance = "Check NEO4J_USERNAME and NEO4J_PASSWORD in .env"
            else:
                guidance = "Check Neo4j configuration in .env (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)"

            if backend_explicit:
                # User explicitly requested Neo4j - fail fast
                logger.error(f"Neo4j configuration error: {e}")
                _print(
                    f"   ❌ Neo4j configuration error: {e}\n"
                    f"   {guidance}\n"
                    f"   Cannot fallback when KG_BACKEND=neo4j is explicitly set."
                )
                raise RuntimeError(
                    f"Neo4j backend misconfigured. {guidance}\n"
                    f"Fix Neo4j config or change KG_BACKEND=simple in .env"
                ) from e
            else:
                # Not explicitly requested - can fallback
                logger.warning(f"Neo4j config error, falling back to JSON: {e}")
                _print(
                    f"   ⚠️  Neo4j configuration error: {e}\n"
                    f"   {guidance}\n"
                    f"   Falling back to JSON (some tools will be degraded)"
                )
                kg_backend = "simple"

        except Neo4jConnectionError as e:
            # Network/connection errors - could be transient
            error_msg = str(e)
            guidance = (
                "Check:\n"
                "   - NEO4J_URI is correct\n"
                "   - Neo4j server is running\n"
                "   - Network connectivity"
            )

            if backend_explicit:
                # User explicitly requested Neo4j - fail with guidance
                logger.error(f"Neo4j connection failed: {e}")
                _print(
                    f"   ❌ Cannot connect to Neo4j: {e}\n"
                    f"   {guidance}\n"
                    f"   Cannot fallback when KG_BACKEND=neo4j is explicitly set."
                )
                raise RuntimeError(
                    f"Neo4j connection failed: {e}\n{guidance}\n"
                    f"Fix connection or change KG_BACKEND=simple in .env"
                ) from e
            else:
                # Not explicitly requested - can fallback
                logger.warning(f"Neo4j connection failed, falling back to JSON: {e}")
                _print(
                    f"   ⚠️  Neo4j connection failed: {e}\n"
                    f"   Falling back to JSON (some tools will be degraded)"
                )
                kg_backend = "simple"

        except Neo4jTimeoutError as e:
            # Timeout - transient, could retry but fallback is safer
            if backend_explicit:
                logger.error(f"Neo4j timeout: {e}")
                _print(
                    f"   ❌ Neo4j timeout: {e}\n"
                    f"   Server may be overloaded or query too complex.\n"
                    f"   Cannot fallback when KG_BACKEND=neo4j is explicitly set."
                )
                raise RuntimeError(
                    f"Neo4j timeout: {e}\n"
                    f"Check server load or change KG_BACKEND=simple in .env"
                ) from e
            else:
                logger.warning(f"Neo4j timeout, falling back to JSON: {e}")
                _print(
                    f"   ⚠️  Neo4j timeout: {e}\n"
                    f"   Falling back to JSON (may be stale data)"
                )
                kg_backend = "simple"

        except Exception as e:
            # Truly unexpected - log full context
            logger.critical(f"Unexpected Neo4j error: {type(e).__name__}: {e}", exc_info=True)

            if backend_explicit:
                _print(
                    f"   ❌ Unexpected Neo4j error: {type(e).__name__}: {e}\n"
                    f"   Check logs for details.\n"
                    f"   Cannot fallback when KG_BACKEND=neo4j is explicitly set."
                )
                raise RuntimeError(
                    f"Unexpected Neo4j error: {e}\nCheck logs for details."
                ) from e
            else:
                _print(
                    f"   ⚠️  Unexpected Neo4j error: {type(e).__name__}: {e}\n"
                    f"   Falling back to JSON"
                )
                kg_backend = "simple"

    # Load from JSON if not using Neo4j or if fallback occurred
    if kg_backend != "neo4j":
        from src.graph_retrieval import GraphEnhancedRetriever
        from src.knowledge_graph import KnowledgeGraph

        _print("   Using JSON backend...")
        logger.info(f"Loading knowledge graph from JSON: {kg_path}")

        # Warn if user was expecting Neo4j but got JSON
        if backend_explicit and knowledge_graph is None:
            _print(
                "   ⚠️  WARNING: Running in degraded mode with JSON backend\n"
                "   Some tools will not work optimally:\n"
                "   - browse_entities: Unavailable (requires Neo4j indexed queries)\n"
                "   - multi_hop_search: Slower (no graph database optimization)\n"
                "   - graph_search: Limited to JSON export data"
            )

        # Check if path is a directory or single file
        if kg_path.is_dir():
            # Prefer unified_kg.json if it exists
            unified_kg_path = kg_path / "unified_kg.json"

            if unified_kg_path.exists():
                # Load unified KG (already deduplicated)
                _print(f"   Loading unified knowledge graph from JSON...")
                knowledge_graph = KnowledgeGraph.load_json(str(unified_kg_path))
                _print(
                    f"   Unified KG: {len(knowledge_graph.entities)} entities, "
                    f"{len(knowledge_graph.relationships)} relationships"
                )
                logger.info(
                    f"Loaded unified_kg.json: {len(knowledge_graph.entities)} entities, "
                    f"{len(knowledge_graph.relationships)} relationships"
                )
            else:
                # Fallback: Load all *_kg.json files (old behavior)
                kg_files = sorted(kg_path.glob("*_kg.json"))
                if not kg_files:
                    error_msg = f"No knowledge graph files (*_kg.json or unified_kg.json) found in {kg_path}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)

                _print(f"   Found {len(kg_files)} knowledge graph files (unified_kg.json not found)")
                logger.info(f"Found {len(kg_files)} KG files (no unified_kg.json)")

                # Load and merge all KG files (naive merge without deduplication)
                knowledge_graph = None
                total_entities = 0
                total_relationships = 0

                for kg_file in kg_files:
                    kg = KnowledgeGraph.load_json(str(kg_file))
                    if knowledge_graph is None:
                        knowledge_graph = kg
                    else:
                        # Merge graphs by combining entities and relationships
                        knowledge_graph.entities.extend(kg.entities)
                        knowledge_graph.relationships.extend(kg.relationships)

                    total_entities += len(kg.entities)
                    total_relationships += len(kg.relationships)
                    logger.debug(
                        f"Loaded {kg_file.name}: {len(kg.entities)} entities, "
                        f"{len(kg.relationships)} relationships"
                    )

                _print(
                    f"   Total: {total_entities} entities, "
                    f"{total_relationships} relationships (naive merge)"
                )
                logger.info(f"Merged {len(kg_files)} KG files: {total_entities} entities, {total_relationships} relationships")
                logger.warning(
                    "Naive merge performed (no deduplication) - may contain duplicate entities.\n"
                    "For better performance, build unified_kg.json:\n"
                    "  uv run python scripts/merge_knowledge_graphs.py vector_db/"
                )
        else:
            # Load single file
            knowledge_graph = KnowledgeGraph.load_json(str(kg_path))
            _print(
                f"   Entities: {len(knowledge_graph.entities)}, "
                f"Relationships: {len(knowledge_graph.relationships)}"
            )
            logger.info(
                f"Loaded single KG file: {len(knowledge_graph.entities)} entities, "
                f"{len(knowledge_graph.relationships)} relationships"
            )

        # Create graph retriever (only for JSON backend)
        # Note: Neo4j GraphAdapter doesn't use GraphEnhancedRetriever
        # (it has its own graph query methods)
        if knowledge_graph and vector_store:
            graph_retriever = GraphEnhancedRetriever(
                vector_store=vector_store,
                knowledge_graph=knowledge_graph
            )
            logger.info("Created GraphEnhancedRetriever")

    return knowledge_graph, graph_retriever
