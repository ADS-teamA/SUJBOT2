#!/usr/bin/env python3
"""Direct test of graph tools after .values() fix."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.graph.models import KnowledgeGraph, Entity
from src.graph.neo4j_manager import Neo4jManager
from src.agent.tools.tier2_advanced import BrowseEntitiesInput, BrowseEntitiesTool, GraphSearchInput, GraphSearchTool
from src.faiss_vector_store import FAISSVectorStore
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_graph_tools():
    """Test graph tools directly."""

    # Load config
    config = get_config()

    # Load knowledge graph from Neo4j
    logger.info("Loading knowledge graph from Neo4j...")
    neo4j_manager = Neo4jManager(config=config.neo4j)

    # Load entities
    query = "MATCH (e:Entity) RETURN e"
    result = neo4j_manager.execute_read(query)

    entities = []
    for record in result:
        node = record["e"]
        source_chunk_ids = node.get("source_chunk_ids", [])
        if not isinstance(source_chunk_ids, list):
            source_chunk_ids = []

        entities.append(Entity(
            id=node.get("id", ""),
            type=node.get("type", ""),
            value=node.get("value", ""),
            normalized_value=node.get("normalized_value", node.get("value", "")),
            confidence=node.get("confidence", 1.0),
            source_chunk_ids=set(source_chunk_ids),
            document_id=node.get("document_id", ""),
            first_mention_chunk_id=node.get("first_mention_chunk_id"),
            extraction_method=node.get("extraction_method"),
        ))

    knowledge_graph = KnowledgeGraph(
        source_document_id="neo4j_unified",
        created_at=datetime.now(),
    )
    knowledge_graph.entities = entities

    logger.info(f"✓ Loaded {len(entities)} entities")

    # Load vector store (needed for GraphSearch)
    vector_store = FAISSVectorStore(config.vector_store_path)
    logger.info(f"✓ Loaded vector store: {vector_store.get_stats()}")

    # Test 1: BrowseEntities
    logger.info("\n" + "="*60)
    logger.info("TEST 1: BrowseEntities")
    logger.info("="*60)

    browse_tool = BrowseEntitiesTool(knowledge_graph=knowledge_graph, vector_store=None)

    # Test browsing organization entities
    input1 = BrowseEntitiesInput(entity_type="organization", limit=5)
    logger.info(f"Input: {input1}")

    result1 = browse_tool.execute_impl(**input1.dict())
    logger.info(f"Success: {result1.success}")
    if result1.success:
        logger.info(f"Found {len(result1.data.get('entities', []))} entities")
        logger.info(f"Sample entities: {result1.data.get('entities', [])[:3]}")
    else:
        logger.error(f"Error: {result1.error}")
        logger.error(f"Metadata: {result1.metadata}")

    # Test 2: GraphSearch
    logger.info("\n" + "="*60)
    logger.info("TEST 2: GraphSearch")
    logger.info("="*60)

    search_tool = GraphSearchTool(knowledge_graph=knowledge_graph, vector_store=vector_store)

    input2 = GraphSearchInput(query="jaderná bezpečnost", search_mode="semantic", limit=5)
    logger.info(f"Input: {input2}")

    result2 = search_tool.execute_impl(**input2.dict())
    logger.info(f"Success: {result2.success}")
    if result2.success:
        logger.info(f"Found {len(result2.data.get('results', []))} results")
        logger.info(f"Sample results: {result2.data.get('results', [])[:3]}")
    else:
        logger.error(f"Error: {result2.error}")
        logger.error(f"Metadata: {result2.metadata}")

    neo4j_manager.close()
    logger.info("\n✓ All tests completed")

if __name__ == "__main__":
    test_graph_tools()
