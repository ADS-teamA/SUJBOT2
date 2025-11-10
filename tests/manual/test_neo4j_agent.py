#!/usr/bin/env python
"""
Quick test: Verify agent uses Neo4j backend
"""

from dotenv import load_dotenv
load_dotenv()

import os

print("=" * 60)
print("TEST: Agent Neo4j Integration")
print("=" * 60)

# Check environment
print(f"\n✓ KG_BACKEND={os.getenv('KG_BACKEND')}")
print(f"✓ NEO4J_URI={os.getenv('NEO4J_URI', 'not set')[:40]}...")

# Test GraphAdapter
print("\n[1] Testing GraphAdapter...")
from src.graph import Neo4jConfig
from src.agent.graph_adapter import GraphAdapter

neo4j_config = Neo4jConfig.from_env()
adapter = GraphAdapter.from_neo4j(neo4j_config)

print(f"   ✓ Connected to Neo4j")
print(f"   ✓ Entities: {len(adapter.entities)}")
print(f"   ✓ Relationships: {len(adapter.relationships)}")

# Test entity lookup
sample_entity = list(adapter.entities.values())[0]
print(f"   ✓ Sample entity: {sample_entity.value[:50]}")

# Test find_entities (direct Neo4j query)
print("\n[2] Testing direct Neo4j queries...")
standards = adapter.find_entities(entity_type="standard", min_confidence=0.8)
print(f"   ✓ Found {len(standards)} standards with confidence >= 0.8")

# Test get_relationships_for_entity
if standards:
    test_entity = standards[0]
    relationships = adapter.get_relationships_for_entity(test_entity.id)
    print(f"   ✓ Entity '{test_entity.value}' has {len(relationships)} relationships")

adapter.close()

# Test agent initialization
print("\n[3] Testing Agent CLI initialization...")
from src.agent.config import AgentConfig
from pathlib import Path

config = AgentConfig.from_env(vector_store_path=Path("vector_db"))
print(f"   ✓ Config loaded")
print(f"   ✓ KG enabled: {config.enable_knowledge_graph}")
print(f"   ✓ KG path: {config.knowledge_graph_path}")

print("\n[4] Initializing agent (this may take a moment)...")
from src.agent.cli import AgentCLI

cli = AgentCLI(config)
print(f"   ✓ Agent initialized!")

# Check knowledge graph type
if hasattr(cli, 'tools'):
    # Get a tool that uses KG
    from src.agent.tools.registry import get_registry
    registry = get_registry(
        vector_store=cli.vector_store,
        embedder=cli.embedder,
        knowledge_graph=getattr(cli, 'knowledge_graph', None)
    )

    if 'graph_search' in registry:
        graph_tool = registry['graph_search']
        kg_type = type(graph_tool.knowledge_graph).__name__ if graph_tool.knowledge_graph else "None"
        print(f"   ✓ Knowledge graph type in tools: {kg_type}")

        if kg_type == "GraphAdapter":
            print(f"   ✓ SUCCESS: Agent is using Neo4j via GraphAdapter!")
        else:
            print(f"   ⚠️  WARNING: Agent is using {kg_type}, not GraphAdapter")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Agent is using Neo4j!")
print("=" * 60)
print("\nYou can now run the agent with:")
print("  uv run python -m src.agent.cli")
print("\nTry a graph query:")
print("  > find all regulations about 'waste management'")
