#!/usr/bin/env python
"""
Quick test: Verify browse_entities tool is registered
"""

from dotenv import load_dotenv
load_dotenv()

from src.agent.tools.registry import get_registry
from src.agent.graph_adapter import GraphAdapter
from src.graph import Neo4jConfig
from src.hybrid_search import HybridVectorStore
from src.embedding_generator import EmbeddingGenerator
from pathlib import Path

print("=" * 60)
print("TEST: browse_entities Tool Registration")
print("=" * 60)

# Load vector store
print("\n[1] Loading vector store...")
vector_store = HybridVectorStore.load(Path("vector_db"))
print(f"   ✓ Vector store loaded")

# Load embedder
print("\n[2] Loading embedder...")
embedder = EmbeddingGenerator()
print(f"   ✓ Embedder loaded")

# Load GraphAdapter
print("\n[3] Connecting to Neo4j...")
neo4j_config = Neo4jConfig.from_env()
knowledge_graph = GraphAdapter.from_neo4j(neo4j_config)
entity_count = len(knowledge_graph.entities)
print(f"   ✓ GraphAdapter loaded ({entity_count} entities)")

# Get registry
print("\n[4] Getting tool registry...")
registry = get_registry()
registry.initialize_tools(
    vector_store=vector_store,
    embedder=embedder,
    knowledge_graph=knowledge_graph,
)

# Check if browse_entities is registered
print("\n[5] Checking tool registration...")
tool = registry.get_tool('browse_entities')
if tool:
    print("   ✅ browse_entities tool is REGISTERED!")
    print(f"   ✓ Tool name: {tool.name}")
    print(f"   ✓ Tool tier: {tool.tier}")
    print(f"   ✓ Requires KG: {tool.requires_kg}")
    print(f"   ✓ Description: {tool.description[:80]}...")
else:
    print("   ❌ browse_entities tool NOT FOUND in registry")
    all_tools = [t.name for t in registry.get_all_tools()]
    print(f"   Available tools: {all_tools}")

# Count tools by tier
all_tools = registry.get_all_tools()
tier1 = [t.name for t in all_tools if t.tier == 1]
tier2 = [t.name for t in all_tools if t.tier == 2]
tier3 = [t.name for t in all_tools if t.tier == 3]

print("\n[6] Tool Statistics:")
print(f"   Total tools: {len(all_tools)}")
print(f"   Tier 1 (Basic): {len(tier1)} - {tier1}")
print(f"   Tier 2 (Advanced): {len(tier2)} - {tier2}")
print(f"   Tier 3 (Analysis): {len(tier3)} - {tier3}")

# Test execution
if tool:
    print("\n[7] Testing browse_entities execution...")
    result = tool.execute(entity_type="standard", limit=5)

    if result.success:
        print(f"   ✅ Execution SUCCESS!")
        print(f"   ✓ Found {result.metadata['count']} entities")
        if result.data:
            print(f"   ✓ Sample entity: {result.data[0]['value']}")
            print(f"   ✓ Confidence: {result.data[0]['confidence']}")
    else:
        print(f"   ❌ Execution FAILED: {result.error}")

knowledge_graph.close()

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)
