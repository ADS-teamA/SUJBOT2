#!/usr/bin/env python
"""
Quick test: Verify multi-hop BFS search works with chunk retrieval
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
print("TEST: Multi-Hop BFS Search with Chunk Retrieval")
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

# Get graph_search tool
tool = registry.get_tool('graph_search')
if not tool:
    print("   ❌ graph_search tool NOT FOUND")
    exit(1)
print("   ✓ graph_search tool found")

# Find a regulation entity to start from
print("\n[5] Finding a regulation entity...")
sample_regulations = knowledge_graph.find_entities(
    entity_type="standard",
    min_confidence=0.8
)

if not sample_regulations:
    print("   ❌ No regulations found")
    exit(1)

start_entity = sample_regulations[0]
print(f"   ✓ Using entity: '{start_entity.value}'")
print(f"   ✓ Entity ID: {start_entity.id}")
print(f"   ✓ Confidence: {start_entity.confidence}")

# Test multi-hop BFS search
print("\n[6] Testing multi-hop BFS search...")
print(f"   Starting from: '{start_entity.value}'")
print(f"   Max hops: 2")
print(f"   Top k chunks: 5")

result = tool.execute(
    mode="multi_hop",
    entity_value=start_entity.value,
    max_hops=2,
    k=5,
    include_metadata=True
)

if result.success:
    print(f"   ✅ Multi-hop search SUCCESS!")
    print(f"\n[7] Traversal Results:")

    traversal = result.metadata.get("traversal", {})
    print(f"   Start entity: {traversal.get('start_entity', {}).get('value')}")
    print(f"   Total entities discovered: {traversal.get('total_entities_discovered', 0)}")
    print(f"   Total relationships traversed: {traversal.get('total_relationships_traversed', 0)}")
    print(f"   Max hop reached: {traversal.get('max_hop_reached', 0)}")

    entities_by_hop = traversal.get('entities_by_hop', {})
    if entities_by_hop:
        print(f"\n   Entities discovered by hop:")
        for hop, count in sorted(entities_by_hop.items()):
            print(f"     Hop {hop}: {count} entities")

    print(f"\n[8] Retrieved Chunks:")
    chunks = result.data.get("chunks", [])
    print(f"   Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
        print(f"\n   Chunk {i}:")
        print(f"     Document: {chunk['document_id']}")
        print(f"     Chunk ID: {chunk['chunk_id']}")
        print(f"     Graph score: {chunk.get('graph_score', 'N/A')}")

        mentioned = chunk.get('mentioned_entities', [])
        if mentioned:
            print(f"     Mentioned entities: {len(mentioned)}")
            for entity in mentioned[:2]:  # Show first 2 entities
                print(f"       - {entity['value']} (hop {entity['hop']})")

        # Show preview of content
        content = chunk.get('text', chunk.get('content', ''))
        preview = content[:100] + "..." if len(content) > 100 else content
        print(f"     Content preview: {preview}")

    print(f"\n[9] Citations:")
    for citation in result.citations[:5]:  # Show first 5
        print(f"   - {citation}")

else:
    print(f"   ❌ Multi-hop search FAILED: {result.error}")

knowledge_graph.close()

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)
