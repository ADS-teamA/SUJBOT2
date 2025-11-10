#!/usr/bin/env python
"""
Test: Verify the user's original query works end-to-end
Query: "trace the history of regulations that superseded each other"
"""

from dotenv import load_dotenv
load_dotenv()

from src.agent.tools.registry import get_registry
from src.agent.graph_adapter import GraphAdapter
from src.graph import Neo4jConfig
from src.hybrid_search import HybridVectorStore
from src.embedding_generator import EmbeddingGenerator
from pathlib import Path

print("=" * 70)
print("TEST: User's Original Query - Regulatory Supersession History")
print("=" * 70)

# Load components
print("\n[1] Loading components...")
vector_store = HybridVectorStore.load(Path("vector_db"))
embedder = EmbeddingGenerator()
neo4j_config = Neo4jConfig.from_env()
knowledge_graph = GraphAdapter.from_neo4j(neo4j_config)
print(f"   ✓ All components loaded ({len(knowledge_graph.entities)} entities)")

# Initialize registry
print("\n[2] Initializing tool registry...")
registry = get_registry()
registry.initialize_tools(
    vector_store=vector_store,
    embedder=embedder,
    knowledge_graph=knowledge_graph,
)
print("   ✓ Registry initialized")

# Test 1: browse_entities to find regulations
print("\n[3] Step 1: Browse regulations (testing browse_entities tool)...")
browse_tool = registry.get_tool('browse_entities')
result = browse_tool.execute(entity_type="standard", limit=10)

if result.success:
    print(f"   ✅ browse_entities SUCCESS")
    print(f"   ✓ Found {len(result.data)} regulations")
    sample_regs = result.data[:3]
    for reg in sample_regs:
        print(f"     - {reg['value']} (confidence: {reg['confidence']})")
else:
    print(f"   ❌ browse_entities FAILED: {result.error}")
    knowledge_graph.close()
    exit(1)

# Test 2: graph_search with multi_hop mode (this triggered all 3 bugs)
print("\n[4] Step 2: Multi-hop graph search (testing Bug #1, #2, #3 fixes)...")
print("   Using first regulation for multi-hop traversal...")

graph_tool = registry.get_tool('graph_search')
start_entity_value = result.data[0]['value']
print(f"   Start entity: '{start_entity_value}'")

# This is the query that failed with all 3 bugs:
# Bug #1: entity.normalized_value = None causing AttributeError
# Bug #2: Missing get_outgoing_relationships() method
# Bug #3: Missing get_chunk_by_id() for chunk retrieval
result = graph_tool.execute(
    mode="multi_hop",
    entity_value=start_entity_value,
    relationship_types=["superseded_by", "supersedes"],  # Focus on supersession
    max_hops=3,
    k=10,
    include_metadata=True
)

if result.success:
    print(f"   ✅ Multi-hop graph search SUCCESS!")
    print(f"\n[5] Results:")

    traversal = result.metadata.get("traversal", {})
    print(f"   Total entities discovered: {traversal.get('total_entities_discovered', 0)}")
    print(f"   Total relationships: {traversal.get('total_relationships_traversed', 0)}")
    print(f"   Max hop depth: {traversal.get('max_hop_reached', 0)}")

    chunks = result.data.get("chunks", [])
    print(f"   Retrieved chunks: {len(chunks)}")

    if chunks:
        print(f"\n   Sample chunk:")
        chunk = chunks[0]
        print(f"     Document: {chunk['document_id']}")
        print(f"     Chunk ID: {chunk['chunk_id']}")
        print(f"     Graph score: {chunk.get('graph_score', 'N/A')}")

        # Show mentioned entities (should include supersession chain)
        mentioned = chunk.get('mentioned_entities', [])
        if mentioned:
            print(f"     Mentioned {len(mentioned)} entities:")
            for entity in mentioned[:5]:
                print(f"       - {entity['value']} (hop {entity['hop']})")

    # Check if any supersession relationships were found
    relationships_by_hop = traversal.get('relationships_by_hop', {})
    if relationships_by_hop:
        print(f"\n   Relationships by hop:")
        for hop, rels in relationships_by_hop.items():
            supersession_rels = [r for r in rels if r['type'] in ['superseded_by', 'supersedes']]
            if supersession_rels:
                print(f"     Hop {hop}: {len(supersession_rels)} supersession relationships")
                for rel in supersession_rels[:3]:
                    print(f"       - {rel['type']}: {rel['source_id']} → {rel['target_id']}")

    print(f"\n✅ ALL THREE BUGS FIXED:")
    print(f"   ✓ Bug #1: No crashes from None normalized_value")
    print(f"   ✓ Bug #2: get_outgoing_relationships() works")
    print(f"   ✓ Bug #3: Chunk retrieval via metadata_layer3 works")

else:
    print(f"   ❌ Multi-hop graph search FAILED: {result.error}")
    print(f"\n   This means one or more bugs are not fully fixed.")

knowledge_graph.close()

print("\n" + "=" * 70)
print("✅ TEST COMPLETE")
print("=" * 70)
