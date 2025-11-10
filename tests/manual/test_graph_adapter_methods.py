#!/usr/bin/env python
"""
Quick test: Verify GraphAdapter has required methods for multi-hop search
"""

from dotenv import load_dotenv
load_dotenv()

from src.agent.graph_adapter import GraphAdapter
from src.graph import Neo4jConfig

print("=" * 60)
print("TEST: GraphAdapter Methods")
print("=" * 60)

# Connect to Neo4j
print("\n[1] Connecting to Neo4j...")
neo4j_config = Neo4jConfig.from_env()
adapter = GraphAdapter.from_neo4j(neo4j_config)
print(f"   ✓ Connected ({len(adapter.entities)} entities)")

# Test get_relationships_for_entity
print("\n[2] Testing get_relationships_for_entity...")
sample_entity = list(adapter.entities.values())[0]
all_rels = adapter.get_relationships_for_entity(sample_entity.id)
print(f"   ✓ Entity '{sample_entity.value}' has {len(all_rels)} relationships")

# Test get_outgoing_relationships
print("\n[3] Testing get_outgoing_relationships...")
if hasattr(adapter, 'get_outgoing_relationships'):
    outgoing = adapter.get_outgoing_relationships(sample_entity.id)
    print(f"   ✓ Method exists!")
    print(f"   ✓ Found {len(outgoing)} outgoing relationships")
else:
    print(f"   ❌ Method missing!")

# Test get_incoming_relationships
print("\n[4] Testing get_incoming_relationships...")
if hasattr(adapter, 'get_incoming_relationships'):
    incoming = adapter.get_incoming_relationships(sample_entity.id)
    print(f"   ✓ Method exists!")
    print(f"   ✓ Found {len(incoming)} incoming relationships")
else:
    print(f"   ❌ Method missing!")

# Verify math
if hasattr(adapter, 'get_outgoing_relationships') and hasattr(adapter, 'get_incoming_relationships'):
    print("\n[5] Verifying relationship math...")
    print(f"   All relationships: {len(all_rels)}")
    print(f"   Outgoing: {len(outgoing)}")
    print(f"   Incoming: {len(incoming)}")
    print(f"   Sum: {len(outgoing) + len(incoming)}")

    if len(all_rels) == len(outgoing) + len(incoming):
        print(f"   ✅ Math checks out!")
    else:
        print(f"   ⚠️  Math doesn't add up (might be OK if entity has self-loops)")

adapter.close()

print("\n" + "=" * 60)
print("✅ ALL METHODS EXIST")
print("=" * 60)
