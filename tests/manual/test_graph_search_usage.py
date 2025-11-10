#!/usr/bin/env python
"""
Test správného použití graph_search nástroje
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
print("TEST: Správné použití graph_search")
print("=" * 70)

# Načtení komponent
vector_store = HybridVectorStore.load(Path("vector_db"))
embedder = EmbeddingGenerator()
neo4j_config = Neo4jConfig.from_env()
knowledge_graph = GraphAdapter.from_neo4j(neo4j_config)

registry = get_registry()
registry.initialize_tools(
    vector_store=vector_store,
    embedder=embedder,
    knowledge_graph=knowledge_graph,
)

graph_tool = registry.get_tool('graph_search')
browse_tool = registry.get_tool('browse_entities')

# ❌ ŠPATNĚ: Bez entity_value
print("\n[1] ❌ ŠPATNÉ použití - bez entity_value:")
result = graph_tool.execute(
    mode="relationships",
    entity_type="regulation",
    relationship_types=["superseded_by"]
)
print(f"   Výsledek: {result.success}")
if not result.success:
    print(f"   Chyba: {result.error[:100]}...")

# ✅ SPRÁVNĚ: Nejdřív browse, pak graph_search pro každou entitu
print("\n[2] ✅ SPRÁVNÉ použití - browse + graph_search:")

# Krok 1: Najít regulace
print("   Krok 1: browse_entities pro nalezení regulací...")
browse_result = browse_tool.execute(entity_type="standard", limit=5)
print(f"   ✓ Nalezeno {len(browse_result.data)} regulací")

# Krok 2: Pro každou regulaci zkontrolovat vztahy
print("\n   Krok 2: graph_search pro každou regulaci...")
for entity in browse_result.data[:3]:
    result = graph_tool.execute(
        mode="relationships",
        entity_value=entity['value'],  # ✅ KLÍČOVÝ PARAMETR!
        relationship_types=["superseded_by", "supersedes"],
        k=5
    )
    
    if result.success:
        rels = result.data.get("relationships", [])
        print(f"   ✓ '{entity['value'][:50]}': {len(rels)} vztahů")
    else:
        print(f"   ❌ '{entity['value'][:50]}': chyba")

knowledge_graph.close()

print("\n" + "=" * 70)
print("✅ TEST DOKONČEN")
print("\n📝 ZÁVĚR:")
print("   graph_search VŽDY vyžaduje entity_value parametr!")
print("   Pro bulk operace: browse_entities + smyčka přes graph_search")
print("=" * 70)
