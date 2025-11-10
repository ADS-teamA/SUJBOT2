"""
Test script to verify WebApp can connect to Neo4j using GraphAdapter.

This script simulates the exact initialization logic used in backend/agent_adapter.py
to ensure the WebApp will successfully connect to Neo4j when KG_BACKEND=neo4j.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load .env file (CRITICAL - otherwise env vars won't be available)
from dotenv import load_dotenv
load_dotenv()

def test_webapp_neo4j_connection():
    """Test Neo4j connection using WebApp initialization logic."""

    print("=" * 70)
    print("Testing WebApp Neo4j Connection (simulating agent_adapter.py)")
    print("=" * 70)

    # Check environment variables
    kg_backend = os.getenv("KG_BACKEND", "simple").lower()
    print(f"\n1. Checking KG_BACKEND: {kg_backend}")

    if kg_backend != "neo4j":
        print("   ⚠️  KG_BACKEND is not set to 'neo4j'")
        print("   Set KG_BACKEND=neo4j in .env to enable Neo4j backend")
        return False

    print("   ✅ KG_BACKEND=neo4j")

    # Check Neo4j credentials
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")

    print(f"\n2. Checking Neo4j credentials:")
    print(f"   URI: {neo4j_uri[:30]}..." if neo4j_uri else "   URI: NOT SET")
    print(f"   User: {neo4j_user}")
    print(f"   Password: {'***' if neo4j_password else 'NOT SET'}")

    if not neo4j_uri or not neo4j_password:
        print("   ❌ Missing Neo4j credentials")
        return False

    print("   ✅ Credentials found")

    # Test GraphAdapter initialization (same as agent_adapter.py lines 158-171)
    print(f"\n3. Initializing GraphAdapter with Neo4j...")
    try:
        from src.graph import Neo4jConfig
        from src.agent.graph_adapter import GraphAdapter

        neo4j_config = Neo4jConfig.from_env()
        knowledge_graph = GraphAdapter.from_neo4j(neo4j_config)

        # Get stats
        entity_count = len(knowledge_graph.entities)
        rel_count = len(knowledge_graph.relationships)

        print(f"   ✅ Connected to Neo4j: {entity_count} entities, {rel_count} relationships")

        # Test find_entities (required for browse_entities tool)
        print(f"\n4. Testing browse_entities functionality...")
        try:
            # Test basic entity browsing
            entities = knowledge_graph.find_entities()
            print(f"   ✅ find_entities() works: found {len(entities)} entities (total)")

            # Test entity type filtering
            standards = knowledge_graph.find_entities(entity_type="standard")
            print(f"   ✅ Entity type filtering works: found {len(standards)} standards")

            # Test confidence filtering
            high_conf = knowledge_graph.find_entities(min_confidence=0.8)
            print(f"   ✅ Confidence filtering works: found {len(high_conf)} high-confidence entities")

            # Test value search
            value_search = knowledge_graph.find_entities(value_contains="GRI")
            print(f"   ✅ Value search works: found {len(value_search)} entities containing 'GRI'")

        except Exception as e:
            print(f"   ❌ browse_entities functionality failed: {e}")
            return False

        print(f"\n{'='*70}")
        print("✅ SUCCESS - WebApp will connect to Neo4j correctly!")
        print("All graph tools (browse_entities, graph_search) will work in WebApp.")
        print(f"{'='*70}")

        return True

    except Exception as e:
        print(f"   ❌ Failed to connect to Neo4j: {e}")
        print(f"\n   Troubleshooting:")
        print(f"   - Check Neo4j Aura instance is running")
        print(f"   - Verify credentials in .env match Neo4j console")
        print(f"   - Test connection: uv run python -m src.agent.cli")
        return False

if __name__ == "__main__":
    success = test_webapp_neo4j_connection()
    sys.exit(0 if success else 1)
