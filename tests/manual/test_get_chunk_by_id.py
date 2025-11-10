"""Test HybridVectorStore.get_chunk_by_id() method."""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hybrid_search import HybridVectorStore

def test_get_chunk_by_id():
    """Test that get_chunk_by_id() works correctly."""

    print("="*70)
    print("Testing HybridVectorStore.get_chunk_by_id()")
    print("="*70)

    # Load vector store
    print("\n1. Loading HybridVectorStore from vector_db...")
    try:
        vector_store = HybridVectorStore.load(Path("vector_db"))
        stats = vector_store.get_stats()
        print(f"   ✅ Loaded: {stats['layer3_count']} chunks in layer 3")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return False

    # Get a sample chunk ID from metadata
    print("\n2. Getting sample chunk ID from metadata_layer3...")
    if not vector_store.faiss_store.metadata_layer3:
        print("   ❌ No chunks in metadata_layer3")
        return False

    sample_chunk = vector_store.faiss_store.metadata_layer3[0]
    chunk_id = sample_chunk["chunk_id"]
    print(f"   Sample chunk ID: {chunk_id}")
    print(f"   Document: {sample_chunk.get('document_id', 'N/A')}")

    # Test retrieval
    print("\n3. Testing get_chunk_by_id()...")
    try:
        result = vector_store.get_chunk_by_id(chunk_id)

        if result is None:
            print(f"   ❌ get_chunk_by_id() returned None")
            return False

        if result["chunk_id"] != chunk_id:
            print(f"   ❌ Wrong chunk returned:")
            print(f"      Expected: {chunk_id}")
            print(f"      Got: {result['chunk_id']}")
            return False

        print(f"   ✅ get_chunk_by_id() works correctly!")
        print(f"      Chunk ID: {result['chunk_id']}")
        print(f"      Document: {result.get('document_id', 'N/A')}")
        print(f"      Content length: {len(result.get('content', ''))} chars")

    except Exception as e:
        print(f"   ❌ get_chunk_by_id() failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test non-existent chunk
    print("\n4. Testing with non-existent chunk ID...")
    result = vector_store.get_chunk_by_id("NON_EXISTENT_ID")
    if result is None:
        print("   ✅ Correctly returns None for non-existent ID")
    else:
        print("   ❌ Should return None for non-existent ID")
        return False

    print(f"\n{'='*70}")
    print("✅ ALL TESTS PASSED")
    print("graph_search tool will now work with entity_mentions mode!")
    print(f"{'='*70}")

    return True

if __name__ == "__main__":
    success = test_get_chunk_by_id()
    sys.exit(0 if success else 1)
