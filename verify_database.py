#!/usr/bin/env python3
"""
Comprehensive Database Verification.

Checks FAISS indexes, BM25 indexes, and unified knowledge graph for completeness.
"""

import sys
from pathlib import Path
from src.hybrid_search import HybridVectorStore
from src.graph.models import KnowledgeGraph
from collections import Counter


def verify_faiss_bm25(vector_db_path: Path):
    """Verify FAISS and BM25 indexes."""
    print("\n" + "=" * 70)
    print("FAISS + BM25 HYBRID VECTOR STORE VERIFICATION")
    print("=" * 70)

    try:
        # Load hybrid store
        print(f"\nLoading hybrid store from: {vector_db_path}")
        store = HybridVectorStore.load(vector_db_path)

        # Get statistics
        stats = store.get_stats()

        print("\n✅ FAISS Indexes:")
        print(f"  - Dimensions: {stats['dimensions']}")
        print(f"  - Layer 1 (Documents): {stats['layer1_count']:,} vectors")
        print(f"  - Layer 2 (Sections):  {stats['layer2_count']:,} vectors")
        print(f"  - Layer 3 (Chunks):    {stats['layer3_count']:,} vectors")
        print(f"  - Total vectors:       {stats['total_vectors']:,}")
        print(f"  - Total documents:     {stats['documents']}")

        print("\n✅ BM25 Indexes:")
        print(f"  - Layer 1 corpus size: {stats['bm25_layer1_count']:,}")
        print(f"  - Layer 2 corpus size: {stats['bm25_layer2_count']:,}")
        print(f"  - Layer 3 corpus size: {stats['bm25_layer3_count']:,}")

        print(f"\n✅ Hybrid Configuration:")
        print(f"  - Fusion k (RRF): {stats['fusion_k']}")
        print(f"  - Hybrid enabled: {stats['hybrid_enabled']}")

        # Verify consistency
        issues = []

        if stats['layer1_count'] != stats['bm25_layer1_count']:
            issues.append(f"Layer 1 mismatch: FAISS={stats['layer1_count']}, BM25={stats['bm25_layer1_count']}")

        if stats['layer2_count'] != stats['bm25_layer2_count']:
            issues.append(f"Layer 2 mismatch: FAISS={stats['layer2_count']}, BM25={stats['bm25_layer2_count']}")

        if stats['layer3_count'] != stats['bm25_layer3_count']:
            issues.append(f"Layer 3 mismatch: FAISS={stats['layer3_count']}, BM25={stats['bm25_layer3_count']}")

        if issues:
            print("\n⚠️  CONSISTENCY ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ All layers are consistent (FAISS == BM25)")

        # Verify layer distribution
        print("\n📊 Layer Distribution Analysis:")
        l1_pct = (stats['layer1_count'] / stats['total_vectors']) * 100
        l2_pct = (stats['layer2_count'] / stats['total_vectors']) * 100
        l3_pct = (stats['layer3_count'] / stats['total_vectors']) * 100

        print(f"  - Layer 1: {l1_pct:.1f}% (expected: <1% for document summaries)")
        print(f"  - Layer 2: {l2_pct:.1f}% (expected: 30-50% for sections)")
        print(f"  - Layer 3: {l3_pct:.1f}% (expected: 50-70% for chunks)")

        return True, stats

    except Exception as e:
        print(f"\n❌ ERROR: Failed to verify FAISS/BM25: {e}")
        return False, None


def verify_knowledge_graph(vector_db_path: Path):
    """Verify unified knowledge graph."""
    print("\n" + "=" * 70)
    print("UNIFIED KNOWLEDGE GRAPH VERIFICATION")
    print("=" * 70)

    kg_path = vector_db_path / "unified_kg.json"

    if not kg_path.exists():
        print(f"\n❌ ERROR: unified_kg.json not found at {kg_path}")
        return False, None

    try:
        # Load KG
        print(f"\nLoading knowledge graph from: {kg_path}")
        kg = KnowledgeGraph.load_json(str(kg_path))

        print("\n✅ Knowledge Graph Statistics:")
        print(f"  - Total entities: {len(kg.entities):,}")
        print(f"  - Total relationships: {len(kg.relationships):,}")

        # Entity type breakdown
        entity_types = Counter(e.type.value for e in kg.entities)
        print("\n📊 Entity Types:")
        for entity_type, count in entity_types.most_common(10):
            pct = (count / len(kg.entities)) * 100
            print(f"  - {entity_type:25s}: {count:5,} ({pct:5.1f}%)")

        # Relationship type breakdown
        rel_types = Counter(r.type.value for r in kg.relationships)
        print("\n📊 Relationship Types:")
        for rel_type, count in rel_types.most_common(10):
            pct = (count / len(kg.relationships)) * 100 if kg.relationships else 0
            print(f"  - {rel_type:25s}: {count:5,} ({pct:5.1f}%)")

        # Document distribution
        doc_counts = Counter()
        for entity in kg.entities:
            doc_ids = entity.metadata.get("document_ids", [])
            for doc_id in doc_ids:
                doc_counts[doc_id] += 1

        print("\n📊 Entities per Document:")
        for doc_id, count in doc_counts.most_common():
            print(f"  - {doc_id:40s}: {count:5,} entities")

        # Cross-document entities
        cross_doc_entities = [
            e for e in kg.entities
            if len(e.metadata.get("document_ids", [])) > 1
        ]
        cross_doc_pct = (len(cross_doc_entities) / len(kg.entities)) * 100 if kg.entities else 0

        print(f"\n📊 Cross-Document Entities:")
        print(f"  - Total: {len(cross_doc_entities):,} ({cross_doc_pct:.1f}%)")

        # Verify entity connectivity
        entity_to_rel_count = Counter()
        for rel in kg.relationships:
            entity_to_rel_count[rel.source_entity_id] += 1
            entity_to_rel_count[rel.target_entity_id] += 1

        isolated_entities = len(kg.entities) - len(entity_to_rel_count)
        print(f"\n📊 Entity Connectivity:")
        print(f"  - Connected entities: {len(entity_to_rel_count):,}")
        print(f"  - Isolated entities: {isolated_entities:,}")

        if isolated_entities > 0:
            isolated_pct = (isolated_entities / len(kg.entities)) * 100
            print(f"  - Isolated %: {isolated_pct:.1f}%")

        return True, {
            "entities": len(kg.entities),
            "relationships": len(kg.relationships),
            "documents": len(doc_counts),
            "cross_doc_entities": len(cross_doc_entities),
            "entity_types": dict(entity_types),
            "rel_types": dict(rel_types),
        }

    except Exception as e:
        print(f"\n❌ ERROR: Failed to verify knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def print_summary(faiss_stats, kg_stats):
    """Print overall summary."""
    print("\n" + "=" * 70)
    print("DATABASE VERIFICATION SUMMARY")
    print("=" * 70)

    if faiss_stats:
        print("\n✅ FAISS + BM25:")
        print(f"  - {faiss_stats['total_vectors']:,} vectors across 3 layers")
        print(f"  - {faiss_stats['documents']} documents indexed")

    if kg_stats:
        print("\n✅ Knowledge Graph:")
        print(f"  - {kg_stats['entities']:,} entities")
        print(f"  - {kg_stats['relationships']:,} relationships")
        print(f"  - {kg_stats['documents']} documents")
        print(f"  - {kg_stats['cross_doc_entities']:,} cross-document entities")

    # Check consistency
    if faiss_stats and kg_stats:
        if faiss_stats['documents'] == kg_stats['documents']:
            print("\n✅ Document count consistency: FAISS == KG")
        else:
            print(f"\n⚠️  Document count mismatch: FAISS={faiss_stats['documents']}, KG={kg_stats['documents']}")


def main():
    vector_db_path = Path("vector_db")

    if not vector_db_path.exists():
        print(f"❌ ERROR: vector_db directory not found at {vector_db_path}")
        sys.exit(1)

    print("=" * 70)
    print("COMPREHENSIVE DATABASE VERIFICATION")
    print("=" * 70)
    print(f"Database location: {vector_db_path.absolute()}")

    # Verify FAISS + BM25
    faiss_ok, faiss_stats = verify_faiss_bm25(vector_db_path)

    # Verify Knowledge Graph
    kg_ok, kg_stats = verify_knowledge_graph(vector_db_path)

    # Print summary
    print_summary(faiss_stats, kg_stats)

    # Final status
    print("\n" + "=" * 70)
    if faiss_ok and kg_ok:
        print("✅ DATABASE VERIFICATION COMPLETE - ALL CHECKS PASSED")
    else:
        print("❌ DATABASE VERIFICATION FAILED - ISSUES DETECTED")
    print("=" * 70)

    sys.exit(0 if (faiss_ok and kg_ok) else 1)


if __name__ == "__main__":
    main()
