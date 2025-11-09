#!/usr/bin/env python3
"""
Rebuild vector_db from individual output/ directories.

Manually merges all per-document vector stores back into unified vector_db/.
"""

import sys
from pathlib import Path
from src.hybrid_search import HybridVectorStore

def main():
    output_dir = Path("output")
    target_dir = Path("vector_db")

    # Find all per-document vector stores
    doc_stores = []
    for doc_dir in output_dir.iterdir():
        if not doc_dir.is_dir():
            continue
        vector_store_path = doc_dir / "phase4_vector_store"
        if vector_store_path.exists():
            doc_stores.append(vector_store_path)

    print(f"Found {len(doc_stores)} document vector stores:")
    for store in doc_stores:
        print(f"  - {store.parent.name}")

    if not doc_stores:
        print("ERROR: No vector stores found in output/")
        sys.exit(1)

    # Clear existing vector_db
    print(f"\nClearing existing vector_db...")
    if target_dir.exists():
        import shutil
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Merge all stores sequentially
    print(f"\nMerging stores into {target_dir}...")
    merged_store = None

    for i, store_path in enumerate(doc_stores, 1):
        doc_name = store_path.parent.name
        print(f"\n[{i}/{len(doc_stores)}] Loading {doc_name}...")

        try:
            # Load store using classmethod
            store = HybridVectorStore.load(store_path)

            stats = store.get_stats()
            print(f"  Stats: {stats['total_vectors']} vectors, {stats['documents']} docs")

            if merged_store is None:
                # First store becomes the base
                merged_store = store
                print(f"  → Initialized merged store")
            else:
                # Merge subsequent stores
                print(f"  → Merging into unified store...")
                merged_store.merge(store)
                print(f"  → Merge complete (see logs for details)")

        except Exception as e:
            print(f"  ERROR: Failed to load {doc_name}: {e}")
            continue

    if merged_store is None:
        print("\nERROR: No stores were successfully loaded!")
        sys.exit(1)

    # Save merged store
    print(f"\nSaving merged store to {target_dir}...")
    merged_store.save(str(target_dir))

    final_stats = merged_store.get_stats()
    print(f"\n✅ SUCCESS!")
    print(f"  Total vectors: {final_stats['total_vectors']}")
    print(f"  Documents: {final_stats['documents']}")
    print(f"  Layer 1: {final_stats['layer1_count']}")
    print(f"  Layer 2: {final_stats['layer2_count']}")
    print(f"  Layer 3: {final_stats['layer3_count']}")
    print(f"\nMerged store saved to: {target_dir}/")

if __name__ == "__main__":
    main()
