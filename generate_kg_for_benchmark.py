"""
Generate knowledge graph for benchmark documents retroactively.

Reads already indexed documents from benchmark_dataset/privacy_qa/
and generates KG using simple backend + KnowledgeGraphPipeline.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.graph.kg_pipeline import KnowledgeGraphPipeline
from src.graph.config import KnowledgeGraphConfig, GraphBackend
from src.graph.unified_kg_manager import UnifiedKnowledgeGraphManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_chunks_from_text(text: str, document_id: str) -> List[Dict[str, Any]]:
    """
    Create chunks from document text.

    For KG extraction, we split into ~500-word chunks (optimal for entity extraction).
    """
    # Split by newlines (sentences in privacy policies)
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Group lines into chunks of ~500 words (~2000 chars)
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    for line in lines:
        line_length = len(line)

        # If adding this line exceeds chunk size, save current chunk
        if current_length + line_length > 2000 and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                "id": f"{document_id}_chunk_{chunk_index}",
                "content": chunk_text,
                "metadata": {
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "layer": 3,
                }
            })
            chunk_index += 1
            current_chunk = []
            current_length = 0

        current_chunk.append(line)
        current_length += line_length

    # Add final chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append({
            "id": f"{document_id}_chunk_{chunk_index}",
            "content": chunk_text,
            "metadata": {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "layer": 3,
            }
        })

    return chunks if chunks else [{"id": f"{document_id}_chunk_0", "content": text[:2000], "metadata": {"document_id": document_id, "chunk_index": 0, "layer": 3}}]


def main():
    """Generate KG for benchmark documents."""
    logger.info("=" * 80)
    logger.info("GENERATING KNOWLEDGE GRAPH FOR BENCHMARK")
    logger.info("=" * 80)

    # Paths
    source_dir = Path("benchmark_dataset/privacy_qa")
    output_dir = Path("benchmark_db")

    # Check source documents
    doc_files = sorted(source_dir.glob("*.txt"))

    if not doc_files:
        raise FileNotFoundError(f"No .txt files found in {source_dir}")

    logger.info(f"Found {len(doc_files)} documents to process:")
    for doc_file in doc_files:
        logger.info(f"  - {doc_file.name}")

    # Initialize KG pipeline with simple backend
    logger.info("\nInitializing Knowledge Graph pipeline (simple backend)...")
    kg_config = KnowledgeGraphConfig.from_env()
    kg_config.graph_storage.backend = GraphBackend.SIMPLE  # Force simple backend

    kg_pipeline = KnowledgeGraphPipeline(kg_config)

    # Initialize unified KG manager
    unified_manager = UnifiedKnowledgeGraphManager(storage_dir=str(output_dir))
    unified_kg = unified_manager.load_or_create()

    # Process each document
    total_entities = 0
    total_relationships = 0

    for doc_file in doc_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {doc_file.name}")
        logger.info(f"{'='*60}")

        document_id = doc_file.stem

        # Read document content
        with open(doc_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Create chunks
        chunks = create_chunks_from_text(text, document_id)
        logger.info(f"Created {len(chunks)} chunks for KG extraction")

        # Extract KG using pipeline
        try:
            doc_kg = kg_pipeline.build_from_chunks(
                chunks=chunks,
                document_id=document_id
            )

            logger.info(f"  ✓ Extracted {len(doc_kg.entities)} entities, "
                       f"{len(doc_kg.relationships)} relationships")

            # Merge into unified KG
            unified_kg = unified_manager.merge_document_graph(
                unified_kg,
                doc_kg,
                document_id=document_id
            )

            total_entities += len(doc_kg.entities)
            total_relationships += len(doc_kg.relationships)

        except Exception as e:
            logger.error(f"  ✗ Failed to process {doc_file.name}: {e}")
            continue

    # Save unified KG
    logger.info("\n" + "=" * 80)
    logger.info("SAVING UNIFIED KNOWLEDGE GRAPH")
    logger.info("=" * 80)

    unified_manager.save(unified_kg)

    # Compute stats manually
    entity_types = {}
    rel_types = {}
    documents = set()

    for entity in unified_kg.entities:
        entity_types[entity.type.value] = entity_types.get(entity.type.value, 0) + 1
        if entity.metadata and "document_ids" in entity.metadata:
            documents.update(entity.metadata["document_ids"])

    for rel in unified_kg.relationships:
        rel_types[rel.type.value] = rel_types.get(rel.type.value, 0) + 1

    stats = {
        "total_entities": len(unified_kg.entities),
        "total_relationships": len(unified_kg.relationships),
        "documents": len(documents),
        "entity_types": entity_types,
        "relationship_types": rel_types
    }

    logger.info(f"Knowledge graph saved to: {output_dir / 'unified_kg.json'}")
    logger.info(f"\nStatistics:")
    logger.info(f"  - Total entities: {stats['total_entities']}")
    logger.info(f"  - Total relationships: {stats['total_relationships']}")
    logger.info(f"  - Documents: {stats['documents']}")
    logger.info(f"  - Entity types: {list(stats['entity_types'].keys()) if stats['entity_types'] else 'None'}")
    logger.info(f"  - Relationship types: {list(stats['relationship_types'].keys()) if stats['relationship_types'] else 'None'}")

    # Save stats
    stats_path = Path("benchmark_results") / "kg_stats.json"
    stats_path.parent.mkdir(exist_ok=True)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nStatistics saved to: {stats_path}")
    logger.info("\n✓ Knowledge graph generation complete!")
    logger.info(f"  Ready for benchmark with KG-enabled agent tools")


if __name__ == "__main__":
    main()
