"""
Example usage of the Legal Embedding and Indexing System

Demonstrates:
1. Creating legal chunks
2. Generating contextualized embeddings
3. Building multi-document vector store
4. Searching across documents
5. Reference-based lookup
6. Index persistence
"""

import asyncio
import logging
from pathlib import Path
import yaml

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from embeddings import LegalEmbedder, LegalChunk, EmbeddingConfig
from indexing import (
    MultiDocumentVectorStore,
    VectorStoreConfig,
    IndexPersistence,
    ReferenceMap
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_sample_chunks() -> tuple:
    """Create sample legal chunks for demonstration"""

    # Law code chunks
    law_chunks = [
        LegalChunk(
            chunk_id="law_89_chunk_1",
            content="Dodavatel odpovídá za vady díla, které má věc při převzetí kupujícím. Zejména dodavatel odpovídá za to, že věc při převzetí odpovídá jakosti a provedení.",
            document_id="law_89_2012",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §89",
            legal_reference="§89",
            structural_level="paragraph",
            metadata={
                "part": "II",
                "chapter": "III",
                "paragraph": 89,
                "content_type": "obligation",
                "law_citation": "Zákon č. 89/2012 Sb.",
                "references_to": ["§88", "§90"],
                "token_count": 45
            }
        ),
        LegalChunk(
            chunk_id="law_89_chunk_2",
            content="Záruční doba je 24 měsíců ode dne převzetí věci kupujícím. U použitých věcí je záruční doba nejméně 12 měsíců.",
            document_id="law_89_2012",
            document_type="law_code",
            hierarchy_path="Část II > Hlava III > §89 > odst. 2",
            legal_reference="§89 odst. 2",
            structural_level="subsection",
            metadata={
                "part": "II",
                "chapter": "III",
                "paragraph": 89,
                "subsection": 2,
                "content_type": "definition",
                "law_citation": "Zákon č. 89/2012 Sb.",
                "token_count": 28
            }
        ),
    ]

    # Contract chunks
    contract_chunks = [
        LegalChunk(
            chunk_id="contract_123_chunk_1",
            content="Dodavatel se zavazuje dodat zboží podle specifikace v příloze A a kupující se zavazuje zaplatit kupní cenu do 30 dnů od doručení faktury.",
            document_id="contract_123",
            document_type="contract",
            hierarchy_path="Článek 5",
            legal_reference="Článek 5",
            structural_level="article",
            metadata={
                "article": 5,
                "article_title": "Závazky stran",
                "contains_obligation": True,
                "parties_mentioned": ["dodavatel", "kupující"],
                "token_count": 35
            }
        ),
        LegalChunk(
            chunk_id="contract_123_chunk_2",
            content="V případě prodlení s dodáním zboží zaplatí dodavatel smluvní pokutu ve výši 0,1% z ceny za každý den prodlení.",
            document_id="contract_123",
            document_type="contract",
            hierarchy_path="Článek 8.1",
            legal_reference="Článek 8.1",
            structural_level="article_point",
            metadata={
                "article": 8,
                "article_title": "Smluvní pokuty",
                "point": 1,
                "contains_penalty": True,
                "token_count": 25
            }
        ),
    ]

    return law_chunks, contract_chunks


async def example_basic_embedding():
    """Example 1: Basic embedding generation"""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Embedding Generation")
    logger.info("=" * 60)

    # Create embedder
    config = EmbeddingConfig(
        model_name="BAAI/bge-m3",
        device="cpu",
        add_hierarchical_context=True
    )
    embedder = LegalEmbedder(config)

    # Create sample chunk
    chunk = LegalChunk(
        chunk_id="example_1",
        content="Dodavatel odpovídá za vady díla.",
        document_type="law_code",
        hierarchy_path="Část II > §89",
        legal_reference="§89"
    )

    # Generate embedding
    embeddings = await embedder.embed_chunks([chunk])

    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
    logger.info(f"First 5 values: {embeddings[0][:5]}")

    return embedder


async def example_vector_store():
    """Example 2: Multi-document vector store"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Multi-Document Vector Store")
    logger.info("=" * 60)

    # Create embedder and vector store
    embedder = LegalEmbedder()
    vector_store = MultiDocumentVectorStore(embedder)

    # Get sample chunks
    law_chunks, contract_chunks = create_sample_chunks()

    # Add documents
    logger.info("Adding law document...")
    await vector_store.add_document(
        chunks=law_chunks,
        document_id="law_89_2012",
        document_type="law_code",
        metadata={"name": "Občanský zákoník", "year": 2012}
    )

    logger.info("Adding contract document...")
    await vector_store.add_document(
        chunks=contract_chunks,
        document_id="contract_123",
        document_type="contract",
        metadata={"contract_number": "123/2024"}
    )

    # Get stats
    logger.info(f"\nVector store stats:")
    logger.info(f"  Total documents: {vector_store.get_document_count()}")
    logger.info(f"  Total chunks: {vector_store.get_chunk_count()}")
    logger.info(f"  Law chunks: {vector_store.get_chunk_count('law_89_2012')}")
    logger.info(f"  Contract chunks: {vector_store.get_chunk_count('contract_123')}")

    return vector_store


async def example_search():
    """Example 3: Semantic search"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Semantic Search")
    logger.info("=" * 60)

    # Create vector store
    embedder = LegalEmbedder()
    vector_store = MultiDocumentVectorStore(embedder)
    law_chunks, contract_chunks = create_sample_chunks()

    await vector_store.add_document(law_chunks, "law_89_2012", "law_code")
    await vector_store.add_document(contract_chunks, "contract_123", "contract")

    # Search across all documents
    query = "odpovědnost za vady"
    logger.info(f"\nSearching for: '{query}'")

    results = await vector_store.search(query, top_k=3)

    logger.info(f"\nFound {len(results)} results:")
    for result in results:
        logger.info(f"\n  Rank {result.rank}:")
        logger.info(f"    Document: {result.document_id}")
        logger.info(f"    Reference: {result.chunk.legal_reference}")
        logger.info(f"    Score: {result.score:.4f}")
        logger.info(f"    Content: {result.chunk.content[:100]}...")

    # Search in specific document
    logger.info(f"\nSearching only in law document:")
    results = await vector_store.search(
        query,
        document_ids=["law_89_2012"],
        top_k=2
    )

    for result in results:
        logger.info(f"  {result.chunk.legal_reference}: {result.score:.4f}")


async def example_reference_lookup():
    """Example 4: Reference-based lookup"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Reference-based Lookup")
    logger.info("=" * 60)

    # Create vector store
    embedder = LegalEmbedder()
    vector_store = MultiDocumentVectorStore(embedder)
    law_chunks, _ = create_sample_chunks()

    await vector_store.add_document(law_chunks, "law_89_2012", "law_code")

    # Lookup by reference
    ref = "§89"
    logger.info(f"\nLooking up reference: {ref}")

    chunk = await vector_store.search_by_reference(ref)

    if chunk:
        logger.info(f"Found chunk:")
        logger.info(f"  ID: {chunk.chunk_id}")
        logger.info(f"  Reference: {chunk.legal_reference}")
        logger.info(f"  Citation: {chunk.get_citation()}")
        logger.info(f"  Content: {chunk.content[:150]}...")
    else:
        logger.info(f"No chunk found for reference {ref}")


async def example_persistence():
    """Example 5: Index persistence"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Index Persistence")
    logger.info("=" * 60)

    # Create persistence handler
    persistence = IndexPersistence(index_dir=Path("./indexes"))

    # Create and populate vector store
    embedder = LegalEmbedder()
    vector_store = MultiDocumentVectorStore(embedder)
    law_chunks, _ = create_sample_chunks()

    await vector_store.add_document(law_chunks, "law_89_2012", "law_code")

    # Save index
    doc_id = "law_89_2012"
    logger.info(f"Saving index for document: {doc_id}")

    await persistence.save(
        document_id=doc_id,
        index=vector_store.indices[doc_id],
        metadata=vector_store.document_info[doc_id],
        chunks=law_chunks,
        reference_map=vector_store.reference_map
    )

    logger.info(f"Index saved to: {persistence.index_dir / doc_id}")

    # List saved indices
    saved_docs = persistence.list_documents()
    logger.info(f"\nSaved indices: {saved_docs}")

    # Load index
    logger.info(f"\nLoading index for document: {doc_id}")
    index, metadata, chunks, ref_map = await persistence.load(doc_id)

    logger.info(f"Loaded:")
    logger.info(f"  Index vectors: {index.ntotal}")
    logger.info(f"  Chunks: {len(chunks)}")
    logger.info(f"  Metadata: {metadata}")
    if ref_map:
        logger.info(f"  References: {len(ref_map.ref_to_chunks)}")


async def example_filtering():
    """Example 6: Metadata filtering"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: Metadata Filtering")
    logger.info("=" * 60)

    # Create vector store
    embedder = LegalEmbedder()
    vector_store = MultiDocumentVectorStore(embedder)
    law_chunks, contract_chunks = create_sample_chunks()

    await vector_store.add_document(law_chunks, "law_89_2012", "law_code")
    await vector_store.add_document(contract_chunks, "contract_123", "contract")

    # Search with filter
    query = "dodavatel"
    filter_metadata = {"content_type": "obligation"}

    logger.info(f"Searching for: '{query}'")
    logger.info(f"Filter: {filter_metadata}")

    results = await vector_store.search(
        query,
        top_k=5,
        filter_metadata=filter_metadata
    )

    logger.info(f"\nFound {len(results)} results with filter:")
    for result in results:
        logger.info(f"  {result.chunk.legal_reference}: {result.chunk.metadata.get('content_type')}")


async def main():
    """Run all examples"""
    logger.info("Legal Embedding & Indexing System - Example Usage\n")

    try:
        # Run examples
        await example_basic_embedding()
        await example_vector_store()
        await example_search()
        await example_reference_lookup()
        await example_persistence()
        await example_filtering()

        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Import numpy here for the example
    import numpy as np

    # Run async main
    asyncio.run(main())
