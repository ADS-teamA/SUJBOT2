"""
Tests for Legal Indexing Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings import LegalEmbedder, LegalChunk, EmbeddingConfig
from indexing import (
    MultiDocumentVectorStore,
    VectorStoreConfig,
    ReferenceMap,
    IndexPersistence,
    SearchResult,
    IndexError,
    IndexPersistenceError
)


@pytest.fixture
def embedder():
    """Create embedder instance"""
    config = EmbeddingConfig(
        model_name="BAAI/bge-m3",
        device="cpu",
        batch_size=2,
        show_progress_bar=False
    )
    return LegalEmbedder(config)


@pytest.fixture
def sample_chunks():
    """Create sample legal chunks"""
    return [
        LegalChunk(
            chunk_id="law_chunk_1",
            content="Dodavatel odpovídá za vady díla.",
            document_id="law_89_2012",
            document_type="law_code",
            legal_reference="§89",
            hierarchy_path="Část II > §89",
            metadata={"paragraph": 89, "content_type": "obligation"}
        ),
        LegalChunk(
            chunk_id="law_chunk_2",
            content="Záruční doba je 24 měsíců.",
            document_id="law_89_2012",
            document_type="law_code",
            legal_reference="§89 odst. 2",
            hierarchy_path="Část II > §89 > odst. 2",
            metadata={"paragraph": 89, "subsection": 2, "content_type": "definition"}
        ),
        LegalChunk(
            chunk_id="contract_chunk_1",
            content="Dodavatel dodá zboží do 30 dnů.",
            document_id="contract_123",
            document_type="contract",
            legal_reference="Článek 5",
            hierarchy_path="Článek 5",
            metadata={"article": 5, "contains_obligation": True}
        ),
    ]


class TestReferenceMap:
    """Test ReferenceMap class"""

    @pytest.mark.asyncio
    async def test_build_reference_map(self, sample_chunks):
        """Test building reference map"""
        ref_map = ReferenceMap()
        await ref_map.build(sample_chunks)

        # Check references are indexed
        assert "§89" in ref_map.ref_to_chunks
        assert "§89 odst. 2" in ref_map.ref_to_chunks
        assert "Článek 5" in ref_map.ref_to_chunks

    @pytest.mark.asyncio
    async def test_get_chunks_by_reference(self, sample_chunks):
        """Test getting chunks by reference"""
        ref_map = ReferenceMap()
        await ref_map.build(sample_chunks)

        chunks = ref_map.get_chunks_by_reference("§89")
        assert len(chunks) == 1
        assert "law_chunk_1" in chunks

    @pytest.mark.asyncio
    async def test_get_references_from_chunk(self):
        """Test getting outgoing references"""
        chunk = LegalChunk(
            chunk_id="test_chunk",
            content="Test",
            legal_reference="§89",
            metadata={"references_to": ["§88", "§90"]}
        )

        ref_map = ReferenceMap()
        await ref_map.build([chunk])

        refs = ref_map.get_references_from_chunk("test_chunk")
        assert refs == ["§88", "§90"]

    def test_serialize_deserialize(self):
        """Test serialization and deserialization"""
        ref_map = ReferenceMap()
        ref_map.ref_to_chunks["§89"] = ["chunk_1", "chunk_2"]
        ref_map.chunk_to_refs["chunk_1"] = ["§88", "§90"]

        # Serialize
        data = ref_map.serialize()

        # Deserialize
        new_ref_map = ReferenceMap.deserialize(data)

        assert new_ref_map.get_chunks_by_reference("§89") == ["chunk_1", "chunk_2"]
        assert new_ref_map.get_references_from_chunk("chunk_1") == ["§88", "§90"]


class TestMultiDocumentVectorStore:
    """Test MultiDocumentVectorStore class"""

    @pytest.mark.asyncio
    async def test_add_document(self, embedder, sample_chunks):
        """Test adding a document"""
        vector_store = MultiDocumentVectorStore(embedder)

        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]

        await vector_store.add_document(
            chunks=law_chunks,
            document_id="law_89_2012",
            document_type="law_code"
        )

        # Check document was added
        assert "law_89_2012" in vector_store.indices
        assert "law_89_2012" in vector_store.metadata_stores
        assert vector_store.get_document_count() == 1
        assert vector_store.get_chunk_count("law_89_2012") == 2

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, embedder, sample_chunks):
        """Test adding multiple documents"""
        vector_store = MultiDocumentVectorStore(embedder)

        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        contract_chunks = [c for c in sample_chunks if c.document_id == "contract_123"]

        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")
        await vector_store.add_document(contract_chunks, "contract_123", "contract")

        assert vector_store.get_document_count() == 2
        assert vector_store.get_chunk_count() == 3

    @pytest.mark.asyncio
    async def test_search_all_documents(self, embedder, sample_chunks):
        """Test searching across all documents"""
        vector_store = MultiDocumentVectorStore(embedder)

        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        contract_chunks = [c for c in sample_chunks if c.document_id == "contract_123"]

        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")
        await vector_store.add_document(contract_chunks, "contract_123", "contract")

        # Search
        results = await vector_store.search("dodavatel", top_k=5)

        # Should find results from both documents
        assert len(results) > 0
        assert any(r.document_id == "law_89_2012" for r in results)

    @pytest.mark.asyncio
    async def test_search_specific_document(self, embedder, sample_chunks):
        """Test searching in specific document"""
        vector_store = MultiDocumentVectorStore(embedder)

        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        contract_chunks = [c for c in sample_chunks if c.document_id == "contract_123"]

        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")
        await vector_store.add_document(contract_chunks, "contract_123", "contract")

        # Search only in contract
        results = await vector_store.search(
            "dodavatel",
            document_ids=["contract_123"],
            top_k=5
        )

        # All results should be from contract
        assert all(r.document_id == "contract_123" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, embedder, sample_chunks):
        """Test searching with metadata filter"""
        vector_store = MultiDocumentVectorStore(embedder)

        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")

        # Search with filter
        results = await vector_store.search(
            "záruční",
            filter_metadata={"content_type": "definition"},
            top_k=5
        )

        # All results should match filter
        assert all(r.chunk.metadata.get("content_type") == "definition" for r in results)

    @pytest.mark.asyncio
    async def test_search_by_reference(self, embedder, sample_chunks):
        """Test direct reference lookup"""
        vector_store = MultiDocumentVectorStore(embedder)

        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")

        # Lookup by reference
        chunk = await vector_store.search_by_reference("§89")

        assert chunk is not None
        assert chunk.legal_reference == "§89"
        assert chunk.chunk_id == "law_chunk_1"

    @pytest.mark.asyncio
    async def test_get_document_info(self, embedder, sample_chunks):
        """Test getting document info"""
        vector_store = MultiDocumentVectorStore(embedder)

        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        await vector_store.add_document(
            law_chunks,
            "law_89_2012",
            "law_code",
            metadata={"name": "Občanský zákoník"}
        )

        info = vector_store.get_document_info("law_89_2012")
        assert info is not None
        assert info["document_type"] == "law_code"
        assert info["num_chunks"] == 2
        assert info["metadata"]["name"] == "Občanský zákoník"


class TestIndexPersistence:
    """Test IndexPersistence class"""

    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary index directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_save_and_load(self, embedder, sample_chunks, temp_index_dir):
        """Test saving and loading index"""
        persistence = IndexPersistence(index_dir=temp_index_dir)

        # Create vector store
        vector_store = MultiDocumentVectorStore(embedder)
        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")

        # Save
        doc_id = "law_89_2012"
        await persistence.save(
            document_id=doc_id,
            index=vector_store.indices[doc_id],
            metadata=vector_store.document_info[doc_id],
            chunks=law_chunks,
            reference_map=vector_store.reference_map
        )

        # Check files exist
        assert (temp_index_dir / doc_id / "faiss.index").exists()
        assert (temp_index_dir / doc_id / "metadata.json").exists()
        assert (temp_index_dir / doc_id / "chunks.pkl").exists()
        assert (temp_index_dir / doc_id / "reference_map.json").exists()

        # Load
        index, metadata, chunks, ref_map = await persistence.load(doc_id)

        assert index.ntotal == 2  # 2 chunks
        assert metadata["document_type"] == "law_code"
        assert len(chunks) == 2
        assert ref_map is not None

    @pytest.mark.asyncio
    async def test_exists(self, embedder, sample_chunks, temp_index_dir):
        """Test checking if index exists"""
        persistence = IndexPersistence(index_dir=temp_index_dir)

        # Initially doesn't exist
        assert not persistence.exists("law_89_2012")

        # Create and save
        vector_store = MultiDocumentVectorStore(embedder)
        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")

        await persistence.save(
            document_id="law_89_2012",
            index=vector_store.indices["law_89_2012"],
            metadata=vector_store.document_info["law_89_2012"],
            chunks=law_chunks
        )

        # Now exists
        assert persistence.exists("law_89_2012")

    @pytest.mark.asyncio
    async def test_list_documents(self, embedder, sample_chunks, temp_index_dir):
        """Test listing saved documents"""
        persistence = IndexPersistence(index_dir=temp_index_dir)

        # Save multiple documents
        vector_store = MultiDocumentVectorStore(embedder)

        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        contract_chunks = [c for c in sample_chunks if c.document_id == "contract_123"]

        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")
        await vector_store.add_document(contract_chunks, "contract_123", "contract")

        for doc_id in ["law_89_2012", "contract_123"]:
            await persistence.save(
                document_id=doc_id,
                index=vector_store.indices[doc_id],
                metadata=vector_store.document_info[doc_id],
                chunks=law_chunks if doc_id == "law_89_2012" else contract_chunks
            )

        # List documents
        docs = persistence.list_documents()
        assert "law_89_2012" in docs
        assert "contract_123" in docs
        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_delete(self, embedder, sample_chunks, temp_index_dir):
        """Test deleting index"""
        persistence = IndexPersistence(index_dir=temp_index_dir)

        # Save
        vector_store = MultiDocumentVectorStore(embedder)
        law_chunks = [c for c in sample_chunks if c.document_id == "law_89_2012"]
        await vector_store.add_document(law_chunks, "law_89_2012", "law_code")

        await persistence.save(
            document_id="law_89_2012",
            index=vector_store.indices["law_89_2012"],
            metadata=vector_store.document_info["law_89_2012"],
            chunks=law_chunks
        )

        assert persistence.exists("law_89_2012")

        # Delete
        await persistence.delete("law_89_2012")

        assert not persistence.exists("law_89_2012")

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, temp_index_dir):
        """Test loading nonexistent index"""
        persistence = IndexPersistence(index_dir=temp_index_dir)

        with pytest.raises(IndexPersistenceError):
            await persistence.load("nonexistent_doc")


class TestVectorStoreConfig:
    """Test VectorStoreConfig dataclass"""

    def test_default_config(self):
        """Test default configuration"""
        config = VectorStoreConfig()

        assert config.index_type == "flat"
        assert config.vector_size == 1024
        assert config.enable_gpu is False

    def test_ivf_config(self):
        """Test IVF configuration"""
        config = VectorStoreConfig(
            index_type="ivf",
            ivf_nlist=200,
            ivf_nprobe=20
        )

        assert config.index_type == "ivf"
        assert config.ivf_nlist == 200
        assert config.ivf_nprobe == 20

    def test_hnsw_config(self):
        """Test HNSW configuration"""
        config = VectorStoreConfig(
            index_type="hnsw",
            hnsw_m=64,
            hnsw_ef_search=100
        )

        assert config.index_type == "hnsw"
        assert config.hnsw_m == 64
        assert config.hnsw_ef_search == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
