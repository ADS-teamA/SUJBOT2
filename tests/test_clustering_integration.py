"""
Integration tests for semantic clustering with indexing pipeline.

Tests:
- Clustering integration with pipeline
- Cluster metadata in FAISS vector store
- Cluster-aware agent tools
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.indexing_pipeline import IndexingPipeline, IndexingConfig
from src.config import ClusteringConfig
from src.multi_layer_chunker import Chunk, ChunkMetadata


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunks = []
    
    # Create 30 chunks with different content
    topics = [
        "waste management regulations",
        "safety standards compliance",
        "environmental protection measures",
    ]
    
    for i in range(30):
        topic = topics[i % 3]
        chunk = Chunk(
            chunk_id=f"test_doc_L3_section_{i // 10}_chunk_{i}",
            content=f"This is about {topic}. Content {i}.",
            raw_content=f"This is about {topic}. Content {i}.",
            metadata=ChunkMetadata(
                chunk_id=f"test_doc_L3_section_{i // 10}_chunk_{i}",
                layer=3,
                document_id="test_doc",
                section_id=f"section_{i // 10}",
                section_title=f"Section {i // 10}",
                page_number=i // 10,
            ),
        )
        chunks.append(chunk)
    
    return {
        "layer1": [],
        "layer2": [],
        "layer3": chunks,
    }


# ============================================================================
# Pipeline Integration Tests
# ============================================================================


def test_clustering_in_pipeline(sample_chunks, temp_output_dir, monkeypatch):
    """Test clustering integration with indexing pipeline."""
    # Mock environment for testing
    monkeypatch.setenv("ENABLE_SEMANTIC_CLUSTERING", "true")
    monkeypatch.setenv("CLUSTERING_ALGORITHM", "agglomerative")
    monkeypatch.setenv("CLUSTERING_N_CLUSTERS", "3")
    monkeypatch.setenv("CLUSTERING_ENABLE_VIZ", "false")
    
    # Create config with clustering enabled
    config = IndexingConfig.from_env(
        enable_semantic_clustering=True,
        enable_knowledge_graph=False,
        enable_hybrid_search=False,
    )
    
    # Verify clustering config was initialized
    assert config.enable_semantic_clustering is True
    assert config.clustering_config is not None
    assert config.clustering_config.algorithm == "agglomerative"
    assert config.clustering_config.n_clusters == 3


def test_cluster_metadata_in_chunks(sample_chunks):
    """Test that cluster metadata is added to chunks."""
    from src.clustering import SemanticClusterer, ClusteringConfig
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig
    
    # Generate embeddings
    embedding_config = EmbeddingConfig(
        provider="huggingface",
        model="BAAI/bge-small-en-v1.5",
        batch_size=32,
    )
    embedder = EmbeddingGenerator(embedding_config)
    
    chunks_list = sample_chunks["layer3"]
    embeddings = embedder.embed_chunks(chunks_list, layer=3)
    
    # Perform clustering
    clustering_config = ClusteringConfig(
        algorithm="agglomerative",
        n_clusters=3,
        enable_cluster_labels=False,
        enable_visualization=False,
    )
    clusterer = SemanticClusterer(clustering_config)
    
    chunk_ids = [c.chunk_id for c in chunks_list]
    result = clusterer.cluster_embeddings(embeddings, chunk_ids)
    
    # Update chunk metadata
    for chunk in chunks_list:
        cluster_info = result.get_chunk_cluster(chunk.chunk_id)
        if cluster_info:
            chunk.metadata.cluster_id = cluster_info.cluster_id
            chunk.metadata.cluster_label = cluster_info.label
            
            # Calculate confidence
            chunk_idx = chunk_ids.index(chunk.chunk_id)
            chunk_embedding = embeddings[chunk_idx]
            centroid = cluster_info.centroid
            if centroid is not None:
                distance = 1 - np.dot(chunk_embedding, centroid)
                chunk.metadata.cluster_confidence = float(distance)
    
    # Verify cluster metadata was added
    clustered_chunks = [c for c in chunks_list if c.metadata.cluster_id is not None]
    assert len(clustered_chunks) > 0
    
    for chunk in clustered_chunks:
        assert chunk.metadata.cluster_id is not None
        assert chunk.metadata.cluster_id >= 0
        assert chunk.metadata.cluster_confidence is not None
        assert 0 <= chunk.metadata.cluster_confidence <= 2  # Cosine distance range


def test_cluster_metadata_in_faiss(sample_chunks):
    """Test that cluster metadata is stored in FAISS vector store."""
    from src.faiss_vector_store import FAISSVectorStore
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig
    from src.clustering import SemanticClusterer, ClusteringConfig
    
    # Generate embeddings
    embedding_config = EmbeddingConfig(
        provider="huggingface",
        model="BAAI/bge-small-en-v1.5",
        batch_size=32,
    )
    embedder = EmbeddingGenerator(embedding_config)
    
    chunks_list = sample_chunks["layer3"]
    embeddings_array = embedder.embed_chunks(chunks_list, layer=3)
    
    # Perform clustering
    clustering_config = ClusteringConfig(
        algorithm="agglomerative",
        n_clusters=3,
        enable_cluster_labels=False,
        enable_visualization=False,
    )
    clusterer = SemanticClusterer(clustering_config)
    
    chunk_ids = [c.chunk_id for c in chunks_list]
    result = clusterer.cluster_embeddings(embeddings_array, chunk_ids)
    
    # Update chunk metadata
    for chunk in chunks_list:
        cluster_info = result.get_chunk_cluster(chunk.chunk_id)
        if cluster_info:
            chunk.metadata.cluster_id = cluster_info.cluster_id
            chunk.metadata.cluster_label = cluster_info.label
    
    # Create FAISS vector store
    vector_store = FAISSVectorStore(dimensions=embedder.dimensions)
    
    embeddings_dict = {
        "layer1": None,
        "layer2": None,
        "layer3": embeddings_array,
    }
    
    vector_store.add_chunks(sample_chunks, embeddings_dict)
    
    # Verify cluster metadata in stored metadata
    assert len(vector_store.metadata_layer3) > 0
    
    for meta in vector_store.metadata_layer3:
        # Check that cluster fields exist (may be None if not clustered)
        assert "cluster_id" in meta
        assert "cluster_label" in meta
        assert "cluster_confidence" in meta


# ============================================================================
# Cluster Distribution Tests
# ============================================================================


def test_cluster_distribution(sample_chunks):
    """Test that chunks are distributed across clusters."""
    from src.clustering import SemanticClusterer, ClusteringConfig
    from src.embedding_generator import EmbeddingGenerator, EmbeddingConfig
    
    # Generate embeddings
    embedding_config = EmbeddingConfig(
        provider="huggingface",
        model="BAAI/bge-small-en-v1.5",
        batch_size=32,
    )
    embedder = EmbeddingGenerator(embedding_config)
    
    chunks_list = sample_chunks["layer3"]
    embeddings = embedder.embed_chunks(chunks_list, layer=3)
    
    # Perform clustering
    clustering_config = ClusteringConfig(
        algorithm="agglomerative",
        n_clusters=3,
        enable_cluster_labels=False,
        enable_visualization=False,
    )
    clusterer = SemanticClusterer(clustering_config)
    
    chunk_ids = [c.chunk_id for c in chunks_list]
    result = clusterer.cluster_embeddings(embeddings, chunk_ids)
    
    # Check cluster distribution
    cluster_counts = {}
    for cluster_id in result.cluster_assignments.values():
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    
    # Should have 3 clusters
    assert len(cluster_counts) == 3
    
    # Each cluster should have some chunks
    for count in cluster_counts.values():
        assert count > 0


# ============================================================================
# Performance Tests
# ============================================================================


def test_clustering_performance():
    """Test clustering performance with larger dataset."""
    import time
    
    # Generate 1000 embeddings
    np.random.seed(42)
    embeddings = np.random.randn(1000, 128)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    chunk_ids = [f"chunk_{i}" for i in range(1000)]
    
    # Test HDBSCAN performance
    config_hdbscan = ClusteringConfig(
        algorithm="hdbscan",
        min_cluster_size=10,
        enable_cluster_labels=False,
        enable_visualization=False,
    )
    clusterer_hdbscan = SemanticClusterer(config_hdbscan)
    
    start = time.time()
    result_hdbscan = clusterer_hdbscan.cluster_embeddings(embeddings, chunk_ids)
    time_hdbscan = time.time() - start
    
    # Test Agglomerative performance
    config_agg = ClusteringConfig(
        algorithm="agglomerative",
        n_clusters=10,
        enable_cluster_labels=False,
        enable_visualization=False,
    )
    clusterer_agg = SemanticClusterer(config_agg)
    
    start = time.time()
    result_agg = clusterer_agg.cluster_embeddings(embeddings, chunk_ids)
    time_agg = time.time() - start
    
    # Both should complete in reasonable time (< 10 seconds)
    assert time_hdbscan < 10
    assert time_agg < 10
    
    # Both should produce valid results
    assert result_hdbscan.n_clusters > 0
    assert result_agg.n_clusters == 10

