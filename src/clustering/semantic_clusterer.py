"""
Semantic clustering for vector database chunks.

Implements HDBSCAN, Agglomerative, guided spherical k-means, and fuzzy
c-means clustering with cosine distance metrics. All algorithms operate on
normalized embeddings for consistency with the vector store's cosine
similarity search.

Research backing:
- Compliance Check (papers/compliance_check.md): K-means clustering reduces
  LLM calls by 40-70% through semantic deduplication
- Uses cosine distance for consistency with embedding space
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import ClusteringConfig from centralized config
from src.config import ClusteringConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterInfo:
    """Information about a single cluster."""
    
    cluster_id: int
    label: Optional[str] = None  # Semantic label (e.g., "Waste Management Regulations")
    size: int = 0  # Number of chunks in cluster
    centroid: Optional[np.ndarray] = None  # Cluster centroid (mean embedding)
    representative_chunks: List[str] = field(default_factory=list)  # Chunk IDs closest to centroid


@dataclass
class ClusteringResult:
    """
    Result of semantic clustering operation.
    
    Attributes:
        cluster_assignments: Mapping from chunk_id to cluster_id
        cluster_info: Information about each cluster
        n_clusters: Total number of clusters (excluding noise)
        noise_count: Number of chunks marked as noise (-1 cluster_id)
        quality_metrics: Clustering quality metrics (silhouette, etc.)
    """
    
    cluster_assignments: Dict[str, int]  # chunk_id -> cluster_id
    cluster_info: Dict[int, ClusterInfo]  # cluster_id -> ClusterInfo
    n_clusters: int
    noise_count: int = 0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    memberships: Optional[np.ndarray] = None  # Soft membership matrix (N x K)
    
    def get_chunk_cluster(self, chunk_id: str) -> Optional[ClusterInfo]:
        """Get cluster info for a specific chunk."""
        cluster_id = self.cluster_assignments.get(chunk_id)
        if cluster_id is None or cluster_id == -1:
            return None
        return self.cluster_info.get(cluster_id)
    
    def get_cluster_chunks(self, cluster_id: int) -> List[str]:
        """Get all chunk IDs in a specific cluster."""
        return [
            chunk_id for chunk_id, cid in self.cluster_assignments.items()
            if cid == cluster_id
        ]


class SemanticClusterer:
    """
    Semantic clustering for vector database chunks.
    
    Supports:
    - HDBSCAN: Density-based clustering with automatic cluster count
    - Agglomerative: Hierarchical clustering with cosine distance
    - Guided modes: Spherical k-means, nearest-centroid, and fuzzy c-means
    
    All algorithms use cosine distance for consistency with normalized embeddings.
    """
    
    def __init__(self, config: ClusteringConfig):
        """
        Initialize semantic clusterer.
        
        Args:
            config: Clustering configuration
        """
        self.config = config
        logger.info(
            f"SemanticClusterer initialized: algorithm={config.algorithm}, "
            f"metric=cosine (fixed)"
        )
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
    ) -> ClusteringResult:
        """
        Cluster embeddings and return assignments.
        
        Args:
            embeddings: Normalized embedding vectors (N x D)
            chunk_ids: List of chunk IDs corresponding to embeddings
        
        Returns:
            ClusteringResult with cluster assignments and metadata
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and chunk_ids ({len(chunk_ids)}) "
                f"must have same length"
            )
        
        if len(embeddings) == 0:
            logger.warning("No embeddings to cluster")
            return ClusteringResult(
                cluster_assignments={},
                cluster_info={},
                n_clusters=0,
                noise_count=0
            )
        
        logger.info(f"Clustering {len(embeddings)} embeddings using {self.config.algorithm}")
        
        # Ensure embeddings are normalized for cosine distance
        embeddings = self._normalize_embeddings(embeddings)
        
        # Perform clustering
        if self.config.algorithm == "hdbscan":
            labels = self._cluster_hdbscan(embeddings)
        elif self.config.algorithm == "agglomerative":
            labels = self._cluster_agglomerative(embeddings)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        # Build result
        result = self._build_clustering_result(embeddings, chunk_ids, labels, memberships=None)
        
        # Generate cluster labels if enabled
        if self.config.enable_cluster_labels:
            logger.info("Generating semantic labels for clusters...")
            # TODO: Implement LLM-based label generation in next step
            logger.warning("Cluster label generation not yet implemented")

        # Generate visualization if enabled
        if self.config.enable_visualization:
            from pathlib import Path
            import time

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config.visualization_output_dir) / f"clusters_{timestamp}.png"

            self.visualize_clusters(
                embeddings=embeddings,
                labels=labels,
                cluster_info=result.cluster_info,
                output_path=str(output_path),
                title=f"Semantic Clusters ({self.config.algorithm.upper()}, n={result.n_clusters})",
                memberships=None,
            )

        logger.info(
            f"Clustering complete: {result.n_clusters} clusters, "
            f"{result.noise_count} noise points"
        )

        return result

    def guided_cluster(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
        seed_centroids: Optional[np.ndarray] = None,
        seed_labels: Optional[List[str]] = None,
        algorithm: str = "spherical_kmeans",
        max_iter: int = 30,
        tol: float = 1e-4,
        n_clusters: Optional[int] = None,
        fuzziness: float = 2.0,
        init_method: str = "kmeans++",
    ) -> ClusteringResult:
        """
        Guided or semi-guided clustering with optional seed centroids.

        Workflow:
        1) Provide user queries to seed centroids or specify n_clusters for auto init
        2) Embeddings are normalized for cosine similarity
        3) Points are clustered with the selected algorithm

        Supported algorithms:
        - spherical_kmeans: Iterative refinement on unit sphere (cosine)
        - agglomerative: Nearest-centroid assignment by cosine
        - fuzzy_cmeans: Soft clustering with cosine distance and fuzziness parameter

        Args:
            embeddings: Embedding matrix (N x D), normalized internally
            chunk_ids: Chunk identifiers aligned with embeddings
            seed_centroids: Optional seed centroids (K x D) from user queries
            seed_labels: Optional semantic labels for clusters (length K)
            algorithm: 'spherical_kmeans', 'agglomerative', or 'fuzzy_cmeans'
            max_iter: Max iterations for iterative algorithms
            tol: Convergence tolerance for centroid change
            n_clusters: Cluster count when auto-initializing centroids
            fuzziness: Fuzziness coefficient m (>1) for fuzzy c-means
            init_method: Centroid initialization strategy ('kmeans++' or 'random')

        Returns:
            ClusteringResult with assignments and metadata
        """
        if len(embeddings) != len(chunk_ids):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and chunk_ids ({len(chunk_ids)}) must have same length"
            )

        if embeddings.size == 0:
            logger.warning("No embeddings to cluster (guided)")
            return ClusteringResult(
                cluster_assignments={},
                cluster_info={},
                n_clusters=0,
                noise_count=0,
            )

        # Normalize for cosine distance
        embeddings = self._normalize_embeddings(embeddings)

        k = None
        if seed_centroids is not None and seed_centroids.size > 0:
            seed_centroids = self._normalize_embeddings(seed_centroids)
            k = seed_centroids.shape[0]

        if n_clusters is not None:
            if k is not None and k != n_clusters:
                logger.warning(
                    "n_clusters (%d) does not match number of seed centroids (%d); using seed count",
                    n_clusters,
                    k,
                )
            else:
                k = n_clusters

        if k is None:
            raise ValueError(
                "Unable to determine cluster count. Provide seed centroids or set n_clusters."
            )

        if algorithm == "fuzzy_cmeans" and fuzziness <= 1.0:
            raise ValueError("fuzziness parameter must be > 1 for fuzzy c-means")

        if seed_labels is not None and len(seed_labels) != k:
            raise ValueError("seed_labels length must match number of seed centroids")

        logger.info(
            f"Guided clustering: algorithm={algorithm}, n_samples={len(embeddings)}, k={k}, "
            f"seed_mode={'guided' if seed_centroids is not None and seed_centroids.size > 0 else 'auto'}"
        )

        memberships: Optional[np.ndarray] = None

        if algorithm == "spherical_kmeans":
            if seed_centroids is None or seed_centroids.size == 0:
                centroids = self._initialize_centroids(embeddings, k, method=init_method)
            else:
                centroids = seed_centroids
            labels = self._cluster_spherical_kmeans(
                embeddings,
                centroids,
                max_iter=max_iter,
                tol=tol,
            )
        elif algorithm == "agglomerative":
            if seed_centroids is None or seed_centroids.size == 0:
                centroids = self._initialize_centroids(embeddings, k, method=init_method)
            else:
                centroids = seed_centroids
            sims = np.dot(embeddings, centroids.T)
            labels = np.argmax(sims, axis=1).astype(int)
        elif algorithm == "fuzzy_cmeans":
            labels, memberships = self._cluster_fuzzy_cmeans(
                embeddings=embeddings,
                init_centroids=seed_centroids,
                n_clusters=k,
                max_iter=max_iter,
                tol=tol,
                fuzziness=fuzziness,
                init_method=init_method,
            )
        else:
            raise ValueError(
                "Unsupported guided algorithm. Use 'spherical_kmeans', 'agglomerative', or 'fuzzy_cmeans'"
            )

        result = self._build_clustering_result(embeddings, chunk_ids, labels, memberships=memberships)

        # Apply provided seed labels to cluster info if available
        if seed_labels is not None:
            for cid, info in result.cluster_info.items():
                if 0 <= cid < len(seed_labels):
                    info.label = seed_labels[cid]

        if memberships is not None:
            membership_entropy = -float(
                np.mean(np.sum(memberships * np.log(np.maximum(memberships, 1e-12)), axis=1))
            )
            result.quality_metrics["fuzzy_membership_entropy"] = membership_entropy
            result.quality_metrics["fuzziness"] = float(fuzziness)

        # Optional visualization
        if self.config.enable_visualization:
            from pathlib import Path
            import time

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config.visualization_output_dir) / f"guided_clusters_{timestamp}.png"
            self.visualize_clusters(
                embeddings=embeddings,
                labels=labels,
                cluster_info=result.cluster_info,
                output_path=str(output_path),
                title=f"Guided Clusters ({algorithm}, k={k})",
                memberships=memberships,
            )

        logger.info(
            f"Guided clustering complete: {result.n_clusters} clusters, labels set={seed_labels is not None}"
        )

        return result

    def _cluster_spherical_kmeans(
        self,
        embeddings: np.ndarray,
        init_centroids: np.ndarray,
        max_iter: int = 30,
        tol: float = 1e-4,
    ) -> np.ndarray:
        """
        Spherical k-means clustering with fixed initialization.

        - Operates on L2-normalized vectors (unit sphere)
        - Uses cosine similarity (dot product) for assignments

        Args:
            embeddings: Normalized embeddings (N x D)
            init_centroids: Initial centroids (K x D), normalized
            max_iter: Maximum iterations
            tol: Convergence threshold on centroid L2 change

        Returns:
            labels: np.ndarray shape (N,) with cluster assignments in [0..K-1]
        """
        n_samples, dim = embeddings.shape
        centroids = init_centroids.copy()
        k = centroids.shape[0]

        # Safety normalize
        centroids = self._normalize_embeddings(centroids)

        labels = np.zeros(n_samples, dtype=int)
        for it in range(max_iter):
            # Assignment step (cosine similarity)
            sims = np.dot(embeddings, centroids.T)
            new_labels = np.argmax(sims, axis=1).astype(int)

            # Update step: compute new centroids as normalized mean
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = new_labels == c
                if np.any(mask):
                    cvec = embeddings[mask].mean(axis=0)
                    norm = np.linalg.norm(cvec)
                    if norm > 0:
                        new_centroids[c] = cvec / norm
                    else:
                        new_centroids[c] = centroids[c]
                else:
                    # Empty cluster: keep previous centroid
                    new_centroids[c] = centroids[c]

            # Check convergence
            delta = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels = new_labels

            logger.debug(f"sKMeans iter {it+1}: delta={delta:.6f}")
            if delta < tol:
                break

        return labels

    def _cluster_fuzzy_cmeans(
        self,
        embeddings: np.ndarray,
        init_centroids: Optional[np.ndarray],
        n_clusters: int,
        max_iter: int = 50,
        tol: float = 1e-4,
        fuzziness: float = 2.0,
        init_method: str = "kmeans++",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuzzy C-means clustering tailored for cosine similarity.

        Args:
            embeddings: Normalized embeddings (N x D)
            init_centroids: Optional initial centroids (K x D)
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            tol: Convergence tolerance on centroid movement
            fuzziness: Fuzziness coefficient m (> 1)
            init_method: Initialization strategy if centroids not provided

        Returns:
            Tuple of (hard labels, membership matrix)
        """

        n_samples = embeddings.shape[0]
        if init_centroids is None or init_centroids.size == 0:
            centroids = self._initialize_centroids(embeddings, n_clusters, method=init_method)
        else:
            centroids = init_centroids.copy()

        centroids = self._normalize_embeddings(centroids)

        memberships = np.full((n_samples, n_clusters), 1.0 / n_clusters, dtype=np.float64)
        m = float(fuzziness)
        exponent = 2.0 / (m - 1.0)
        eps = 1e-9

        for it in range(max_iter):
            sims = np.dot(embeddings, centroids.T)
            np.clip(sims, -1.0, 1.0, out=sims)
            dist = 1.0 - sims
            dist = np.maximum(dist, eps)

            inv_dist = dist ** (-exponent)
            memberships = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)

            um = memberships ** m
            numerator = np.dot(um.T, embeddings)
            denominator = np.sum(um, axis=0)[:, None]
            safe_denominator = np.maximum(denominator, eps)

            new_centroids = numerator / safe_denominator
            new_centroids = self._normalize_embeddings(new_centroids)

            # Reinitialize any empty centroids
            empty_mask = np.squeeze(denominator <= eps)
            if np.any(empty_mask):
                replacement = self._initialize_centroids(
                    embeddings,
                    int(np.sum(empty_mask)),
                    method=init_method,
                )
                new_centroids[empty_mask] = replacement

            delta = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            logger.debug(f"fuzzy c-means iter {it+1}: delta={delta:.6f}")
            if delta < tol:
                break

        labels = np.argmax(memberships, axis=1).astype(int)
        return labels, memberships

    def _initialize_centroids(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        method: str = "kmeans++",
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Initialize centroids on the unit sphere using cosine-aware strategies.

        Args:
            embeddings: Normalized embeddings (N x D)
            n_clusters: Number of centroids to produce
            method: 'kmeans++' (default) or 'random'
            random_state: RNG seed for reproducibility
        """

        n_samples = embeddings.shape[0]
        if n_clusters > n_samples:
            raise ValueError("Number of clusters cannot exceed number of samples")

        rng = np.random.default_rng(random_state)

        if method == "random":
            indices = rng.choice(n_samples, size=n_clusters, replace=False)
            return embeddings[indices].copy()

        if method != "kmeans++":
            raise ValueError(f"Unsupported init_method: {method}")

        centroids = np.zeros((n_clusters, embeddings.shape[1]), dtype=embeddings.dtype)
        idx = rng.integers(0, n_samples)
        centroids[0] = embeddings[idx]

        for i in range(1, n_clusters):
            sims = np.dot(embeddings, centroids[:i].T)
            if sims.ndim == 1:
                sims = sims[:, None]
            np.clip(sims, -1.0, 1.0, out=sims)
            distances = 1.0 - np.max(sims, axis=1)
            distances = np.maximum(distances, 1e-9)
            probs = distances / distances.sum()
            idx = rng.choice(n_samples, p=probs)
            centroids[i] = embeddings[idx]

        return centroids

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine distance."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    
    def _cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster using HDBSCAN with cosine distance.
        
        HDBSCAN advantages:
        - Automatic cluster count detection
        - Handles noise (outliers marked as -1)
        - Density-based (finds clusters of varying shapes)
        
        Args:
            embeddings: Normalized embeddings (N x D)
        
        Returns:
            Cluster labels (N,) with -1 for noise
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan required for HDBSCAN clustering. "
                "Install with: uv pip install hdbscan"
            )
        
        logger.info(f"Running HDBSCAN with min_cluster_size={self.config.min_cluster_size}")

        # Compute cosine distance matrix explicitly to avoid compatibility issues
        # with scikit-learn's pairwise_distances changes. For normalized vectors,
        # cosine distance = 1 - (A @ B^T).
        sim = np.dot(embeddings, embeddings.T)
        np.clip(sim, -1.0, 1.0, out=sim)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        # HDBSCAN expects double precision for precomputed distances
        if dist.dtype != np.float64:
            dist = dist.astype(np.float64, copy=False)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.min_cluster_size,
            metric="precomputed",
            cluster_selection_method="eom",
        )

        labels = clusterer.fit_predict(dist)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        logger.info(
            f"HDBSCAN complete: {n_clusters} clusters, {n_noise} noise points"
        )
        
        return labels
    
    def _cluster_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster using Agglomerative clustering with cosine distance.
        
        Agglomerative advantages:
        - Hierarchical structure
        - Deterministic results
        - Works well with cosine distance
        
        Args:
            embeddings: Normalized embeddings (N x D)
        
        Returns:
            Cluster labels (N,)
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError(
                "scikit-learn required for Agglomerative clustering. "
                "Install with: uv pip install scikit-learn"
            )
        
        # Determine number of clusters
        if self.config.n_clusters is not None:
            n_clusters = self.config.n_clusters
            logger.info(f"Using specified n_clusters={n_clusters}")
        else:
            n_clusters = self._determine_optimal_clusters(embeddings)
            logger.info(f"Auto-detected optimal n_clusters={n_clusters}")
        
        logger.info(f"Running Agglomerative clustering with n_clusters={n_clusters}")
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",  # Average linkage works well with cosine
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        logger.info(f"Agglomerative complete: {n_clusters} clusters")

        return labels

    def _determine_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """
        Determine optimal number of clusters using silhouette score.

        Strategy:
        - Try different cluster counts from min_clusters to max_clusters
        - Compute silhouette score for each
        - Select cluster count with highest silhouette score
        - Fallback to sqrt(n) heuristic if all scores are poor

        Args:
            embeddings: Normalized embeddings (N x D)

        Returns:
            Optimal number of clusters
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError(
                "scikit-learn required for optimal cluster detection. "
                "Install with: uv pip install scikit-learn"
            )

        n_samples = len(embeddings)

        # Determine search range
        min_k = max(self.config.min_clusters, 2)  # Need at least 2 clusters
        max_k = min(self.config.max_clusters, n_samples - 1)

        # Fallback if not enough samples
        if n_samples < min_k:
            logger.warning(
                f"Not enough samples ({n_samples}) for clustering. "
                f"Minimum {min_k} required. Using n_clusters=2"
            )
            return 2

        # Use sqrt(n) heuristic as default
        default_k = max(min_k, min(int(np.sqrt(n_samples)), max_k))

        logger.info(
            f"Searching for optimal clusters: range=[{min_k}, {max_k}], "
            f"default={default_k}"
        )

        # Try different cluster counts (sample every 2-3 to speed up)
        step = max(1, (max_k - min_k) // 10)
        k_values = list(range(min_k, max_k + 1, step))
        if max_k not in k_values:
            k_values.append(max_k)

        best_score = -1
        best_k = default_k

        for k in k_values:
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=k,
                    metric="cosine",
                    linkage="average",
                )
                labels = clusterer.fit_predict(embeddings)

                # Compute silhouette score (higher is better)
                score = silhouette_score(embeddings, labels, metric="cosine")

                logger.debug(f"k={k}: silhouette={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                logger.warning(f"Failed to cluster with k={k}: {e}")
                continue

        logger.info(
            f"Optimal clusters: k={best_k} (silhouette={best_score:.4f})"
        )

        return best_k

    def _build_clustering_result(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
        labels: np.ndarray,
        memberships: Optional[np.ndarray] = None,
    ) -> ClusteringResult:
        """
        Build ClusteringResult from clustering output.

        Args:
            embeddings: Normalized embeddings (N x D)
            chunk_ids: List of chunk IDs
            labels: Cluster labels (N,)

        Returns:
            ClusteringResult with assignments and metadata
        """
        # Build cluster assignments
        cluster_assignments = {
            chunk_id: int(label)
            for chunk_id, label in zip(chunk_ids, labels)
        }

        # Count noise points
        noise_count = int(np.sum(labels == -1))

        # Get unique cluster IDs (excluding noise)
        unique_labels = set(labels)
        unique_labels.discard(-1)
        n_clusters = len(unique_labels)

        # Build cluster info
        cluster_info = {}

        for cluster_id in unique_labels:
            # Get chunk indices in this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Get cluster embeddings
            cluster_embeddings = embeddings[cluster_mask]

            # Compute centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # Normalize

            # Find representative chunks (closest to centroid)
            distances = 1 - np.dot(cluster_embeddings, centroid)  # Cosine distance
            closest_indices = np.argsort(distances)[:5]  # Top 5 closest
            representative_chunks = [
                chunk_ids[cluster_indices[i]] for i in closest_indices
            ]

            cluster_info[int(cluster_id)] = ClusterInfo(
                cluster_id=int(cluster_id),
                label=None,  # Will be filled by label generation
                size=len(cluster_indices),
                centroid=centroid,
                representative_chunks=representative_chunks,
            )

        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(embeddings, labels)

        return ClusteringResult(
            cluster_assignments=cluster_assignments,
            cluster_info=cluster_info,
            n_clusters=n_clusters,
            noise_count=noise_count,
            quality_metrics=quality_metrics,
            memberships=memberships,
        )

    def _compute_quality_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.

        Metrics:
        - silhouette_score: Measures cluster separation (-1 to 1, higher is better)
        - davies_bouldin_score: Measures cluster compactness (lower is better)
        - n_clusters: Number of clusters
        - noise_ratio: Ratio of noise points

        Args:
            embeddings: Normalized embeddings (N x D)
            labels: Cluster labels (N,)

        Returns:
            Dictionary of quality metrics
        """
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score
        except ImportError:
            logger.warning("scikit-learn not available for quality metrics")
            return {}

        metrics = {}

        # Filter out noise points for metrics
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) < 2:
            logger.warning("Not enough non-noise points for quality metrics")
            return metrics

        filtered_embeddings = embeddings[non_noise_mask]
        filtered_labels = labels[non_noise_mask]

        # Silhouette score (higher is better, range: -1 to 1)
        try:
            silhouette = silhouette_score(
                filtered_embeddings,
                filtered_labels,
                metric="cosine"
            )
            metrics["silhouette_score"] = float(silhouette)
        except Exception as e:
            logger.warning(f"Failed to compute silhouette score: {e}")

        # Davies-Bouldin score (lower is better, range: 0 to inf)
        try:
            davies_bouldin = davies_bouldin_score(
                filtered_embeddings,
                filtered_labels
            )
            metrics["davies_bouldin_score"] = float(davies_bouldin)
        except Exception as e:
            logger.warning(f"Failed to compute Davies-Bouldin score: {e}")

        # Basic statistics
        metrics["n_clusters"] = len(set(filtered_labels))
        metrics["noise_ratio"] = float(np.sum(labels == -1) / len(labels))

        return metrics

    def visualize_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_info: Dict[int, ClusterInfo],
        output_path: str,
        title: str = "Semantic Clusters (UMAP Projection)",
        memberships: Optional[np.ndarray] = None,
    ) -> None:
        """
        Visualize clusters using UMAP dimensionality reduction with confidence cues.

        Creates a 2D scatter plot with:
        - Each point is a chunk
        - Colors represent clusters
        - Marker size/opacity encode membership confidence (if available)
        - Cluster labels annotated with size and mean confidence
        - Noise points shown in gray

        Args:
            embeddings: Normalized embeddings (N x D)
            labels: Cluster labels (N,)
            cluster_info: Cluster metadata with labels
            output_path: Path to save visualization (e.g., "output/clusters/viz.png")
            title: Plot title
            memberships: Optional fuzzy membership matrix (N x K)
        """
        try:
            import umap
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from pathlib import Path
        except ImportError:
            logger.error(
                "UMAP visualization requires umap-learn and matplotlib. "
                "Install with: uv pip install umap-learn matplotlib"
            )
            return

        logger.info(f"Generating UMAP visualization: {output_path}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Reduce to 2D using UMAP
        logger.info("Running UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )
        embedding_2d = reducer.fit_transform(embeddings)

        # Determine confidence scores (membership max) when available
        if memberships is not None and memberships.shape[0] == embeddings.shape[0]:
            confidence = memberships.max(axis=1)
        else:
            if memberships is not None:
                logger.warning(
                    "Membership matrix shape %s does not match embeddings %s; ignoring memberships",
                    memberships.shape,
                    embeddings.shape,
                )
            confidence = np.ones(len(labels), dtype=np.float32)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Get unique clusters (excluding noise)
        unique_labels = set(labels)
        noise_mask = labels == -1
        unique_labels.discard(-1)

        # Generate colors for clusters and legend handles
        n_clusters = len(unique_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
        legend_handles: List[Line2D] = []
        low_conf_overlays = False

        # Plot noise points first (gray)
        if np.any(noise_mask):
            ax.scatter(
                embedding_2d[noise_mask, 0],
                embedding_2d[noise_mask, 1],
                c="lightgray",
                s=25,
                alpha=0.35,
                edgecolors="gray",
                linewidths=0.4,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color="lightgray",
                    label=f"Noise ({int(np.sum(noise_mask))})",
                    markerfacecolor="lightgray",
                    markeredgecolor="gray",
                    markersize=6,
                    alpha=0.6,
                )
            )

        # Plot each cluster
        for idx, cluster_id in enumerate(sorted(unique_labels)):
            cluster_mask = labels == cluster_id
            cluster_points = embedding_2d[cluster_mask]
            cluster_conf = confidence[cluster_mask]

            if cluster_points.size == 0:
                continue

            # Get cluster label
            cluster_label = cluster_info.get(cluster_id, ClusterInfo(cluster_id=cluster_id)).label
            if cluster_label is None:
                cluster_label = f"Cluster {cluster_id}"

            base_color = np.array(colors[idx % len(colors)], copy=True)
            if base_color.shape[0] == 3:
                base_color = np.append(base_color, 1.0)

            rgba = np.repeat(base_color[np.newaxis, :], cluster_points.shape[0], axis=0)
            rgba[:, 3] = 0.3 + 0.6 * cluster_conf
            sizes = 28 + 90 * cluster_conf

            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=rgba,
                s=sizes,
                edgecolors="black",
                linewidths=0.35,
            )

            # Highlight uncertain points (low confidence)
            uncertain_idx = np.where(cluster_conf < 0.55)[0]
            if uncertain_idx.size > 0:
                low_conf_overlays = True
                ax.scatter(
                    cluster_points[uncertain_idx, 0],
                    cluster_points[uncertain_idx, 1],
                    marker="x",
                    c="#2b2b2b",
                    s=55,
                    linewidths=0.7,
                    alpha=0.85,
                )

            # Annotate cluster centroid with stats
            centroid_2d = np.mean(cluster_points, axis=0)
            avg_conf = float(cluster_conf.mean()) if cluster_conf.size else 0.0
            annotation = f"{cluster_label}\nN={len(cluster_points)} | avg={avg_conf:.2f}"
            ax.annotate(
                annotation,
                xy=centroid_2d,
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=base_color, alpha=0.78),
            )

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=base_color,
                    label=f"{cluster_label} ({len(cluster_points)})",
                    markerfacecolor=base_color,
                    markeredgecolor="black",
                    markersize=7,
                    alpha=0.9,
                )
            )

        if low_conf_overlays:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    linestyle="",
                    color="#2b2b2b",
                    label="Low confidence (<0.55)",
                    markersize=7,
                    markeredgewidth=1.0,
                )
            )

        # Styling
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("UMAP Dimension 1", fontsize=12)
        ax.set_ylabel("UMAP Dimension 2", fontsize=12)
        ax.grid(True, alpha=0.3)

        if legend_handles:
            fig.subplots_adjust(right=0.8)
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                framealpha=0.95,
                fontsize=9,
            )

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✓ Visualization saved: {output_path}")
        logger.info(f"  Clusters: {n_clusters}, Noise points: {np.sum(noise_mask)}")
