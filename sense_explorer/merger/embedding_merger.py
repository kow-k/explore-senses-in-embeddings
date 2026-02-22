#!/usr/bin/env python3
"""
embedding_merger.py - Embedding Merger Module for SenseExplorer
================================================================

Combines word embeddings from multiple sources into a unified semantic space.

Theoretical Foundation:
    Embedding merger requires sense separation as a prerequisite. Without
    decomposing polysemous words into sense components, merging embeddings
    would just combine superpositions into messier superpositions.
    
    With sense separation (via SSR), we can:
    1. Extract sense components from each embedding
    2. Identify shared semantic basis across embeddings
    3. Project senses onto this basis for comparison
    4. Cluster to find convergent (shared) vs source-specific senses

Clustering Methods (v0.9.3):
    - 'hierarchical': Standard agglomerative clustering on similarity matrix
    - 'spectral': Spectral clustering with eigengap-based k selection
    - 'spectral_hierarchical': Hybrid - spectral embedding + hierarchical clustering
                               (recommended - wave-aware + dendrogram visualization)

Cognitive Motivation:
    Humans build unified lexicons from diverse linguistic experiences.
    This module models that process: merging Wikipedia's encyclopedic
    knowledge with Twitter's colloquial usage, for example.

Integration with SenseExplorer:
    ```python
    from sense_explorer import SenseExplorer
    from sense_explorer.merger import EmbeddingMerger
    
    # Load multiple embeddings
    se_wiki = SenseExplorer.from_file("glove-wiki-100d.txt")
    se_twitter = SenseExplorer.from_file("glove-twitter-100d.txt")
    
    # Create merger with spectral-hierarchical clustering
    merger = EmbeddingMerger(clustering_method='spectral_hierarchical')
    merger.add_embedding("wikipedia", se_wiki.embeddings)
    merger.add_embedding("twitter", se_twitter.embeddings)
    
    # Extract and merge senses
    result = merger.merge_senses("bank", sense_extractor=se_wiki.induce_senses)
    ```

Author: Kow Kuroda & Claude (Anthropic)
Version: 0.2.0 (for SenseExplorer v0.9.3)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import warnings

try:
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.manifold import spectral_embedding
    from scipy.sparse.csgraph import laplacian
    from scipy.linalg import eigh
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available; clustering features disabled")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SenseComponent:
    """
    A sense component extracted from a specific embedding source.
    
    This is the output of sense separation (SSR) — the prerequisite for merger.
    """
    word: str
    sense_id: str
    vector: np.ndarray
    source: str  # e.g., "wikipedia", "twitter", "news"
    top_neighbors: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize vector
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm


@dataclass
class MergerBasis:
    """
    The shared semantic basis for comparing two sense components.
    """
    core_words: List[str]  # Words in both neighborhoods
    extension_u: List[str]  # Words only in sense u's neighborhood
    extension_v: List[str]  # Words only in sense v's neighborhood
    all_words: List[str]  # Combined basis
    overlap_ratio: float  # |core| / |union|
    
    @property
    def core_size(self) -> int:
        return len(self.core_words)
    
    @property
    def total_size(self) -> int:
        return len(self.all_words)


@dataclass
class SpectralInfo:
    """
    Information from spectral analysis of similarity matrix.
    """
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    eigengaps: np.ndarray
    suggested_k: int
    spectral_coords: np.ndarray  # Senses embedded in spectral space


@dataclass
class MergerResult:
    """
    Result of embedding merger for a single word.
    """
    word: str
    sense_components: List[SenseComponent]
    similarity_matrix: np.ndarray
    clusters: Dict[str, int]  # sense_id -> cluster_label
    analysis: Dict[int, Dict]  # cluster_label -> analysis
    pairwise_stats: Dict[Tuple[str, str], Dict]
    threshold_used: float
    clustering_method: str = "hierarchical"
    spectral_info: Optional[SpectralInfo] = None
    
    @property
    def n_clusters(self) -> int:
        return len(set(self.clusters.values()))
    
    @property
    def n_convergent(self) -> int:
        """Count clusters containing senses from multiple sources."""
        return sum(1 for info in self.analysis.values() if info.get("is_convergent", False))
    
    @property
    def n_source_specific(self) -> int:
        return self.n_clusters - self.n_convergent
    
    def get_convergent_senses(self) -> List[Dict]:
        """Get information about convergent clusters."""
        return [info for info in self.analysis.values() if info.get("is_convergent", False)]
    
    def get_source_specific_senses(self) -> List[Dict]:
        """Get information about source-specific clusters."""
        return [info for info in self.analysis.values() if not info.get("is_convergent", False)]


class ClusteringMethod(Enum):
    """Available clustering methods for merger."""
    HIERARCHICAL = "hierarchical"
    SPECTRAL = "spectral"
    SPECTRAL_HIERARCHICAL = "spectral_hierarchical"  # Hybrid (recommended)


# =============================================================================
# SPECTRAL ANALYSIS FUNCTIONS
# =============================================================================

def compute_laplacian_spectrum(
    similarity_matrix: np.ndarray,
    n_components: int = None,
    normalized: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of the graph Laplacian.
    
    Args:
        similarity_matrix: Symmetric similarity matrix (n x n)
        n_components: Number of eigenvectors to compute (default: all)
        normalized: Use normalized Laplacian (recommended)
        
    Returns:
        (eigenvalues, eigenvectors)
    """
    n = similarity_matrix.shape[0]
    if n_components is None:
        n_components = n
    
    # Ensure non-negative similarities
    S = np.maximum(similarity_matrix, 0)
    np.fill_diagonal(S, 0)  # No self-loops
    
    # Degree matrix
    D = np.diag(S.sum(axis=1))
    
    if normalized:
        # Normalized Laplacian: L = I - D^{-1/2} S D^{-1/2}
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
        L = np.eye(n) - D_inv_sqrt @ S @ D_inv_sqrt
    else:
        # Unnormalized Laplacian: L = D - S
        L = D - S
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = eigh(L)
    
    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]
    
    return eigenvalues, eigenvectors


def find_k_by_eigengap(
    eigenvalues: np.ndarray,
    min_k: int = 2,
    max_k: int = None
) -> Tuple[int, np.ndarray]:
    """
    Find optimal number of clusters using eigengap heuristic.
    
    The eigengap is the difference between consecutive eigenvalues.
    A large gap indicates a natural clustering boundary.
    
    Args:
        eigenvalues: Sorted eigenvalues from Laplacian
        min_k: Minimum number of clusters
        max_k: Maximum number of clusters
        
    Returns:
        (optimal_k, eigengaps)
    """
    n = len(eigenvalues)
    if max_k is None:
        max_k = min(n - 1, 10)
    
    # Compute gaps
    gaps = np.diff(eigenvalues)
    
    # Find largest gap in valid range
    valid_range = range(min_k - 1, min(max_k, len(gaps)))
    if len(valid_range) == 0:
        return min_k, gaps
    
    best_idx = max(valid_range, key=lambda i: gaps[i])
    optimal_k = best_idx + 1  # +1 because gap[i] is between eigenvalue[i] and eigenvalue[i+1]
    
    return optimal_k, gaps


def spectral_embedding_from_similarity(
    similarity_matrix: np.ndarray,
    n_components: int = None
) -> Tuple[np.ndarray, SpectralInfo]:
    """
    Embed senses into spectral space based on similarity matrix.
    
    Args:
        similarity_matrix: Pairwise similarity matrix
        n_components: Dimensions of spectral embedding (default: auto via eigengap)
        
    Returns:
        (spectral_coordinates, SpectralInfo)
    """
    n = similarity_matrix.shape[0]
    
    # Compute Laplacian spectrum
    eigenvalues, eigenvectors = compute_laplacian_spectrum(
        similarity_matrix, 
        n_components=min(n, 10)
    )
    
    # Find optimal k via eigengap
    suggested_k, eigengaps = find_k_by_eigengap(eigenvalues, min_k=2, max_k=n-1)
    
    # Use k eigenvectors for embedding (skip first trivial eigenvector)
    if n_components is None:
        n_components = suggested_k
    
    # Spectral coordinates: rows of eigenvector matrix (excluding first)
    # First eigenvector is constant for connected graphs
    spectral_coords = eigenvectors[:, 1:n_components+1]
    
    # Normalize rows (each point on unit sphere)
    row_norms = np.linalg.norm(spectral_coords, axis=1, keepdims=True)
    spectral_coords = spectral_coords / (row_norms + 1e-10)
    
    info = SpectralInfo(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        eigengaps=eigengaps,
        suggested_k=suggested_k,
        spectral_coords=spectral_coords
    )
    
    return spectral_coords, info


# =============================================================================
# MAIN CLASS
# =============================================================================

class EmbeddingMerger:
    """
    Merge word embeddings from multiple sources via sense alignment.
    
    This class implements the embedding merger algorithm:
    1. Takes sense components from multiple embeddings (via SSR or other methods)
    2. Constructs shared semantic basis for each sense pair
    3. Projects senses onto shared basis for comparison
    4. Clusters to identify convergent vs source-specific senses
    
    Clustering Methods:
        - 'hierarchical': Standard agglomerative clustering (original)
        - 'spectral': Pure spectral clustering with eigengap k-selection
        - 'spectral_hierarchical': Hybrid - spectral embedding + hierarchical
                                   clustering (recommended for wave-like senses)
    
    Example:
        ```python
        merger = EmbeddingMerger(
            clustering_method='spectral_hierarchical',
            verbose=True
        )
        merger.add_embedding("wiki", wiki_vectors)
        merger.add_embedding("twitter", twitter_vectors)
        
        result = merger.merge_senses("bank", n_senses=3)
        print(f"Spectral suggested k={result.spectral_info.suggested_k}")
        ```
    """
    
    def __init__(
        self,
        neighbor_k: int = 50,
        max_basis_size: int = 40,
        default_threshold: float = 0.05,
        clustering_method: str = 'spectral_hierarchical',
        verbose: bool = False
    ):
        """
        Initialize the embedding merger.
        
        Args:
            neighbor_k: Number of neighbors to consider for basis construction
            max_basis_size: Maximum size of merger basis
            default_threshold: Default distance threshold for clustering
            clustering_method: 'hierarchical', 'spectral', or 'spectral_hierarchical'
            verbose: Print progress information
        """
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.neighbor_k = neighbor_k
        self.max_basis_size = max_basis_size
        self.default_threshold = default_threshold
        self.verbose = verbose
        
        # Validate clustering method
        valid_methods = ['hierarchical', 'spectral', 'spectral_hierarchical']
        if clustering_method not in valid_methods:
            raise ValueError(f"clustering_method must be one of {valid_methods}")
        self.clustering_method = clustering_method
        
        self._shared_vocab: Optional[Set[str]] = None
    
    def add_embedding(
        self, 
        name: str, 
        vectors: Dict[str, np.ndarray],
        normalize: bool = True,
        align_dimensions: bool = True
    ) -> 'EmbeddingMerger':
        """
        Add an embedding source for merger.
        
        Args:
            name: Identifier for this embedding (e.g., "wikipedia", "twitter")
            vectors: Dictionary mapping words to vectors
            normalize: Whether to L2-normalize vectors
            align_dimensions: If True, align to first embedding's dimension
            
        Returns:
            self (for chaining)
        """
        # Get dimension of this embedding
        sample_vec = next(iter(vectors.values()))
        this_dim = len(sample_vec)
        
        # Determine target dimension (first embedding sets it)
        if not hasattr(self, '_target_dim') or self._target_dim is None:
            self._target_dim = this_dim
        
        target_dim = self._target_dim
        
        # Align dimensions if needed
        if align_dimensions and this_dim != target_dim:
            if self.verbose:
                print(f"  Aligning '{name}' from {this_dim}d to {target_dim}d")
            
            aligned = {}
            for word, vec in vectors.items():
                if this_dim > target_dim:
                    # Truncate
                    aligned[word] = vec[:target_dim]
                else:
                    # Pad with zeros
                    padded = np.zeros(target_dim)
                    padded[:this_dim] = vec
                    aligned[word] = padded
            vectors = aligned
        
        if normalize:
            normalized = {}
            for word, vec in vectors.items():
                norm = np.linalg.norm(vec)
                normalized[word] = vec / norm if norm > 0 else vec
            self.embeddings[name] = normalized
        else:
            self.embeddings[name] = vectors
        
        # Invalidate cached shared vocabulary
        self._shared_vocab = None
        
        if self.verbose:
            print(f"Added embedding '{name}': {len(vectors)} words, {target_dim}d")
        
        return self
    
    @classmethod
    def from_sense_explorers(
        cls,
        explorers: Dict[str, 'SenseExplorer'],
        **kwargs
    ) -> 'EmbeddingMerger':
        """
        Create merger from multiple SenseExplorer instances.
        
        Args:
            explorers: Dict mapping names to SenseExplorer instances
            **kwargs: Additional arguments for EmbeddingMerger
            
        Returns:
            Configured EmbeddingMerger
        """
        merger = cls(**kwargs)
        for name, se in explorers.items():
            merger.add_embedding(name, se.embeddings)
        return merger
    
    @property
    def shared_vocabulary(self) -> Set[str]:
        """Get vocabulary present in all embeddings."""
        if self._shared_vocab is None:
            if not self.embeddings:
                return set()
            
            vocab_sets = [set(emb.keys()) for emb in self.embeddings.values()]
            self._shared_vocab = set.intersection(*vocab_sets)
            
            # Filter to alphabetic words
            self._shared_vocab = {w for w in self._shared_vocab 
                                  if w.isalpha() and len(w) > 1}
        
        return self._shared_vocab
    
    @property
    def n_embeddings(self) -> int:
        return len(self.embeddings)
    
    # =========================================================================
    # SENSE EXTRACTION
    # =========================================================================
    
    def extract_senses_simple(
        self,
        word: str,
        source: str,
        n_senses: int = 3,
        n_neighbors: int = 30
    ) -> List[SenseComponent]:
        """
        Simple sense extraction using k-means on neighbor space.
        
        This is a fallback when SSR is not available. For best results,
        use SenseExplorer's induce_senses() method instead.
        """
        from sklearn.cluster import KMeans
        
        vectors = self.embeddings[source]
        
        if word not in vectors:
            if self.verbose:
                print(f"  Warning: '{word}' not in {source}")
            return []
        
        target_vec = vectors[word]
        
        # Get neighbors
        all_words = [w for w in vectors.keys() if w != word]
        similarities = []
        for w in all_words:
            vec = vectors[w]
            sim = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-10)
            similarities.append((w, sim))
        
        similarities.sort(key=lambda x: -x[1])
        top_neighbors = similarities[:n_neighbors]
        neighbor_words = [w for w, _ in top_neighbors]
        neighbor_vecs = np.array([vectors[w] for w in neighbor_words])
        
        # Cluster neighbors
        n_actual = min(n_senses, len(neighbor_words))
        if n_actual < 2:
            return []
        
        kmeans = KMeans(n_clusters=n_actual, random_state=42, n_init=10)
        labels = kmeans.fit_predict(neighbor_vecs)
        
        senses = []
        for sense_idx in range(n_actual):
            mask = labels == sense_idx
            sense_neighbors = [neighbor_words[i] for i in range(len(neighbor_words)) if mask[i]]
            
            if not sense_neighbors:
                continue
            
            # Sense vector = centroid of cluster
            sense_vec = np.mean([vectors[w] for w in sense_neighbors], axis=0)
            
            # Compute neighbor similarities
            neighbors_with_sim = []
            for w in sense_neighbors:
                sim = np.dot(sense_vec, vectors[w]) / (np.linalg.norm(sense_vec) * np.linalg.norm(vectors[w]) + 1e-10)
                neighbors_with_sim.append((w, float(sim)))
            neighbors_with_sim.sort(key=lambda x: -x[1])
            
            senses.append(SenseComponent(
                word=word,
                sense_id=f"{source}_{word}_s{sense_idx}",
                vector=sense_vec,
                source=source,
                top_neighbors=neighbors_with_sim
            ))
        
        return senses
    
    # =========================================================================
    # CORE MERGER ALGORITHM
    # =========================================================================
    
    def compute_neighborhood(
        self,
        sense: SenseComponent,
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """Compute or return neighborhood for a sense component."""
        if sense.top_neighbors and len(sense.top_neighbors) >= (top_k or self.neighbor_k):
            return sense.top_neighbors[:top_k or self.neighbor_k]
        
        # Compute neighborhood
        vectors = self.embeddings[sense.source]
        similarities = []
        
        for word, vec in vectors.items():
            if word == sense.word:
                continue
            sim = np.dot(sense.vector, vec)
            similarities.append((word, float(sim)))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k or self.neighbor_k]
    
    def construct_merger_basis(
        self,
        sense_u: SenseComponent,
        sense_v: SenseComponent,
        neighbors_u: List[Tuple[str, float]] = None,
        neighbors_v: List[Tuple[str, float]] = None
    ) -> MergerBasis:
        """
        Construct the merger basis for aligning two sense components.
        
        The basis consists of:
        - Core: words in both neighborhoods (where embeddings agree)
        - Extensions: words unique to each neighborhood
        """
        if neighbors_u is None:
            neighbors_u = self.compute_neighborhood(sense_u)
        if neighbors_v is None:
            neighbors_v = self.compute_neighborhood(sense_v)
        
        # Get neighbor word sets
        words_u = {w for w, _ in neighbors_u}
        words_v = {w for w, _ in neighbors_v}
        
        # Filter to shared vocabulary
        shared = self.shared_vocabulary
        words_u_shared = words_u & shared
        words_v_shared = words_v & shared
        
        # Core: intersection
        core = words_u_shared & words_v_shared
        
        # Extensions
        ext_u = words_u_shared - core
        ext_v = words_v_shared - core
        
        # Build basis: core first, then interleaved extensions
        basis = list(core)
        ext_u_list = list(ext_u)
        ext_v_list = list(ext_v)
        
        i, j = 0, 0
        while len(basis) < self.max_basis_size and (i < len(ext_u_list) or j < len(ext_v_list)):
            if i < len(ext_u_list):
                basis.append(ext_u_list[i])
                i += 1
            if len(basis) < self.max_basis_size and j < len(ext_v_list):
                basis.append(ext_v_list[j])
                j += 1
        
        union_size = len(words_u_shared | words_v_shared)
        overlap_ratio = len(core) / union_size if union_size > 0 else 0
        
        return MergerBasis(
            core_words=sorted(core),
            extension_u=list(ext_u),
            extension_v=list(ext_v),
            all_words=basis,
            overlap_ratio=overlap_ratio
        )
    
    def project_sense_onto_basis(
        self,
        sense: SenseComponent,
        basis_words: List[str]
    ) -> Tuple[np.ndarray, float]:
        """
        Project a sense vector onto a vocabulary basis.
        
        Returns (projection_vector, quality_score)
        """
        vectors = self.embeddings[sense.source]
        projection = []
        
        for word in basis_words:
            if word in vectors:
                vec = vectors[word]
                sim = np.dot(sense.vector, vec)
                projection.append(sim)
            else:
                projection.append(0.0)
        
        projection = np.array(projection)
        
        # Quality = normalized magnitude
        quality = np.linalg.norm(projection) / (len(projection) ** 0.5 + 1e-10)
        
        return projection, quality
    
    def compute_merged_similarity(
        self,
        sense_u: SenseComponent,
        sense_v: SenseComponent
    ) -> Tuple[float, Dict]:
        """
        Compute similarity between two senses in the merged space.
        
        Returns (similarity, stats_dict)
        """
        # Compute neighborhoods
        neighbors_u = self.compute_neighborhood(sense_u)
        neighbors_v = self.compute_neighborhood(sense_v)
        
        # Construct merger basis
        basis = self.construct_merger_basis(sense_u, sense_v, neighbors_u, neighbors_v)
        
        if len(basis.all_words) < 3:
            return 0.0, {"basis_size": 0, "core_overlap": 0}
        
        # Project both senses
        proj_u, qual_u = self.project_sense_onto_basis(sense_u, basis.all_words)
        proj_v, qual_v = self.project_sense_onto_basis(sense_v, basis.all_words)
        
        # Compute similarity
        norm_u = np.linalg.norm(proj_u)
        norm_v = np.linalg.norm(proj_v)
        
        if norm_u < 1e-10 or norm_v < 1e-10:
            return 0.0, {"basis_size": len(basis.all_words), "core_overlap": basis.core_size}
        
        similarity = np.dot(proj_u, proj_v) / (norm_u * norm_v)
        
        stats = {
            "basis_size": len(basis.all_words),
            "core_overlap": basis.core_size,
            "overlap_ratio": basis.overlap_ratio,
            "core_words": basis.core_words[:10],
            "similarity": float(similarity)
        }
        
        return float(similarity), stats
    
    # =========================================================================
    # CLUSTERING METHODS
    # =========================================================================
    
    def _cluster_hierarchical(
        self,
        senses: List[SenseComponent],
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, Optional[SpectralInfo]]:
        """Standard hierarchical clustering."""
        dist_matrix = 1 - similarity_matrix
        dist_matrix = np.clip(dist_matrix, 0, 2)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="average"
        )
        
        labels = clustering.fit_predict(dist_matrix)
        return labels, None
    
    def _cluster_spectral(
        self,
        senses: List[SenseComponent],
        similarity_matrix: np.ndarray,
        n_clusters: int = None
    ) -> Tuple[np.ndarray, SpectralInfo]:
        """Pure spectral clustering with eigengap k-selection."""
        # Get spectral embedding
        spectral_coords, info = spectral_embedding_from_similarity(similarity_matrix)
        
        # Use suggested k or provided n_clusters
        k = n_clusters if n_clusters else info.suggested_k
        k = min(k, len(senses) - 1)  # Can't have more clusters than senses
        k = max(k, 2)  # At least 2 clusters
        
        # K-means on spectral coordinates
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(spectral_coords)
        
        return labels, info
    
    def _cluster_spectral_hierarchical(
        self,
        senses: List[SenseComponent],
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[np.ndarray, SpectralInfo]:
        """
        Hybrid: Spectral embedding + Hierarchical clustering.
        
        This combines:
        - Spectral's wave-aware similarity computation
        - Hierarchical's tree structure and threshold-based cutting
        """
        # Get spectral embedding
        spectral_coords, info = spectral_embedding_from_similarity(similarity_matrix)
        
        # Compute distances in spectral space
        n = len(senses)
        spectral_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance in spectral space
                d = np.linalg.norm(spectral_coords[i] - spectral_coords[j])
                spectral_dist[i, j] = d
                spectral_dist[j, i] = d
        
        # Normalize distances to [0, 1] range for threshold compatibility
        max_dist = np.max(spectral_dist)
        if max_dist > 0:
            spectral_dist = spectral_dist / max_dist
        
        # Hierarchical clustering on spectral distances
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="average"
        )
        
        labels = clustering.fit_predict(spectral_dist)
        
        # Update spectral_info with spectral distance matrix
        info.spectral_coords = spectral_coords
        
        return labels, info
    
    def _cluster_and_analyze(
        self,
        senses: List[SenseComponent],
        similarity_matrix: np.ndarray,
        threshold: float,
        n_clusters: int = None
    ) -> Tuple[Dict[str, int], Dict[int, Dict], Optional[SpectralInfo]]:
        """Cluster senses and analyze results."""
        
        # Select clustering method
        if self.clustering_method == 'hierarchical':
            labels, spectral_info = self._cluster_hierarchical(
                senses, similarity_matrix, threshold
            )
        elif self.clustering_method == 'spectral':
            labels, spectral_info = self._cluster_spectral(
                senses, similarity_matrix, n_clusters
            )
        elif self.clustering_method == 'spectral_hierarchical':
            labels, spectral_info = self._cluster_spectral_hierarchical(
                senses, similarity_matrix, threshold
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Build clusters dict
        clusters = {s.sense_id: int(labels[i]) for i, s in enumerate(senses)}
        
        # Analyze clusters
        cluster_members = defaultdict(list)
        for sense, label in zip(senses, labels):
            cluster_members[int(label)].append(sense)
        
        analysis = {}
        for cluster_id, members in cluster_members.items():
            sources = set(m.source for m in members)
            is_convergent = len(sources) > 1
            
            member_info = []
            for m in members:
                neighbors = [w for w, _ in m.top_neighbors[:5]] if m.top_neighbors else []
                member_info.append({
                    "sense_id": m.sense_id,
                    "source": m.source,
                    "neighbors": neighbors
                })
            
            analysis[cluster_id] = {
                "members": [m.sense_id for m in members],
                "sources": list(sources),
                "is_convergent": is_convergent,
                "member_info": member_info,
                "size": len(members)
            }
        
        return clusters, analysis, spectral_info
    
    # =========================================================================
    # MAIN MERGE FUNCTION
    # =========================================================================
    
    def merge_senses(
        self,
        word: str,
        sense_components: List[SenseComponent] = None,
        n_senses: int = 3,
        distance_threshold: float = None,
        n_clusters: int = None,
        return_all_thresholds: bool = False,
        thresholds: List[float] = None
    ) -> Union[MergerResult, Dict[float, MergerResult]]:
        """
        Merge senses of a word across all loaded embeddings.
        
        Args:
            word: Target word to merge
            sense_components: Pre-extracted sense components (optional)
            n_senses: Number of senses to extract per embedding (if not provided)
            distance_threshold: Clustering threshold (default: self.default_threshold)
            n_clusters: For spectral clustering, override eigengap k-selection
            return_all_thresholds: If True, return results for multiple thresholds
            thresholds: List of thresholds to test (if return_all_thresholds=True)
            
        Returns:
            MergerResult or Dict[float, MergerResult] if return_all_thresholds=True
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for clustering")
        
        if len(self.embeddings) < 2:
            raise ValueError("Need at least 2 embeddings for merger")
        
        if distance_threshold is None:
            distance_threshold = self.default_threshold
        
        # Extract senses if not provided
        if sense_components is None:
            sense_components = []
            for source in self.embeddings:
                if word in self.embeddings[source]:
                    senses = self.extract_senses_simple(word, source, n_senses)
                    sense_components.extend(senses)
                    if self.verbose:
                        print(f"  {source}: {len(senses)} senses extracted")
        
        if len(sense_components) < 2:
            raise ValueError(f"Need at least 2 sense components, got {len(sense_components)}")
        
        if self.verbose:
            print(f"\n[EMBEDDING MERGER] word='{word}'")
            print(f"  Clustering method: {self.clustering_method}")
            print(f"  Sense components: {len(sense_components)}")
            print(f"  Shared vocabulary: {len(self.shared_vocabulary)}")
        
        # Compute pairwise similarities
        n = len(sense_components)
        similarity_matrix = np.eye(n)
        pairwise_stats = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                sim, stats = self.compute_merged_similarity(
                    sense_components[i], 
                    sense_components[j]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
                
                sid_i = sense_components[i].sense_id
                sid_j = sense_components[j].sense_id
                pairwise_stats[(sid_i, sid_j)] = stats
        
        if self.verbose:
            # Report spectral analysis if using spectral methods
            if self.clustering_method in ['spectral', 'spectral_hierarchical']:
                _, info = spectral_embedding_from_similarity(similarity_matrix)
                print(f"  Spectral analysis: suggested k={info.suggested_k}")
                print(f"  Top eigengaps: {info.eigengaps[:5].round(3)}")
            
            # Compute cross-source stats
            cross_sims = []
            within_sims = []
            for i in range(n):
                for j in range(i + 1, n):
                    if sense_components[i].source != sense_components[j].source:
                        cross_sims.append(similarity_matrix[i, j])
                    else:
                        within_sims.append(similarity_matrix[i, j])
            
            if cross_sims:
                print(f"  Cross-source similarity: mean={np.mean(cross_sims):.3f}, max={np.max(cross_sims):.3f}")
            if within_sims:
                print(f"  Within-source similarity: mean={np.mean(within_sims):.3f}")
        
        # Cluster at specified threshold(s)
        if return_all_thresholds:
            if thresholds is None:
                thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
            
            results = {}
            for thresh in thresholds:
                clusters, analysis, spectral_info = self._cluster_and_analyze(
                    sense_components, similarity_matrix, thresh, n_clusters
                )
                results[thresh] = MergerResult(
                    word=word,
                    sense_components=sense_components,
                    similarity_matrix=similarity_matrix,
                    clusters=clusters,
                    analysis=analysis,
                    pairwise_stats=pairwise_stats,
                    threshold_used=thresh,
                    clustering_method=self.clustering_method,
                    spectral_info=spectral_info
                )
            return results
        else:
            clusters, analysis, spectral_info = self._cluster_and_analyze(
                sense_components, similarity_matrix, distance_threshold, n_clusters
            )
            return MergerResult(
                word=word,
                sense_components=sense_components,
                similarity_matrix=similarity_matrix,
                clusters=clusters,
                analysis=analysis,
                pairwise_stats=pairwise_stats,
                threshold_used=distance_threshold,
                clustering_method=self.clustering_method,
                spectral_info=spectral_info
            )
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def report(self, result: MergerResult) -> str:
        """Generate a text report of merger results."""
        lines = ["=" * 60]
        lines.append(f"EMBEDDING MERGER RESULTS: '{result.word}'")
        lines.append("=" * 60)
        lines.append(f"Clustering method: {result.clustering_method}")
        lines.append(f"Threshold: {result.threshold_used}")
        lines.append(f"Clusters: {result.n_clusters} ({result.n_convergent} convergent, {result.n_source_specific} source-specific)")
        
        if result.spectral_info:
            lines.append(f"Spectral suggested k: {result.spectral_info.suggested_k}")
            lines.append(f"Top eigengaps: {result.spectral_info.eigengaps[:4].round(3)}")
        
        for cluster_id, info in sorted(result.analysis.items()):
            lines.append(f"\n[Cluster {cluster_id}]")
            lines.append("-" * 40)
            
            for source in sorted(info["sources"]):
                member_ids = [m for m in info["members"] if m.startswith(source)]
                lines.append(f"  {source}: {', '.join(member_ids)}")
            
            lines.append("  Neighbors:")
            for mi in info["member_info"]:
                lines.append(f"    [{mi['source']}] {', '.join(mi['neighbors'])}")
            
            if info["is_convergent"]:
                lines.append(f"  → CONVERGENT")
            else:
                lines.append(f"  → SOURCE-SPECIFIC")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# =============================================================================
# INTEGRATION WITH SENSEEXPLORER
# =============================================================================

def create_merger_from_explorers(
    explorers: Dict[str, 'SenseExplorer'],
    **kwargs
) -> EmbeddingMerger:
    """
    Create an EmbeddingMerger from multiple SenseExplorer instances.
    
    Example:
        ```python
        se_wiki = SenseExplorer.from_file("wiki.txt")
        se_twitter = SenseExplorer.from_file("twitter.txt")
        
        merger = create_merger_from_explorers({
            "wikipedia": se_wiki,
            "twitter": se_twitter
        }, clustering_method='spectral_hierarchical')
        
        result = merger.merge_senses("bank")
        ```
    """
    return EmbeddingMerger.from_sense_explorers(explorers, **kwargs)


def merge_with_ssr(
    explorers: Dict[str, 'SenseExplorer'],
    word: str,
    n_senses: int = None,
    distance_threshold: float = 0.05,
    clustering_method: str = 'spectral_hierarchical',
    verbose: bool = True
) -> MergerResult:
    """
    Merge senses using SSR extraction from each SenseExplorer.
    
    This is the recommended way to merge embeddings — it uses
    SenseExplorer's SSR algorithm for sense extraction rather
    than simple k-means.
    
    Example:
        ```python
        result = merge_with_ssr(
            {"wiki": se_wiki, "twitter": se_twitter},
            "bank",
            n_senses=3,
            clustering_method='spectral_hierarchical'
        )
        print(f"Convergent: {result.n_convergent}")
        print(f"Suggested k: {result.spectral_info.suggested_k}")
        ```
    """
    # Extract senses from each explorer using SSR
    all_senses = []
    
    for name, se in explorers.items():
        if hasattr(se, 'induce_senses'):
            # Use SSR
            ssr_result = se.induce_senses(word, n_senses=n_senses)
            
            # Convert to SenseComponent format
            for sense_name, sense_vec in ssr_result.items():
                # Get neighbors
                neighbors = []
                for w in se.embeddings:
                    if w != word:
                        sim = np.dot(sense_vec, se.embeddings[w])
                        neighbors.append((w, float(sim)))
                neighbors.sort(key=lambda x: -x[1])
                
                all_senses.append(SenseComponent(
                    word=word,
                    sense_id=f"{name}_{sense_name}",
                    vector=sense_vec,
                    source=name,
                    top_neighbors=neighbors[:50]
                ))
        else:
            if verbose:
                print(f"Warning: {name} doesn't have induce_senses, using simple extraction")
    
    # Create merger and run
    merger = create_merger_from_explorers(
        explorers, 
        clustering_method=clustering_method,
        verbose=verbose
    )
    return merger.merge_senses(
        word,
        sense_components=all_senses,
        distance_threshold=distance_threshold
    )


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_merger_dendrogram(
    result: MergerResult,
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 8),
    show_threshold_lines: List[float] = None,
    use_spectral_distances: bool = True
):
    """
    Create a dendrogram visualization of merger results.
    
    Args:
        result: MergerResult from merge_senses()
        output_path: Path to save figure (optional)
        figsize: Figure size
        show_threshold_lines: Threshold values to mark with vertical lines
        use_spectral_distances: If True and spectral_info available, use spectral distances
    
    Requires matplotlib and scipy.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    # Decide which distance matrix to use
    if use_spectral_distances and result.spectral_info is not None:
        # Compute spectral distances
        coords = result.spectral_info.spectral_coords
        n = len(result.sense_components)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        # Normalize
        max_dist = np.max(dist_matrix)
        if max_dist > 0:
            dist_matrix = dist_matrix / max_dist
        title_suffix = " (spectral distances)"
    else:
        # Use similarity-based distances
        dist_matrix = 1 - result.similarity_matrix
        np.fill_diagonal(dist_matrix, 0)
        title_suffix = ""
    
    # Linkage
    dist_condensed = squareform(dist_matrix)
    linkage_matrix = linkage(dist_condensed, method='average')
    
    # Labels with source coloring
    labels = []
    colors = {}
    color_map = {}
    palette = ['#2E86AB', '#E94F37', '#76B041', '#F5A623', '#9B59B6']
    
    sources = list(set(s.source for s in result.sense_components))
    for i, source in enumerate(sources):
        color_map[source] = palette[i % len(palette)]
    
    for sense in result.sense_components:
        short = sense.sense_id.replace(f"{sense.source}_", "").replace(f"_{result.word}_", ":")
        labels.append(short)
        colors[short] = color_map[sense.source]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})
    
    # Dendrogram
    dendro = dendrogram(linkage_matrix, labels=labels, ax=ax1, orientation='left',
                        leaf_font_size=10, color_threshold=0)
    
    # Color labels
    for label in ax1.get_yticklabels():
        text = label.get_text()
        if text in colors:
            label.set_color(colors[text])
            label.set_fontweight('bold')
    
    # Threshold lines
    if show_threshold_lines:
        for thresh in show_threshold_lines:
            ax1.axvline(x=thresh, color='gray', linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Distance')
    ax1.set_title(f'Sense Hierarchy: "{result.word}"{title_suffix}')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[s], label=s) for s in sources]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Add spectral info if available
    if result.spectral_info:
        ax1.text(0.02, 0.98, f"Spectral k={result.spectral_info.suggested_k}",
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Heatmap
    dendro_order = dendro['leaves']
    reordered = dist_matrix[dendro_order][:, dendro_order]
    reordered_labels = [labels[i] for i in dendro_order]
    
    im = ax2.imshow(reordered, cmap='RdYlBu_r', vmin=0, vmax=0.4)
    ax2.set_xticks(range(len(reordered_labels)))
    ax2.set_yticks(range(len(reordered_labels)))
    ax2.set_xticklabels(reordered_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(reordered_labels, fontsize=9)
    
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        text = label.get_text()
        if text in colors:
            label.set_color(colors[text])
    
    plt.colorbar(im, ax=ax2, label='Distance', shrink=0.8)
    ax2.set_title('Distance Matrix')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_spectral_analysis(
    result: MergerResult,
    output_path: str = None,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot spectral analysis: eigenvalues, eigengaps, and spectral embedding.
    
    Args:
        result: MergerResult with spectral_info
        output_path: Path to save figure
        figsize: Figure size
    """
    if result.spectral_info is None:
        raise ValueError("No spectral info available. Use clustering_method='spectral_hierarchical'")
    
    import matplotlib.pyplot as plt
    
    info = result.spectral_info
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Eigenvalues
    ax1 = axes[0]
    ax1.plot(range(1, len(info.eigenvalues) + 1), info.eigenvalues, 'bo-')
    ax1.axvline(x=info.suggested_k, color='r', linestyle='--', label=f'k={info.suggested_k}')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Laplacian Eigenvalues')
    ax1.legend()
    
    # Eigengaps
    ax2 = axes[1]
    ax2.bar(range(1, len(info.eigengaps) + 1), info.eigengaps, color='steelblue')
    ax2.axvline(x=info.suggested_k - 0.5, color='r', linestyle='--')
    ax2.set_xlabel('Gap Index')
    ax2.set_ylabel('Eigengap')
    ax2.set_title('Eigengaps (larger = cluster boundary)')
    
    # Spectral embedding (first 2 dimensions)
    ax3 = axes[2]
    coords = info.spectral_coords
    
    # Color by source
    sources = list(set(s.source for s in result.sense_components))
    palette = ['#2E86AB', '#E94F37', '#76B041', '#F5A623', '#9B59B6']
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(sources)}
    
    for i, sense in enumerate(result.sense_components):
        c = color_map[sense.source]
        if coords.shape[1] >= 2:
            ax3.scatter(coords[i, 0], coords[i, 1], c=c, s=100, edgecolors='black')
            ax3.annotate(sense.sense_id.split('_')[-1], (coords[i, 0], coords[i, 1]),
                        fontsize=8, ha='center', va='bottom')
        else:
            ax3.scatter(coords[i, 0], 0, c=c, s=100, edgecolors='black')
    
    ax3.set_xlabel('Spectral dim 1')
    ax3.set_ylabel('Spectral dim 2' if coords.shape[1] >= 2 else '')
    ax3.set_title(f'Spectral Embedding: "{result.word}"')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[s], label=s) for s in sources]
    ax3.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# WEIGHTED MERGING
# =============================================================================

@dataclass
class WeightedMergerResult(MergerResult):
    """
    MergerResult extended with weighting information.
    """
    source_weights: Dict[str, float] = field(default_factory=dict)
    sense_weights: Dict[str, float] = field(default_factory=dict)
    weighted_similarity_matrix: np.ndarray = None
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weight_summary(self) -> str:
        """Summarize weights used."""
        lines = ["Source Weights:"]
        for name, w in sorted(self.source_weights.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {w:.3f}")
        return "\n".join(lines)


def merge_with_weights(
    explorers: Dict[str, 'SenseExplorer'],
    word: str,
    n_senses: int = None,
    distance_threshold: float = 0.05,
    clustering_method: str = 'spectral_hierarchical',
    weight_config: Dict[str, float] = None,
    assessment_words: List[str] = None,
    verbose: bool = True
) -> WeightedMergerResult:
    """
    Merge senses with quality-based weighting.
    
    This is the recommended way to merge embeddings — it:
    1. Assesses quality of each embedding source
    2. Computes weights based on coherence, separation, vocabulary
    3. Applies weights to similarity computation
    4. Uses SSR for sense extraction
    
    Args:
        explorers: Dict mapping names to SenseExplorer instances
        word: Target word to merge
        n_senses: Number of senses per embedding (None = auto)
        distance_threshold: Clustering threshold
        clustering_method: 'hierarchical', 'spectral', or 'spectral_hierarchical'
        weight_config: Custom weights for quality components:
            {'vocab': 0.15, 'coherence': 0.35, 'separation': 0.35, 'overlap': 0.15}
        assessment_words: Words to use for quality assessment (default: auto)
        verbose: Print progress information
        
    Returns:
        WeightedMergerResult with weights and standard merger results
        
    Example:
        ```python
        result = merge_with_weights(
            {"wiki": se_wiki, "twitter": se_twitter},
            "bank",
            n_senses=3,
            weight_config={'coherence': 0.5, 'separation': 0.5}  # Emphasize semantic quality
        )
        print(f"Source weights: {result.source_weights}")
        print(f"Convergent: {result.n_convergent}")
        ```
    """
    # Import weighting module
    try:
        from .embedding_weights import (
            EmbeddingQualityAssessor,
            weight_similarity_matrix,
            compute_weighted_centroid
        )
    except ImportError:
        from embedding_weights import (
            EmbeddingQualityAssessor,
            weight_similarity_matrix,
            compute_weighted_centroid
        )
    
    # === STEP 1: Quality Assessment ===
    if verbose:
        print("=" * 60)
        print(f"WEIGHTED MERGE: '{word}'")
        print("=" * 60)
    
    # Configure assessor
    if weight_config is None:
        weight_config = {
            'vocab': 0.15,
            'coherence': 0.35,
            'separation': 0.35,
            'overlap': 0.15
        }
    
    assessor = EmbeddingQualityAssessor(
        vocab_weight=weight_config.get('vocab', 0.15),
        coherence_weight=weight_config.get('coherence', 0.35),
        separation_weight=weight_config.get('separation', 0.35),
        overlap_weight=weight_config.get('overlap', 0.15),
        verbose=verbose
    )
    
    # Add embeddings and explorers
    for name, se in explorers.items():
        assessor.add_embedding(name, se.embeddings, se)
    
    # Select assessment words
    if assessment_words is None:
        # Use target word plus some shared vocabulary
        shared = list(assessor.shared_vocabulary)
        assessment_words = [word] + [w for w in shared[:20] if w != word]
    
    # Run assessment
    assessor.assess_all(assessment_words)
    source_weights = assessor.get_weights()
    
    if verbose:
        print("\n" + "-" * 40)
        print("QUALITY-BASED WEIGHTS")
        print("-" * 40)
        for name, w in sorted(source_weights.items(), key=lambda x: -x[1]):
            print(f"  {name}: {w:.3f}")
    
    # === STEP 2: Extract Senses with SSR ===
    all_senses = []
    sense_source_map = {}  # sense_id -> source name
    target_dim = None  # Will be set by first embedding
    
    for name, se in explorers.items():
        if hasattr(se, 'induce_senses'):
            ssr_result = se.induce_senses(word, n_senses=n_senses)
            
            for sense_name, sense_vec in ssr_result.items():
                sense_id = f"{name}_{sense_name}"
                
                # Align dimension if needed
                if target_dim is None:
                    target_dim = len(sense_vec)
                elif len(sense_vec) != target_dim:
                    if verbose:
                        print(f"    Aligning {sense_id} from {len(sense_vec)}d to {target_dim}d")
                    if len(sense_vec) > target_dim:
                        sense_vec = sense_vec[:target_dim]
                    else:
                        padded = np.zeros(target_dim)
                        padded[:len(sense_vec)] = sense_vec
                        sense_vec = padded
                    # Re-normalize after alignment
                    norm = np.linalg.norm(sense_vec)
                    if norm > 0:
                        sense_vec = sense_vec / norm
                
                # Get neighbors (using original embedding for neighbor computation)
                neighbors = []
                for w in se.embeddings:
                    if w != word:
                        # Use original vectors for neighbor similarity
                        orig_vec = ssr_result[sense_name]
                        sim = np.dot(orig_vec, se.embeddings[w])
                        neighbors.append((w, float(sim)))
                neighbors.sort(key=lambda x: -x[1])
                
                all_senses.append(SenseComponent(
                    word=word,
                    sense_id=sense_id,
                    vector=sense_vec,
                    source=name,
                    top_neighbors=neighbors[:50]
                ))
                sense_source_map[sense_id] = name
                
            if verbose:
                print(f"  {name}: {len(ssr_result)} senses extracted (weight={source_weights[name]:.3f})")
        else:
            if verbose:
                print(f"  Warning: {name} doesn't have induce_senses")
    
    if len(all_senses) < 2:
        raise ValueError(f"Need at least 2 sense components, got {len(all_senses)}")
    
    # === STEP 3: Compute Sense-Level Weights ===
    sense_weights = {}
    for sc in all_senses:
        # Base weight from source
        base_weight = source_weights.get(sc.source, 0.5)
        
        # Could add sense-specific quality here (e.g., from anchor coherence)
        # For now, use source weight directly
        sense_weights[sc.sense_id] = base_weight
    
    # Normalize sense weights
    total_sw = sum(sense_weights.values())
    if total_sw > 0:
        sense_weights = {k: v / total_sw for k, v in sense_weights.items()}
    
    # === STEP 4: Compute Similarities via Basis Projection ===
    # Use the principled approach: project senses onto shared vocabulary basis
    # This maps both embeddings into a common space where dimensions correspond
    
    # Build shared vocabulary across all explorers
    shared_vocab = None
    for name, se in explorers.items():
        vocab = set(se.embeddings.keys())
        if shared_vocab is None:
            shared_vocab = vocab
        else:
            shared_vocab &= vocab
    
    shared_vocab_list = sorted(list(shared_vocab))
    
    if verbose:
        print(f"\n  Shared vocabulary for basis: {len(shared_vocab_list)} words")
    
    # Build embedding lookup (with dimension alignment already applied)
    embeddings_lookup = {}
    for name, se in explorers.items():
        embeddings_lookup[name] = se.embeddings
    
    def project_sense_to_basis(sense, basis_words, embeddings):
        """Project sense onto vocabulary basis (dimension-safe)."""
        vectors = embeddings[sense.source]
        projection = []
        
        for word in basis_words:
            if word in vectors:
                # Use source-specific embedding for projection
                # This handles dimension differences automatically
                word_vec = vectors[word]
                sense_vec = sense.vector
                
                # Ensure same dimension for dot product
                min_dim = min(len(sense_vec), len(word_vec))
                sim = np.dot(sense_vec[:min_dim], word_vec[:min_dim])
                projection.append(sim)
            else:
                projection.append(0.0)
        
        return np.array(projection)
    
    def compute_basis_similarity(sense_u, sense_v, embeddings):
        """Compute similarity between two senses via basis projection."""
        # Get neighborhoods for basis construction
        neighbors_u = [(w, s) for w, s in (sense_u.top_neighbors or [])[:50]]
        neighbors_v = [(w, s) for w, s in (sense_v.top_neighbors or [])[:50]]
        
        words_u = set(w for w, _ in neighbors_u) & shared_vocab
        words_v = set(w for w, _ in neighbors_v) & shared_vocab
        
        # Core = intersection, extensions = unique to each
        core = words_u & words_v
        ext_u = words_u - core
        ext_v = words_v - core
        
        # Build basis: core + interleaved extensions
        basis = list(core)
        ext_u_list = list(ext_u)
        ext_v_list = list(ext_v)
        
        max_basis = 100
        i, j = 0, 0
        while len(basis) < max_basis and (i < len(ext_u_list) or j < len(ext_v_list)):
            if i < len(ext_u_list):
                basis.append(ext_u_list[i])
                i += 1
            if j < len(ext_v_list) and len(basis) < max_basis:
                basis.append(ext_v_list[j])
                j += 1
        
        if len(basis) < 3:
            # Fallback to direct similarity if basis too small
            min_dim = min(len(sense_u.vector), len(sense_v.vector))
            return float(np.dot(sense_u.vector[:min_dim], sense_v.vector[:min_dim]))
        
        # Project both senses onto basis
        proj_u = project_sense_to_basis(sense_u, basis, embeddings)
        proj_v = project_sense_to_basis(sense_v, basis, embeddings)
        
        # Compute cosine similarity in projected space
        norm_u = np.linalg.norm(proj_u)
        norm_v = np.linalg.norm(proj_v)
        
        if norm_u < 1e-10 or norm_v < 1e-10:
            return 0.0
        
        return float(np.dot(proj_u, proj_v) / (norm_u * norm_v))
    
    n = len(all_senses)
    similarity_matrix = np.eye(n)
    pairwise_stats = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = compute_basis_similarity(
                all_senses[i], 
                all_senses[j], 
                embeddings_lookup
            )
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            
            sid_i = all_senses[i].sense_id
            sid_j = all_senses[j].sense_id
            
            pairwise_stats[(sid_i, sid_j)] = {
                'similarity': sim,
                'sources': (all_senses[i].source, all_senses[j].source),
                'cross_source': all_senses[i].source != all_senses[j].source
            }
    
    # === STEP 5: Apply Weights to Similarity Matrix ===
    weight_list = [sense_weights[sc.sense_id] for sc in all_senses]
    weighted_sim_matrix = weight_similarity_matrix(
        similarity_matrix,
        weight_list,
        method="symmetric"
    )
    
    if verbose:
        # Report similarity statistics
        cross_sims = []
        within_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                if all_senses[i].source != all_senses[j].source:
                    cross_sims.append(weighted_sim_matrix[i, j])
                else:
                    within_sims.append(weighted_sim_matrix[i, j])
        
        if cross_sims:
            print(f"\n  Cross-source (weighted): mean={np.mean(cross_sims):.3f}, max={np.max(cross_sims):.3f}")
        if within_sims:
            print(f"  Within-source (weighted): mean={np.mean(within_sims):.3f}")
    
    # === STEP 6: Cluster Using Weighted Similarities ===
    # Create minimal merger for clustering (doesn't need embeddings)
    merger = EmbeddingMerger(
        clustering_method=clustering_method,
        verbose=False
    )
    
    # Use the weighted similarity matrix for clustering
    clusters, analysis, spectral_info = merger._cluster_and_analyze(
        all_senses, weighted_sim_matrix, distance_threshold, n_clusters=None
    )
    
    if verbose:
        print(f"\n  Clusters: {len(set(clusters.values()))}")
        convergent = sum(1 for info in analysis.values() if info.get("is_convergent", False))
        print(f"  Convergent: {convergent}")
        if spectral_info:
            print(f"  Spectral suggested k: {spectral_info.suggested_k}")
    
    # === STEP 7: Build Result ===
    result = WeightedMergerResult(
        word=word,
        sense_components=all_senses,
        similarity_matrix=similarity_matrix,
        clusters=clusters,
        analysis=analysis,
        pairwise_stats=pairwise_stats,
        threshold_used=distance_threshold,
        clustering_method=clustering_method,
        spectral_info=spectral_info,
        source_weights=source_weights,
        sense_weights=sense_weights,
        weighted_similarity_matrix=weighted_sim_matrix,
        quality_assessment={
            'config': weight_config,
            'qualities': {name: {
                'vocab_score': q.vocab_score,
                'coherence_score': q.coherence_score,
                'separation_score': q.separation_score,
                'overlap_score': q.overlap_score,
                'composite': q.composite_score
            } for name, q in assessor.qualities.items()}
        }
    )
    
    return result


def weighted_report(result: WeightedMergerResult) -> str:
    """Generate a text report of weighted merger results."""
    lines = ["=" * 60]
    lines.append(f"WEIGHTED EMBEDDING MERGER: '{result.word}'")
    lines.append("=" * 60)
    
    # Weight summary
    lines.append("\nSOURCE WEIGHTS (quality-based):")
    for name, w in sorted(result.source_weights.items(), key=lambda x: -x[1]):
        lines.append(f"  {name}: {w:.3f}")
    
    # Quality breakdown
    if result.quality_assessment and 'qualities' in result.quality_assessment:
        lines.append("\nQUALITY COMPONENTS:")
        for name, q in result.quality_assessment['qualities'].items():
            lines.append(f"  {name}:")
            lines.append(f"    vocab={q['vocab_score']:.2f}, coherence={q['coherence_score']:.2f}, "
                        f"separation={q['separation_score']:.2f}, overlap={q['overlap_score']:.2f}")
    
    # Standard merger info
    lines.append(f"\nCLUSTERING:")
    lines.append(f"  Method: {result.clustering_method}")
    lines.append(f"  Threshold: {result.threshold_used}")
    lines.append(f"  Clusters: {result.n_clusters} ({result.n_convergent} convergent, {result.n_source_specific} source-specific)")
    
    if result.spectral_info:
        lines.append(f"  Spectral suggested k: {result.spectral_info.suggested_k}")
    
    # Cluster details
    lines.append("\nCLUSTER DETAILS:")
    for cluster_id, info in sorted(result.analysis.items()):
        lines.append(f"\n  [Cluster {cluster_id}] {'CONVERGENT' if info['is_convergent'] else 'SOURCE-SPECIFIC'}")
        
        for mi in info["member_info"]:
            weight = result.sense_weights.get(mi['sense_id'], 0)
            lines.append(f"    {mi['source']} (w={weight:.2f}): {', '.join(mi['neighbors'][:5])}")
    
    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Embedding Merger Module for SenseExplorer")
    print("=" * 50)
    print("\nClustering methods:")
    print("  - 'hierarchical': Standard agglomerative")
    print("  - 'spectral': Pure spectral with eigengap k-selection")
    print("  - 'spectral_hierarchical': Hybrid (recommended)")
    print("\nStandard usage:")
    print("  from sense_explorer.merger import EmbeddingMerger")
    print("  merger = EmbeddingMerger(clustering_method='spectral_hierarchical')")
    print("  merger.add_embedding('wiki', wiki_vectors)")
    print("  merger.add_embedding('twitter', twitter_vectors)")
    print("  result = merger.merge_senses('bank')")
    print("\nWeighted usage (recommended):")
    print("  from sense_explorer.merger import merge_with_weights")
    print("  result = merge_with_weights({'wiki': se_wiki, 'twitter': se_twitter}, 'bank')")
    print("  print(f'Source weights: {result.source_weights}')")
    print("  print(f'Convergent: {result.n_convergent}')")
