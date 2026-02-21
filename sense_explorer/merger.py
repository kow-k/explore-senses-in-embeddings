#!/usr/bin/env python3
"""
Embedding Merger Module for SenseExplorer
==========================================

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
    
    # Create merger
    merger = EmbeddingMerger()
    merger.add_embedding("wikipedia", se_wiki.embeddings)
    merger.add_embedding("twitter", se_twitter.embeddings)
    
    # Extract and merge senses
    result = merger.merge_senses("bank", sense_extractor=se_wiki.induce_senses)
    ```

Author: Kow Kuroda & Claude (Anthropic)
Version: 0.1.0 (for SenseExplorer v0.9.3)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import warnings

try:
    from sklearn.cluster import AgglomerativeClustering
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


class MergerMode(Enum):
    """Modes for handling sense extraction."""
    EXTERNAL = "external"  # Use provided sense components
    SIMPLE = "simple"  # Use simple k-means on neighborhoods
    SSR = "ssr"  # Use SenseExplorer's SSR method


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
    
    Example:
        ```python
        merger = EmbeddingMerger(verbose=True)
        merger.add_embedding("wiki", wiki_vectors)
        merger.add_embedding("twitter", twitter_vectors)
        
        # Simple mode (k-means sense extraction)
        result = merger.merge_senses("bank", n_senses=3)
        
        # With external sense components
        senses = [SenseComponent(...), ...]
        result = merger.merge_senses("bank", sense_components=senses)
        ```
    """
    
    def __init__(
        self,
        neighbor_k: int = 50,
        max_basis_size: int = 40,
        default_threshold: float = 0.05,
        verbose: bool = False
    ):
        """
        Initialize the embedding merger.
        
        Args:
            neighbor_k: Number of neighbors to consider for basis construction
            max_basis_size: Maximum size of merger basis
            default_threshold: Default distance threshold for clustering
            verbose: Print progress information
        """
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.neighbor_k = neighbor_k
        self.max_basis_size = max_basis_size
        self.default_threshold = default_threshold
        self.verbose = verbose
        
        self._shared_vocab: Optional[Set[str]] = None
    
    def add_embedding(
        self, 
        name: str, 
        vectors: Dict[str, np.ndarray],
        normalize: bool = True
    ) -> 'EmbeddingMerger':
        """
        Add an embedding source for merger.
        
        Args:
            name: Identifier for this embedding (e.g., "wikipedia", "twitter")
            vectors: Dictionary mapping words to vectors
            normalize: Whether to L2-normalize vectors
            
        Returns:
            self (for chaining)
        """
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
            print(f"Added embedding '{name}': {len(vectors)} words, {len(next(iter(vectors.values())))}d")
        
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
    # MAIN MERGE FUNCTION
    # =========================================================================
    
    def merge_senses(
        self,
        word: str,
        sense_components: List[SenseComponent] = None,
        n_senses: int = 3,
        distance_threshold: float = None,
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
                clusters, analysis = self._cluster_and_analyze(
                    sense_components, similarity_matrix, thresh
                )
                results[thresh] = MergerResult(
                    word=word,
                    sense_components=sense_components,
                    similarity_matrix=similarity_matrix,
                    clusters=clusters,
                    analysis=analysis,
                    pairwise_stats=pairwise_stats,
                    threshold_used=thresh
                )
            return results
        else:
            clusters, analysis = self._cluster_and_analyze(
                sense_components, similarity_matrix, distance_threshold
            )
            return MergerResult(
                word=word,
                sense_components=sense_components,
                similarity_matrix=similarity_matrix,
                clusters=clusters,
                analysis=analysis,
                pairwise_stats=pairwise_stats,
                threshold_used=distance_threshold
            )
    
    def _cluster_and_analyze(
        self,
        senses: List[SenseComponent],
        similarity_matrix: np.ndarray,
        threshold: float
    ) -> Tuple[Dict[str, int], Dict[int, Dict]]:
        """Cluster senses and analyze results."""
        # Convert to distance
        dist_matrix = 1 - similarity_matrix
        dist_matrix = np.clip(dist_matrix, 0, 2)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="precomputed",
            linkage="average"
        )
        
        labels = clustering.fit_predict(dist_matrix)
        
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
        
        return clusters, analysis
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def report(self, result: MergerResult) -> str:
        """Generate a text report of merger results."""
        lines = ["=" * 60]
        lines.append(f"EMBEDDING MERGER RESULTS: '{result.word}'")
        lines.append("=" * 60)
        lines.append(f"Threshold: {result.threshold_used}")
        lines.append(f"Clusters: {result.n_clusters} ({result.n_convergent} convergent, {result.n_source_specific} source-specific)")
        
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
        })
        
        result = merger.merge_senses("bank")
        ```
    """
    return EmbeddingMerger.from_sense_explorers(explorers, **kwargs)


def merge_with_ssr(
    explorers: Dict[str, 'SenseExplorer'],
    word: str,
    n_senses: int = None,
    distance_threshold: float = 0.05,
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
            n_senses=3
        )
        print(f"Convergent: {result.n_convergent}")
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
    merger = create_merger_from_explorers(explorers, verbose=verbose)
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
    show_threshold_lines: List[float] = None
):
    """
    Create a dendrogram visualization of merger results.
    
    Requires matplotlib and scipy.
    """
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    # Convert similarity to distance matrix
    dist_matrix = 1 - result.similarity_matrix
    np.fill_diagonal(dist_matrix, 0)
    
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
    
    ax1.set_xlabel('Distance (1 - similarity)')
    ax1.set_title(f'Sense Hierarchy: "{result.word}"')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[s], label=s) for s in sources]
    ax1.legend(handles=legend_elements, loc='lower right')
    
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


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Embedding Merger Module for SenseExplorer")
    print("=" * 50)
    print("\nUsage:")
    print("  from sense_explorer.merger import EmbeddingMerger")
    print("  merger = EmbeddingMerger()")
    print("  merger.add_embedding('wiki', wiki_vectors)")
    print("  merger.add_embedding('twitter', twitter_vectors)")
    print("  result = merger.merge_senses('bank')")
    print("\nOr with SenseExplorer integration:")
    print("  from sense_explorer.merger import merge_with_ssr")
    print("  result = merge_with_ssr({'wiki': se_wiki, 'twitter': se_twitter}, 'bank')")
