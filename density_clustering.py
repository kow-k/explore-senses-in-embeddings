"""
Density-Based Clustering Along the Main Line

Since all words satisfy sim + drift = 1, the entire space collapses to
a single dimension (similarity). This module finds natural clusters by
detecting gaps in the density distribution.

Algorithm:
    1. Project all words onto the similarity axis (0 to 1)
    2. For each candidate number of bins n:
       - Divide [0, 1] into n equal bins
       - Count words in each bin
       - Compute gap score (emptiness/sparsity of bins)
    3. Find optimal n that maximizes gap clarity
    4. Identify cluster boundaries at gap locations

Gap Metrics:
    - max_gap: Largest consecutive empty/sparse region
    - gap_ratio: Fraction of bins below threshold
    - variance: High variance = uneven distribution = gaps
    - jenks_gvf: Goodness of Variance Fit (Jenks Natural Breaks)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter


@dataclass
class BinAnalysis:
    """Analysis of a single binning configuration."""
    n_bins: int
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    bin_centers: np.ndarray
    
    # Gap metrics
    max_gap_width: float          # Width of largest empty region
    n_empty_bins: int             # Number of empty bins
    gap_ratio: float              # Fraction of empty bins
    count_variance: float         # Variance of bin counts
    count_cv: float               # Coefficient of variation
    
    # Identified gaps
    gap_locations: List[Tuple[float, float]]  # (start, end) of each gap
    
    def report(self) -> str:
        lines = [
            f"Bins: {self.n_bins}",
            f"  Empty bins: {self.n_empty_bins} ({self.gap_ratio*100:.1f}%)",
            f"  Max gap width: {self.max_gap_width:.3f}",
            f"  Count variance: {self.count_variance:.1f}",
            f"  Count CV: {self.count_cv:.3f}",
        ]
        if self.gap_locations:
            lines.append(f"  Gaps: {len(self.gap_locations)}")
            for start, end in self.gap_locations[:5]:
                lines.append(f"    [{start:.3f}, {end:.3f}]")
        return "\n".join(lines)


@dataclass
class DensityCluster:
    """A cluster identified by density analysis."""
    cluster_id: int
    start: float          # Lower bound (similarity)
    end: float            # Upper bound (similarity)
    center: float         # Center of mass
    n_words: int          # Number of words in cluster
    words: List[str]      # Words in this cluster
    density: float        # Words per unit similarity
    
    def __repr__(self):
        return f"Cluster({self.cluster_id}: [{self.start:.3f}, {self.end:.3f}], n={self.n_words})"


@dataclass
class ClusteringResult:
    """Result of density-based clustering."""
    clusters: List[DensityCluster]
    n_clusters: int
    optimal_n_bins: int
    
    # All bin analyses
    bin_analyses: Dict[int, BinAnalysis]
    
    # Gap detection scores by n_bins
    gap_scores: Dict[int, float]
    
    # Word assignments
    word_to_cluster: Dict[str, int]
    
    def report(self) -> str:
        lines = [
            "=" * 60,
            "DENSITY-BASED CLUSTERING RESULT",
            "=" * 60,
            f"Optimal n_bins: {self.optimal_n_bins}",
            f"N clusters: {self.n_clusters}",
            "",
            "Clusters:",
        ]
        for c in self.clusters:
            lines.append(f"  {c.cluster_id}: [{c.start:.3f}, {c.end:.3f}] "
                        f"n={c.n_words}, density={c.density:.1f}")
            # Show sample words
            sample = c.words[:5]
            if sample:
                lines.append(f"      e.g., {', '.join(sample)}")
        return "\n".join(lines)


def analyze_bins(
    similarities: np.ndarray,
    n_bins: int,
    sparse_threshold: float = 0.1
) -> BinAnalysis:
    """
    Analyze a specific binning configuration.
    
    Args:
        similarities: Array of similarity values (0 to 1)
        n_bins: Number of bins
        sparse_threshold: Fraction of mean count below which a bin is "sparse"
        
    Returns:
        BinAnalysis with gap metrics
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts, _ = np.histogram(similarities, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins
    
    # Find empty/sparse bins
    mean_count = np.mean(bin_counts)
    threshold = mean_count * sparse_threshold
    is_sparse = bin_counts <= threshold
    
    n_empty = np.sum(bin_counts == 0)
    n_sparse = np.sum(is_sparse)
    
    # Find gap locations (consecutive sparse bins)
    gap_locations = []
    in_gap = False
    gap_start = 0
    
    for i, sparse in enumerate(is_sparse):
        if sparse and not in_gap:
            in_gap = True
            gap_start = bin_edges[i]
        elif not sparse and in_gap:
            in_gap = False
            gap_end = bin_edges[i]
            gap_locations.append((gap_start, gap_end))
    
    # Handle gap at the end
    if in_gap:
        gap_locations.append((gap_start, bin_edges[-1]))
    
    # Compute max gap width
    if gap_locations:
        max_gap_width = max(end - start for start, end in gap_locations)
    else:
        max_gap_width = 0.0
    
    # Variance metrics
    count_variance = np.var(bin_counts)
    count_mean = np.mean(bin_counts)
    count_cv = np.std(bin_counts) / count_mean if count_mean > 0 else 0
    
    return BinAnalysis(
        n_bins=n_bins,
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        bin_centers=bin_centers,
        max_gap_width=max_gap_width,
        n_empty_bins=n_empty,
        gap_ratio=n_sparse / n_bins,
        count_variance=count_variance,
        count_cv=count_cv,
        gap_locations=gap_locations
    )


def compute_gap_score(analysis: BinAnalysis, method: str = "combined") -> float:
    """
    Compute a score indicating how well gaps separate clusters.
    
    Higher score = clearer gaps = better clustering.
    
    Args:
        analysis: BinAnalysis object
        method: Scoring method
            - "max_gap": Largest gap width
            - "total_gap": Sum of all gap widths
            - "cv": Coefficient of variation (higher = more uneven)
            - "combined": Weighted combination
            
    Returns:
        Gap score (higher = better)
    """
    if method == "max_gap":
        return analysis.max_gap_width
    
    elif method == "total_gap":
        return sum(end - start for start, end in analysis.gap_locations)
    
    elif method == "cv":
        return analysis.count_cv
    
    elif method == "combined":
        # Combine multiple signals
        # Normalize each to [0, 1] range approximately
        gap_score = analysis.max_gap_width  # Already in [0, 1]
        cv_score = min(1.0, analysis.count_cv / 2)  # CV > 2 is very high
        empty_score = analysis.gap_ratio
        
        # Weight: prioritize actual gaps over variance
        return 0.5 * gap_score + 0.3 * cv_score + 0.2 * empty_score
    
    else:
        raise ValueError(f"Unknown method: {method}")


def find_optimal_bins(
    similarities: np.ndarray,
    min_bins: int = 5,
    max_bins: int = 50,
    method: str = "combined",
    verbose: bool = False
) -> Tuple[int, Dict[int, BinAnalysis], Dict[int, float]]:
    """
    Find optimal number of bins that maximizes gap clarity.
    
    Args:
        similarities: Array of similarity values
        min_bins: Minimum bins to try
        max_bins: Maximum bins to try
        method: Gap scoring method
        verbose: Print progress
        
    Returns:
        Tuple of (optimal_n_bins, all_analyses, all_scores)
    """
    analyses = {}
    scores = {}
    
    for n in range(min_bins, max_bins + 1):
        analysis = analyze_bins(similarities, n)
        score = compute_gap_score(analysis, method)
        analyses[n] = analysis
        scores[n] = score
        
        if verbose and n % 10 == 0:
            print(f"  n={n}: score={score:.4f}, max_gap={analysis.max_gap_width:.3f}")
    
    # Find optimal (but prefer smaller n for equal scores - parsimony)
    best_n = max(scores.keys(), key=lambda n: (scores[n], -n))
    
    return best_n, analyses, scores


def extract_clusters(
    words: List[str],
    similarities: np.ndarray,
    analysis: BinAnalysis,
    min_cluster_size: int = 5
) -> List[DensityCluster]:
    """
    Extract clusters from gap analysis.
    
    Clusters are contiguous regions separated by gaps.
    
    Args:
        words: List of words corresponding to similarities
        similarities: Similarity values
        analysis: BinAnalysis with gap locations
        min_cluster_size: Minimum words to form a cluster
        
    Returns:
        List of DensityCluster objects
    """
    # Sort words by similarity
    sorted_indices = np.argsort(similarities)
    sorted_words = [words[i] for i in sorted_indices]
    sorted_sims = similarities[sorted_indices]
    
    # Find cluster boundaries from gaps
    boundaries = [0.0]
    for gap_start, gap_end in sorted(analysis.gap_locations):
        # Use gap midpoint as boundary
        boundaries.append((gap_start + gap_end) / 2)
    boundaries.append(1.0)
    
    # Assign words to clusters
    clusters = []
    cluster_id = 0
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        # Find words in this range
        mask = (sorted_sims >= start) & (sorted_sims < end)
        cluster_words = [w for w, m in zip(sorted_words, mask) if m]
        cluster_sims = sorted_sims[mask]
        
        if len(cluster_words) >= min_cluster_size:
            center = np.mean(cluster_sims) if len(cluster_sims) > 0 else (start + end) / 2
            density = len(cluster_words) / (end - start) if end > start else 0
            
            clusters.append(DensityCluster(
                cluster_id=cluster_id,
                start=start,
                end=end,
                center=center,
                n_words=len(cluster_words),
                words=cluster_words,
                density=density
            ))
            cluster_id += 1
    
    return clusters


def cluster_by_density(
    words: List[str],
    similarities: np.ndarray,
    min_bins: int = 5,
    max_bins: int = 50,
    min_cluster_size: int = 5,
    method: str = "combined",
    verbose: bool = True
) -> ClusteringResult:
    """
    Main entry point for density-based clustering.
    
    Args:
        words: List of words
        similarities: Corresponding similarity values
        min_bins, max_bins: Range of bin counts to try
        min_cluster_size: Minimum words per cluster
        method: Gap scoring method
        verbose: Print progress
        
    Returns:
        ClusteringResult with clusters and analysis
    """
    if verbose:
        print("Finding optimal binning...")
    
    optimal_n, analyses, scores = find_optimal_bins(
        similarities, min_bins, max_bins, method, verbose
    )
    
    if verbose:
        print(f"Optimal n_bins: {optimal_n} (score: {scores[optimal_n]:.4f})")
    
    # Extract clusters using optimal binning
    best_analysis = analyses[optimal_n]
    clusters = extract_clusters(words, similarities, best_analysis, min_cluster_size)
    
    if verbose:
        print(f"Found {len(clusters)} clusters")
    
    # Build word-to-cluster mapping
    word_to_cluster = {}
    for cluster in clusters:
        for word in cluster.words:
            word_to_cluster[word] = cluster.cluster_id
    
    return ClusteringResult(
        clusters=clusters,
        n_clusters=len(clusters),
        optimal_n_bins=optimal_n,
        bin_analyses=analyses,
        gap_scores=scores,
        word_to_cluster=word_to_cluster
    )


def plot_density_analysis(
    similarities: np.ndarray,
    result: ClusteringResult,
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualize the density analysis and clustering.
    
    Creates a 4-panel figure:
    1. Histogram with cluster boundaries
    2. Gap score vs n_bins (or GVF vs n_classes for Jenks)
    3. Cumulative distribution
    4. Cluster summary
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Panel 1: Histogram with cluster boundaries
    ax1 = axes[0, 0]
    
    # Use a reasonable number of bins for visualization
    n_viz_bins = min(50, max(20, len(similarities) // 10))
    ax1.hist(similarities, bins=n_viz_bins, alpha=0.7, edgecolor='black')
    
    # Mark cluster boundaries
    colors = plt.cm.tab10(np.linspace(0, 1, len(result.clusters)))
    for i, cluster in enumerate(result.clusters):
        ax1.axvline(cluster.start, color=colors[i], linestyle='--', linewidth=2, alpha=0.7)
        ax1.axvline(cluster.end, color=colors[i], linestyle='--', linewidth=2, alpha=0.7)
        # Shade cluster region
        ax1.axvspan(cluster.start, cluster.end, alpha=0.1, color=colors[i])
    
    ax1.set_xlabel('Similarity')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Distribution with {result.n_clusters} Clusters')
    
    # Panel 2: Score vs n (gap score or GVF)
    ax2 = axes[0, 1]
    if result.gap_scores:
        n_values = sorted(result.gap_scores.keys())
        scores = [result.gap_scores[n] for n in n_values]
        
        ax2.plot(n_values, scores, 'b-o', linewidth=2, markersize=4)
        ax2.axvline(result.optimal_n_bins, color='red', linestyle='--', 
                    label=f'Optimal: {result.optimal_n_bins}')
        
        # Determine label based on whether it's binning or Jenks
        if result.bin_analyses:
            ylabel = 'Gap Score'
            title = 'Gap Score vs Bin Count'
        else:
            ylabel = 'GVF (Goodness of Variance Fit)'
            title = 'GVF vs Number of Classes'
        
        ax2.set_xlabel('Number of bins/classes')
        ax2.set_ylabel(ylabel)
        ax2.set_title(title)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No score data available', ha='center', va='center',
                transform=ax2.transAxes)
    
    # Panel 3: Cumulative distribution
    ax3 = axes[1, 0]
    sorted_sims = np.sort(similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    
    ax3.plot(sorted_sims, cumulative, 'b-', linewidth=2)
    
    # Mark cluster boundaries with colors
    for i, cluster in enumerate(result.clusters):
        ax3.axvspan(cluster.start, cluster.end, alpha=0.2, color=colors[i],
                   label=f'C{cluster.cluster_id}: n={cluster.n_words}')
    
    ax3.set_xlabel('Similarity')
    ax3.set_ylabel('Cumulative fraction')
    ax3.set_title('Cumulative Distribution with Clusters')
    ax3.legend(loc='lower right', fontsize=8)
    
    # Panel 4: Cluster summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = result.report()
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontfamily='monospace', fontsize=9,
             verticalalignment='top')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# JENKS NATURAL BREAKS (Alternative Method)
# =============================================================================

def jenks_breaks(data: np.ndarray, n_classes: int) -> Tuple[List[float], float]:
    """
    Compute Jenks Natural Breaks.
    
    Finds class boundaries that minimize within-class variance
    and maximize between-class variance.
    
    Args:
        data: 1D array of values
        n_classes: Number of classes
        
    Returns:
        Tuple of (break_points, goodness_of_variance_fit)
    """
    data = np.sort(data)
    n = len(data)
    
    if n_classes >= n:
        return list(data), 1.0
    
    # Initialize matrices
    lower_class_limits = np.zeros((n + 1, n_classes + 1))
    variance_combinations = np.full((n + 1, n_classes + 1), np.inf)
    
    for i in range(1, n_classes + 1):
        lower_class_limits[1, i] = 1
        variance_combinations[1, i] = 0
    
    # Build variance matrix
    for l in range(2, n + 1):
        sum_val = 0
        sum_sq = 0
        w = 0
        
        for m in range(1, l + 1):
            lower_idx = l - m + 1
            val = data[lower_idx - 1]
            
            w += 1
            sum_val += val
            sum_sq += val * val
            
            variance = sum_sq - (sum_val * sum_val) / w
            
            if lower_idx > 1:
                for j in range(2, n_classes + 1):
                    new_var = variance + variance_combinations[lower_idx - 1, j - 1]
                    if new_var < variance_combinations[l, j]:
                        lower_class_limits[l, j] = lower_idx
                        variance_combinations[l, j] = new_var
    
    # Extract breaks
    k = n
    breaks = [data[-1]]
    
    for j in range(n_classes, 1, -1):
        idx = int(lower_class_limits[k, j]) - 1
        breaks.insert(0, data[idx])
        k = int(lower_class_limits[k, j]) - 1
    
    breaks.insert(0, data[0])
    
    # Compute GVF (Goodness of Variance Fit)
    total_variance = np.var(data) * len(data)
    
    within_variance = 0
    for i in range(n_classes):
        class_data = data[(data >= breaks[i]) & (data < breaks[i + 1])]
        if len(class_data) > 0:
            within_variance += np.var(class_data) * len(class_data)
    
    gvf = 1 - (within_variance / total_variance) if total_variance > 0 else 1
    
    return breaks, gvf


def cluster_by_jenks(
    words: List[str],
    similarities: np.ndarray,
    min_classes: int = 2,
    max_classes: int = 10,
    verbose: bool = True
) -> ClusteringResult:
    """
    Cluster using Jenks Natural Breaks.
    
    Finds optimal number of classes by maximizing GVF improvement.
    """
    if verbose:
        print("Computing Jenks Natural Breaks...")
    
    # Try different numbers of classes
    gvf_scores = {}
    all_breaks = {}
    
    for n in range(min_classes, max_classes + 1):
        breaks, gvf = jenks_breaks(similarities, n)
        gvf_scores[n] = gvf
        all_breaks[n] = breaks
        
        if verbose:
            print(f"  n={n}: GVF={gvf:.4f}")
    
    # Find elbow: where GVF improvement drops
    gvf_improvements = {}
    prev_gvf = 0
    for n in range(min_classes, max_classes + 1):
        gvf_improvements[n] = gvf_scores[n] - prev_gvf
        prev_gvf = gvf_scores[n]
    
    # Simple heuristic: find where improvement drops below 0.02
    optimal_n = min_classes
    for n in range(min_classes + 1, max_classes + 1):
        if gvf_improvements[n] < 0.02:
            optimal_n = n - 1
            break
        optimal_n = n
    
    if verbose:
        print(f"Optimal n_classes: {optimal_n} (GVF={gvf_scores[optimal_n]:.4f})")
    
    # Build clusters from breaks
    breaks = all_breaks[optimal_n]
    sorted_indices = np.argsort(similarities)
    sorted_words = [words[i] for i in sorted_indices]
    sorted_sims = similarities[sorted_indices]
    
    clusters = []
    for i in range(len(breaks) - 1):
        start, end = breaks[i], breaks[i + 1]
        mask = (sorted_sims >= start) & (sorted_sims < end)
        cluster_words = [w for w, m in zip(sorted_words, mask) if m]
        cluster_sims = sorted_sims[mask]
        
        if len(cluster_words) > 0:
            clusters.append(DensityCluster(
                cluster_id=i,
                start=start,
                end=end,
                center=np.mean(cluster_sims),
                n_words=len(cluster_words),
                words=cluster_words,
                density=len(cluster_words) / (end - start) if end > start else 0
            ))
    
    # Build word-to-cluster mapping
    word_to_cluster = {}
    for cluster in clusters:
        for word in cluster.words:
            word_to_cluster[word] = cluster.cluster_id
    
    return ClusteringResult(
        clusters=clusters,
        n_clusters=len(clusters),
        optimal_n_bins=optimal_n,
        bin_analyses={},  # Not applicable for Jenks
        gap_scores=gvf_scores,
        word_to_cluster=word_to_cluster
    )


# =============================================================================
# DBSCAN METHOD
# =============================================================================

def dbscan_clustering(
    words: List[str],
    similarities: np.ndarray,
    eps: float = None,
    min_samples: int = 5,
    auto_eps_percentile: float = 5,
    verbose: bool = True
) -> ClusteringResult:
    """
    Cluster using DBSCAN (Density-Based Spatial Clustering).
    
    DBSCAN groups points that are closely packed together, marking
    points in low-density regions as outliers.
    
    For 1D data, this effectively finds "dense runs" along the
    similarity axis, with gaps between them.
    
    Args:
        words: List of words
        similarities: Corresponding similarity values
        eps: Maximum distance between neighbors (None = auto)
        min_samples: Minimum points to form a dense region
        auto_eps_percentile: Percentile of k-distances for auto eps
        verbose: Print progress
        
    Returns:
        ClusteringResult with clusters
    """
    from sklearn.cluster import DBSCAN
    
    if verbose:
        print("Computing DBSCAN clustering...")
    
    # Reshape for sklearn (needs 2D array)
    X = similarities.reshape(-1, 1)
    
    # Auto-select eps using k-distance graph
    if eps is None:
        # Compute k-nearest neighbor distances
        k = min_samples
        from sklearn.neighbors import NearestNeighbors
        
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        k_distances = np.sort(distances[:, -1])
        
        # Use percentile of k-distances as eps
        eps = np.percentile(k_distances, auto_eps_percentile)
        
        if verbose:
            print(f"  Auto eps (percentile {auto_eps_percentile}): {eps:.4f}")
            print(f"  k-distance range: [{k_distances.min():.4f}, {k_distances.max():.4f}]")
    
    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    if verbose:
        print(f"  Found {n_clusters} clusters, {n_noise} noise points")
    
    # Build cluster objects
    clusters = []
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_words = [w for w, m in zip(words, mask) if m]
        cluster_sims = similarities[mask]
        
        if len(cluster_words) > 0:
            clusters.append(DensityCluster(
                cluster_id=cluster_id,
                start=cluster_sims.min(),
                end=cluster_sims.max(),
                center=np.mean(cluster_sims),
                n_words=len(cluster_words),
                words=cluster_words,
                density=len(cluster_words) / (cluster_sims.max() - cluster_sims.min() + 1e-10)
            ))
    
    # Sort clusters by start position
    clusters.sort(key=lambda c: c.start)
    
    # Reassign cluster IDs after sorting
    for i, c in enumerate(clusters):
        c.cluster_id = i
    
    # Build word-to-cluster mapping (including noise as -1)
    word_to_cluster = {}
    for word, label in zip(words, labels):
        word_to_cluster[word] = label
    
    # Store DBSCAN info for plotting
    dbscan_info = {
        'labels': labels,
        'eps': eps,
        'min_samples': min_samples,
        'n_noise': n_noise
    }
    
    result = ClusteringResult(
        clusters=clusters,
        n_clusters=n_clusters,
        optimal_n_bins=n_clusters,
        bin_analyses={},
        gap_scores={},
        word_to_cluster=word_to_cluster
    )
    
    result.dbscan_info = dbscan_info
    
    return result


def find_optimal_dbscan_eps(
    similarities: np.ndarray,
    min_samples: int = 5,
    eps_range: Tuple[float, float] = None,
    n_eps: int = 50,
    verbose: bool = True
) -> Tuple[float, Dict[float, int]]:
    """
    Find optimal eps by testing range and looking for stability.
    
    Args:
        similarities: Similarity values
        min_samples: DBSCAN min_samples parameter
        eps_range: (min_eps, max_eps) or None for auto
        n_eps: Number of eps values to test
        verbose: Print progress
        
    Returns:
        Tuple of (optimal_eps, {eps: n_clusters})
    """
    from sklearn.cluster import DBSCAN
    
    X = similarities.reshape(-1, 1)
    
    # Auto eps range based on data spread
    if eps_range is None:
        data_range = similarities.max() - similarities.min()
        eps_range = (data_range * 0.005, data_range * 0.1)
    
    eps_values = np.linspace(eps_range[0], eps_range[1], n_eps)
    n_clusters_by_eps = {}
    
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters_by_eps[eps] = n_clusters
    
    # Find "elbow" - where number of clusters stabilizes
    # Look for longest run of same cluster count
    counts = list(n_clusters_by_eps.values())
    eps_list = list(n_clusters_by_eps.keys())
    
    # Find the most common cluster count in the stable region
    from collections import Counter
    count_freq = Counter(counts)
    
    # Prefer moderate number of clusters (not 1, not too many)
    best_count = None
    for count, freq in count_freq.most_common():
        if 2 <= count <= 10:
            best_count = count
            break
    
    if best_count is None:
        best_count = count_freq.most_common(1)[0][0]
    
    # Find eps that gives this count (middle of the stable region)
    matching_eps = [e for e, c in n_clusters_by_eps.items() if c == best_count]
    optimal_eps = np.median(matching_eps)
    
    if verbose:
        print(f"  Tested {n_eps} eps values in [{eps_range[0]:.4f}, {eps_range[1]:.4f}]")
        print(f"  Cluster counts: {dict(count_freq.most_common(5))}")
        print(f"  Optimal eps: {optimal_eps:.4f} (gives {best_count} clusters)")
    
    return optimal_eps, n_clusters_by_eps


def plot_dbscan_analysis(
    similarities: np.ndarray,
    result: ClusteringResult,
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualize DBSCAN clustering.
    
    Creates a 4-panel figure:
    1. Scatter plot with cluster colors
    2. Histogram with cluster regions
    3. Cumulative distribution with clusters
    4. Cluster summary
    """
    import matplotlib.pyplot as plt
    
    if not hasattr(result, 'dbscan_info'):
        print("No DBSCAN info available. Use dbscan_clustering() first.")
        return
    
    info = result.dbscan_info
    labels = info['labels']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Color map: -1 (noise) is gray, clusters get distinct colors
    unique_labels = sorted(set(labels))
    colors = {}
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))
    color_idx = 0
    for label in unique_labels:
        if label == -1:
            colors[label] = 'gray'
        else:
            colors[label] = cluster_colors[color_idx]
            color_idx += 1
    
    # Panel 1: Scatter plot (1D projected to strip)
    ax1 = axes[0, 0]
    
    # Add jitter for visibility
    np.random.seed(42)
    y_jitter = np.random.uniform(-0.3, 0.3, len(similarities))
    
    for label in unique_labels:
        mask = labels == label
        label_name = f'Cluster {label}' if label >= 0 else f'Noise (n={info["n_noise"]})'
        ax1.scatter(similarities[mask], y_jitter[mask], 
                   c=[colors[label]], label=label_name, alpha=0.6, s=20)
    
    ax1.set_xlabel('Similarity')
    ax1.set_ylabel('(jittered for visibility)')
    ax1.set_title(f'DBSCAN Clustering (eps={info["eps"]:.4f}, min_samples={info["min_samples"]})')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(-0.5, 0.5)
    
    # Panel 2: Histogram with cluster regions
    ax2 = axes[0, 1]
    
    ax2.hist(similarities, bins=50, alpha=0.5, edgecolor='black', label='All')
    
    # Overlay cluster regions
    for cluster in result.clusters:
        ax2.axvspan(cluster.start, cluster.end, alpha=0.2, 
                   color=colors[cluster.cluster_id],
                   label=f'C{cluster.cluster_id}: [{cluster.start:.2f}, {cluster.end:.2f}]')
    
    ax2.set_xlabel('Similarity')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution with Cluster Regions')
    ax2.legend(fontsize=8)
    
    # Panel 3: Cumulative distribution
    ax3 = axes[1, 0]
    
    sorted_sims = np.sort(similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    
    ax3.plot(sorted_sims, cumulative, 'b-', linewidth=2)
    
    for cluster in result.clusters:
        ax3.axvspan(cluster.start, cluster.end, alpha=0.2, 
                   color=colors[cluster.cluster_id],
                   label=f'C{cluster.cluster_id}: n={cluster.n_words}')
    
    ax3.set_xlabel('Similarity')
    ax3.set_ylabel('Cumulative fraction')
    ax3.set_title('Cumulative Distribution with Clusters')
    ax3.legend(loc='lower right', fontsize=8)
    
    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_lines = [
        "=" * 50,
        "DBSCAN CLUSTERING RESULT",
        "=" * 50,
        f"eps: {info['eps']:.4f}",
        f"min_samples: {info['min_samples']}",
        f"N clusters: {result.n_clusters}",
        f"Noise points: {info['n_noise']}",
        "",
        "Clusters:"
    ]
    
    for c in result.clusters:
        summary_lines.append(
            f"  {c.cluster_id}: [{c.start:.3f}, {c.end:.3f}] "
            f"n={c.n_words}"
        )
        sample = c.words[:5]
        if sample:
            summary_lines.append(f"      e.g., {', '.join(sample)}")
    
    # Noise words
    noise_words = [w for w, l in zip(result.word_to_cluster.keys(), labels) if l == -1]
    if noise_words:
        summary_lines.append(f"\nNoise ({len(noise_words)} words):")
        summary_lines.append(f"    e.g., {', '.join(noise_words[:10])}")
    
    ax4.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax4.transAxes,
             fontfamily='monospace', fontsize=9,
             verticalalignment='top')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# KERNEL DENSITY ESTIMATION (KDE) METHOD
# =============================================================================

def kde_clustering(
    words: List[str],
    similarities: np.ndarray,
    bandwidth: float = None,
    min_cluster_size: int = 5,
    local_minima_threshold: float = 0.1,
    verbose: bool = True
) -> ClusteringResult:
    """
    Cluster using Kernel Density Estimation.
    
    Finds clusters by:
    1. Estimating smooth density function via KDE
    2. Finding local minima (valleys) in the density
    3. Using valleys as cluster boundaries
    
    This naturally handles relative gaps - a valley in a dense region
    is detected even if its absolute density is higher than peaks
    in sparse regions.
    
    Args:
        words: List of words
        similarities: Corresponding similarity values
        bandwidth: KDE bandwidth (None = Scott's rule)
        min_cluster_size: Minimum words per cluster
        local_minima_threshold: Relative depth for valid minima
        verbose: Print progress
        
    Returns:
        ClusteringResult with clusters
    """
    from scipy.stats import gaussian_kde
    from scipy.signal import argrelextrema
    
    if verbose:
        print("Computing KDE-based clustering...")
    
    # Fit KDE
    if bandwidth is None:
        kde = gaussian_kde(similarities, bw_method='scott')
        bandwidth = kde.factor * similarities.std()
        if verbose:
            print(f"  Auto bandwidth (Scott's rule): {bandwidth:.4f}")
    else:
        kde = gaussian_kde(similarities, bw_method=bandwidth)
    
    # Evaluate density on fine grid
    x_grid = np.linspace(0, 1, 1000)
    density = kde(x_grid)
    
    # Find local minima
    minima_idx = argrelextrema(density, np.less, order=10)[0]
    maxima_idx = argrelextrema(density, np.greater, order=10)[0]
    
    if verbose:
        print(f"  Found {len(minima_idx)} local minima, {len(maxima_idx)} local maxima")
    
    # Filter minima by relative depth
    # A valid minimum should be significantly lower than surrounding maxima
    valid_minima = []
    
    for min_idx in minima_idx:
        min_val = density[min_idx]
        min_x = x_grid[min_idx]
        
        # Find surrounding maxima
        left_maxima = [m for m in maxima_idx if m < min_idx]
        right_maxima = [m for m in maxima_idx if m > min_idx]
        
        left_max_val = density[left_maxima[-1]] if left_maxima else min_val
        right_max_val = density[right_maxima[0]] if right_maxima else min_val
        
        # Relative depth: how deep is this valley compared to surrounding peaks?
        surrounding_max = max(left_max_val, right_max_val)
        if surrounding_max > 0:
            relative_depth = 1 - (min_val / surrounding_max)
        else:
            relative_depth = 0
        
        if relative_depth >= local_minima_threshold:
            valid_minima.append((min_x, min_val, relative_depth))
            if verbose:
                print(f"    Valid minimum at x={min_x:.3f}, depth={relative_depth:.3f}")
    
    if verbose:
        print(f"  Valid minima (depth >= {local_minima_threshold}): {len(valid_minima)}")
    
    # Build cluster boundaries from minima
    boundaries = [0.0]
    for min_x, _, _ in sorted(valid_minima):
        boundaries.append(min_x)
    boundaries.append(1.0)
    
    # Assign words to clusters
    sorted_indices = np.argsort(similarities)
    sorted_words = [words[i] for i in sorted_indices]
    sorted_sims = similarities[sorted_indices]
    
    clusters = []
    cluster_id = 0
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        
        # Handle edge case for last cluster (include endpoint)
        if i == len(boundaries) - 2:
            mask = (sorted_sims >= start) & (sorted_sims <= end)
        else:
            mask = (sorted_sims >= start) & (sorted_sims < end)
        
        cluster_words = [w for w, m in zip(sorted_words, mask) if m]
        cluster_sims = sorted_sims[mask]
        
        if len(cluster_words) >= min_cluster_size:
            center = np.mean(cluster_sims) if len(cluster_sims) > 0 else (start + end) / 2
            width = end - start
            density_val = len(cluster_words) / width if width > 0 else 0
            
            clusters.append(DensityCluster(
                cluster_id=cluster_id,
                start=start,
                end=end,
                center=center,
                n_words=len(cluster_words),
                words=cluster_words,
                density=density_val
            ))
            cluster_id += 1
    
    # Build word-to-cluster mapping
    word_to_cluster = {}
    for cluster in clusters:
        for word in cluster.words:
            word_to_cluster[word] = cluster.cluster_id
    
    # Store KDE info for plotting
    kde_info = {
        'x_grid': x_grid,
        'density': density,
        'minima': valid_minima,
        'maxima_idx': maxima_idx,
        'bandwidth': bandwidth
    }
    
    if verbose:
        print(f"  Found {len(clusters)} clusters")
    
    result = ClusteringResult(
        clusters=clusters,
        n_clusters=len(clusters),
        optimal_n_bins=len(clusters),  # Not really bins, but for compatibility
        bin_analyses={},
        gap_scores={},
        word_to_cluster=word_to_cluster
    )
    
    # Attach KDE info for plotting
    result.kde_info = kde_info
    
    return result


def plot_kde_analysis(
    similarities: np.ndarray,
    result: ClusteringResult,
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualize KDE-based clustering.
    
    Creates a 4-panel figure:
    1. Histogram with KDE overlay and cluster boundaries
    2. KDE density with local minima marked
    3. Cumulative distribution with clusters
    4. Cluster summary
    """
    import matplotlib.pyplot as plt
    
    if not hasattr(result, 'kde_info'):
        print("No KDE info available. Use kde_clustering() first.")
        return
    
    kde_info = result.kde_info
    x_grid = kde_info['x_grid']
    density = kde_info['density']
    minima = kde_info['minima']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(result.clusters), 1)))
    
    # Panel 1: Histogram with KDE overlay
    ax1 = axes[0, 0]
    
    # Histogram (normalized to match KDE scale)
    ax1.hist(similarities, bins=50, density=True, alpha=0.5, 
             edgecolor='black', label='Histogram')
    
    # KDE curve
    ax1.plot(x_grid, density, 'b-', linewidth=2, label='KDE')
    
    # Mark cluster boundaries
    for i, cluster in enumerate(result.clusters):
        ax1.axvline(cluster.start, color=colors[i], linestyle='--', 
                   linewidth=2, alpha=0.7)
        ax1.axvline(cluster.end, color=colors[i], linestyle='--', 
                   linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Similarity')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution with KDE (bandwidth={kde_info["bandwidth"]:.4f})')
    ax1.legend()
    
    # Panel 2: KDE with minima/maxima marked
    ax2 = axes[0, 1]
    
    ax2.plot(x_grid, density, 'b-', linewidth=2, label='KDE density')
    
    # Mark valid minima (cluster boundaries)
    for min_x, min_val, depth in minima:
        ax2.plot(min_x, min_val, 'rv', markersize=12, 
                label=f'Min at {min_x:.2f} (depth={depth:.2f})')
        ax2.axvline(min_x, color='red', linestyle=':', alpha=0.5)
    
    # Mark maxima
    maxima_idx = kde_info['maxima_idx']
    ax2.plot(x_grid[maxima_idx], density[maxima_idx], 'g^', markersize=8, 
            label='Local maxima')
    
    ax2.set_xlabel('Similarity')
    ax2.set_ylabel('Density')
    ax2.set_title('KDE with Local Minima (Cluster Boundaries)')
    ax2.legend(fontsize=8)
    
    # Panel 3: Cumulative distribution with clusters
    ax3 = axes[1, 0]
    
    sorted_sims = np.sort(similarities)
    cumulative = np.arange(1, len(sorted_sims) + 1) / len(sorted_sims)
    
    ax3.plot(sorted_sims, cumulative, 'b-', linewidth=2)
    
    for i, cluster in enumerate(result.clusters):
        ax3.axvspan(cluster.start, cluster.end, alpha=0.2, color=colors[i],
                   label=f'C{cluster.cluster_id}: n={cluster.n_words}')
    
    ax3.set_xlabel('Similarity')
    ax3.set_ylabel('Cumulative fraction')
    ax3.set_title('Cumulative Distribution with Clusters')
    ax3.legend(loc='lower right', fontsize=8)
    
    # Panel 4: Cluster summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_lines = [
        "=" * 50,
        "KDE-BASED CLUSTERING RESULT",
        "=" * 50,
        f"Bandwidth: {kde_info['bandwidth']:.4f}",
        f"N clusters: {result.n_clusters}",
        "",
        "Clusters:"
    ]
    
    for c in result.clusters:
        summary_lines.append(
            f"  {c.cluster_id}: [{c.start:.3f}, {c.end:.3f}] "
            f"n={c.n_words}, density={c.density:.1f}"
        )
        sample = c.words[:5]
        if sample:
            summary_lines.append(f"      e.g., {', '.join(sample)}")
    
    ax4.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax4.transAxes,
             fontfamily='monospace', fontsize=9,
             verticalalignment='top')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_clustering_methods(
    words: List[str],
    similarities: np.ndarray,
    output_prefix: str = "clustering_comparison",
    verbose: bool = True
) -> Dict[str, ClusteringResult]:
    """
    Run all clustering methods and compare results.
    
    Args:
        words: List of words
        similarities: Similarity values
        output_prefix: Prefix for output files
        verbose: Print progress
        
    Returns:
        Dict mapping method name to ClusteringResult
    """
    results = {}
    
    # Jenks
    if verbose:
        print("\n" + "=" * 60)
        print("METHOD 1: Jenks Natural Breaks")
        print("=" * 60)
    results['jenks'] = cluster_by_jenks(words, similarities, verbose=verbose)
    
    # KDE
    if verbose:
        print("\n" + "=" * 60)
        print("METHOD 2: KDE-based Clustering")
        print("=" * 60)
    results['kde'] = kde_clustering(words, similarities, verbose=verbose)
    
    # Binning (for completeness, even though it doesn't work well)
    if verbose:
        print("\n" + "=" * 60)
        print("METHOD 3: Adaptive Binning")
        print("=" * 60)
    results['binning'] = cluster_by_density(words, similarities, verbose=verbose)
    
    # Summary comparison
    if verbose:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Method':<15} {'N Clusters':<12} {'Cluster Sizes'}")
        print("-" * 60)
        for name, result in results.items():
            sizes = [c.n_words for c in result.clusters]
            print(f"{name:<15} {result.n_clusters:<12} {sizes}")
    
    # Check agreement between Jenks and KDE
    if 'jenks' in results and 'kde' in results:
        jenks_assignments = results['jenks'].word_to_cluster
        kde_assignments = results['kde'].word_to_cluster
        
        # Count words with same relative ordering
        common_words = set(jenks_assignments.keys()) & set(kde_assignments.keys())
        
        if verbose and common_words:
            print(f"\nAgreement analysis (Jenks vs KDE):")
            print(f"  Common words: {len(common_words)}")
            
            # Simple agreement: are boundaries similar?
            jenks_bounds = sorted(set([c.start for c in results['jenks'].clusters] + 
                                     [c.end for c in results['jenks'].clusters]))
            kde_bounds = sorted(set([c.start for c in results['kde'].clusters] + 
                                   [c.end for c in results['kde'].clusters]))
            
            print(f"  Jenks boundaries: {[f'{b:.2f}' for b in jenks_bounds]}")
            print(f"  KDE boundaries:   {[f'{b:.2f}' for b in kde_bounds]}")
    
    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Density-based clustering along main line")
    parser.add_argument("--input", required=True, help="TSV file with similarity data")
    parser.add_argument("--method", choices=["binning", "jenks", "kde", "dbscan", "compare"], 
                        default="jenks", help="Clustering method")
    parser.add_argument("--min-bins", type=int, default=5)
    parser.add_argument("--max-bins", type=int, default=50)
    parser.add_argument("--bandwidth", type=float, default=None, 
                        help="KDE bandwidth (None=auto)")
    parser.add_argument("--min-depth", type=float, default=0.1,
                        help="Minimum relative depth for KDE minima")
    parser.add_argument("--eps", type=float, default=None,
                        help="DBSCAN eps (None=auto)")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="DBSCAN min_samples")
    parser.add_argument("--output", default="density_clusters", help="Output prefix")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    words = []
    similarities = []
    
    with open(args.input, 'r') as f:
        header = next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                words.append(parts[0])
                similarities.append(float(parts[1]))
    
    similarities = np.array(similarities)
    print(f"Loaded {len(words)} words")
    print(f"Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
    
    # Cluster
    if args.method == "binning":
        result = cluster_by_density(
            words, similarities,
            min_bins=args.min_bins,
            max_bins=args.max_bins,
            verbose=True
        )
        plot_func = plot_density_analysis
        
    elif args.method == "jenks":
        result = cluster_by_jenks(
            words, similarities,
            verbose=True
        )
        plot_func = plot_density_analysis
        
    elif args.method == "kde":
        result = kde_clustering(
            words, similarities,
            bandwidth=args.bandwidth,
            local_minima_threshold=args.min_depth,
            verbose=True
        )
        plot_func = plot_kde_analysis
        
    elif args.method == "dbscan":
        # First find optimal eps if not specified
        if args.eps is None:
            print("\nFinding optimal eps...")
            optimal_eps, eps_to_clusters = find_optimal_dbscan_eps(
                similarities, 
                min_samples=args.min_samples,
                verbose=True
            )
            args.eps = optimal_eps
        
        result = dbscan_clustering(
            words, similarities,
            eps=args.eps,
            min_samples=args.min_samples,
            verbose=True
        )
        plot_func = plot_dbscan_analysis
        
    elif args.method == "compare":
        results = {}
        
        # Jenks
        print("\n" + "=" * 60)
        print("METHOD 1: Jenks Natural Breaks")
        print("=" * 60)
        results['jenks'] = cluster_by_jenks(words, similarities, verbose=True)
        
        # DBSCAN
        print("\n" + "=" * 60)
        print("METHOD 2: DBSCAN")
        print("=" * 60)
        optimal_eps, _ = find_optimal_dbscan_eps(similarities, verbose=True)
        results['dbscan'] = dbscan_clustering(words, similarities, eps=optimal_eps, verbose=True)
        
        # KDE (for completeness)
        print("\n" + "=" * 60)
        print("METHOD 3: KDE")
        print("=" * 60)
        results['kde'] = kde_clustering(words, similarities, verbose=True)
        
        # Summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Method':<15} {'N Clusters':<12} {'Cluster Sizes'}")
        print("-" * 60)
        for name, res in results.items():
            sizes = [c.n_words for c in res.clusters]
            noise = ""
            if hasattr(res, 'dbscan_info'):
                noise = f" (+{res.dbscan_info['n_noise']} noise)"
            print(f"{name:<15} {res.n_clusters:<12} {sizes}{noise}")
        
        # Plot each
        for name, res in results.items():
            if name == 'kde':
                plot_kde_analysis(similarities, res, 
                                 output_path=f"{args.output}_{name}_plot.png")
            elif name == 'dbscan':
                plot_dbscan_analysis(similarities, res,
                                    output_path=f"{args.output}_{name}_plot.png")
            else:
                plot_density_analysis(similarities, res,
                                     output_path=f"{args.output}_{name}_plot.png")
        
        print(f"\nPlots saved to {args.output}_*_plot.png")
        exit(0)
    
    print("\n" + result.report())
    
    # Plot
    try:
        plot_func(
            similarities, result,
            output_path=f"{args.output}_plot.png"
        )
    except ImportError as e:
        print(f"Plotting library not available: {e}")
    except Exception as e:
        print(f"Plotting error: {e}")
        import traceback
        traceback.print_exc()
    
    # Save cluster assignments
    output_file = f"{args.output}_assignments.tsv"
    with open(output_file, 'w') as f:
        f.write("word\tsimilarity\tcluster\n")
        for word, sim in zip(words, similarities):
            cluster_id = result.word_to_cluster.get(word, -1)
            f.write(f"{word}\t{sim:.4f}\t{cluster_id}\n")
    
    print(f"\nCluster assignments saved to {output_file}")
