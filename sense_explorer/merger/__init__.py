"""
sense_explorer.merger - Embedding Merger Subpackage
====================================================

Advanced functionality for combining word embeddings from multiple sources
into a unified semantic space.

**Dependency**: This subpackage requires sense separation (the basic
functionality of SenseExplorer) as a prerequisite. Sense vectors must
be extracted from each embedding before they can be meaningfully merged.

Conceptual Hierarchy:
    sense_explorer (basic)
        └── sense separation: Extract sense vectors from ONE embedding
    
    sense_explorer.merger (advanced)
        └── embedding merger: Combine senses across MULTIPLE embeddings

Why Separation is Required:
    Without sense separation, each word is a single vector — a superposition
    of all its meanings. Merging superpositions just creates messier
    superpositions. With sense separation, we have distinct sense components
    that can be meaningfully aligned across embeddings.

Clustering Methods:
    - 'hierarchical': Standard agglomerative clustering
    - 'spectral': Pure spectral clustering with eigengap k-selection
    - 'spectral_hierarchical': Hybrid (recommended) — wave-aware + dendrograms

Quality-Based Weighting (v0.3.0):
    The merger supports quality-based weighting of embedding sources.
    Weights are computed from:
    - Vocabulary size (log-scaled)
    - Anchor coherence (from IVA distillation)
    - Sense separation quality (R² from geometry)
    - Vocabulary overlap (shared words)

Convergence Validation (v0.4.0):
    Validate whether senses that cluster together truly represent the same
    meaning across embeddings. Methods include:
    - Neighbor overlap (Jaccard similarity)
    - Anchor consistency
    - Cross-embedding projection
    - Semantic coherence scoring

Basic Usage:
    ```python
    from sense_explorer import SenseExplorer
    from sense_explorer.merger import EmbeddingMerger
    
    # Load embeddings (basic functionality)
    se_wiki = SenseExplorer.from_file("glove-wiki-100d.txt")
    se_twitter = SenseExplorer.from_file("glove-twitter-100d.txt")
    
    # Merge senses (advanced functionality)
    merger = EmbeddingMerger(clustering_method='spectral_hierarchical')
    merger.add_embedding("wiki", se_wiki.embeddings)
    merger.add_embedding("twitter", se_twitter.embeddings)
    
    result = merger.merge_senses("bank")
    print(f"Convergent: {result.n_convergent}")
    print(f"Source-specific: {result.n_source_specific}")
    ```

Weighted Merge (Recommended):
    ```python
    from sense_explorer.merger import merge_with_weights
    
    result = merge_with_weights(
        {"wiki": se_wiki, "twitter": se_twitter},
        "bank",
        weight_config={'coherence': 0.4, 'separation': 0.4}  # Custom weights
    )
    print(f"Source weights: {result.source_weights}")
    print(f"Convergent: {result.n_convergent}")
    ```

Convergence Validation:
    ```python
    from sense_explorer.merger import merge_with_weights, validate_convergence
    
    result = merge_with_weights({"wiki": se_wiki, "twitter": se_twitter}, "bank")
    report = validate_convergence(result, {"wiki": se_wiki, "twitter": se_twitter})
    
    print(report.summary())
    print(f"Low confidence clusters: {report.low_confidence_clusters}")
    ```

Convenience Methods (on SenseExplorer):
    ```python
    # Two-way merge via SenseExplorer method
    result = se_wiki.merge_with(se_twitter, "bank")
    
    # Get merger for N-way merge
    merger = se_wiki.get_merger({"twitter": se_twitter, "news": se_news})
    ```

For Large Embeddings (Staged Merger):
    ```python
    from sense_explorer.merger import StagedMerger, MergeStrategy
    
    specs = {
        "wiki": {"path": "wiki.txt", "format": "glove"},
        "twitter": {"path": "twitter.txt", "format": "glove"},
    }
    
    staged = StagedMerger(specs, max_concurrent=2, strategy=MergeStrategy.AFFINITY)
    plan = staged.plan_merge_order(sample_words=["bank", "rock"])
    result = staged.merge_staged("bank")
    ```

Author: Kow Kuroda & Claude (Anthropic)
License: MIT
Version: 0.4.0
"""

from .embedding_merger import (
    # Main class
    EmbeddingMerger,
    
    # Data classes
    SenseComponent,
    MergerResult,
    MergerBasis,
    SpectralInfo,
    
    # Weighted merger classes
    WeightedMergerResult,
    
    # Enums
    ClusteringMethod,
    
    # Integration functions
    create_merger_from_explorers,
    merge_with_ssr,
    merge_with_weights,  # Quality-weighted merge
    weighted_report,     # Report for weighted results
    
    # Visualization
    plot_merger_dendrogram,
    plot_spectral_analysis,
    
    # Spectral analysis functions
    compute_laplacian_spectrum,
    find_k_by_eigengap,
    spectral_embedding_from_similarity,
)

from .embedding_weights import (
    # Quality assessment
    EmbeddingQualityAssessor,
    EmbeddingQuality,
    SenseQuality,
    WeightedSenseComponent,
    
    # Weighted computation
    compute_weighted_similarity,
    compute_weighted_centroid,
    weight_similarity_matrix,
    
    # Convenience
    quick_assess,
)

from .convergence_validation import (
    # Data classes
    SenseValidation,
    ClusterValidation,
    ConvergenceReport,
    
    # Main validation function
    validate_convergence,
    
    # Individual metrics
    compute_neighbor_overlap,
    compute_anchor_consistency,
    compute_cross_projection_match,
    compute_semantic_coherence,
    compute_confidence,
    
    # Convenience functions
    quick_validate,
    validate_multiple,
    summarize_validations,
)

from .register_profiles import (
    # Data classes
    SensePrevalence,
    RegisterNeighbors,
    SenseDrift,
    RegisterSignature,
    RegisterProfile,
    RegisterSpecificityCluster,
    RegisterSpecificityAnalysis,
    
    # Main functions
    create_register_profile,
    compare_registers,
    summarize_register_comparison,
    cluster_by_register_specificity,
    analyze_register_specificity,
    
    # Individual analysis functions
    compute_sense_prevalence,
    compute_register_neighbors,
    compute_sense_drift,
    compute_register_signatures,
    
    # Visualization
    plot_register_profile,
    plot_register_specificity,
    plot_register_specificity_detailed,
)

from .staged_embedding_merger import (
    # Main class
    StagedMerger,
    
    # Data classes
    EmbeddingSpec,
    MergeStep,
    MergePlan,
    StagedMergeResult,
    
    # Enums
    MergeStrategy,
    
    # Convenience function
    quick_staged_merge,
)

__version__ = "0.5.0"
__author__ = "Kow Kuroda & Claude"

__all__ = [
    # Core merger
    'EmbeddingMerger',
    'SenseComponent',
    'MergerResult',
    'MergerBasis',
    'SpectralInfo',
    'ClusteringMethod',
    'create_merger_from_explorers',
    'merge_with_ssr',
    'plot_merger_dendrogram',
    'plot_spectral_analysis',
    
    # Weighted merger
    'WeightedMergerResult',
    'merge_with_weights',
    'weighted_report',
    
    # Quality assessment
    'EmbeddingQualityAssessor',
    'EmbeddingQuality',
    'SenseQuality',
    'WeightedSenseComponent',
    'compute_weighted_similarity',
    'compute_weighted_centroid',
    'weight_similarity_matrix',
    'quick_assess',
    
    # Convergence validation (NEW)
    'SenseValidation',
    'ClusterValidation',
    'ConvergenceReport',
    'validate_convergence',
    'compute_neighbor_overlap',
    'compute_anchor_consistency',
    'compute_cross_projection_match',
    'compute_semantic_coherence',
    'compute_confidence',
    'quick_validate',
    'validate_multiple',
    'summarize_validations',
    
    # Register profiles (NEW)
    'SensePrevalence',
    'RegisterNeighbors',
    'SenseDrift',
    'RegisterSignature',
    'RegisterProfile',
    'RegisterSpecificityCluster',
    'RegisterSpecificityAnalysis',
    'create_register_profile',
    'compare_registers',
    'summarize_register_comparison',
    'cluster_by_register_specificity',
    'analyze_register_specificity',
    'compute_sense_prevalence',
    'compute_register_neighbors',
    'compute_sense_drift',
    'compute_register_signatures',
    'plot_register_profile',
    'plot_register_specificity',
    'plot_register_specificity_detailed',
    
    # Spectral analysis
    'compute_laplacian_spectrum',
    'find_k_by_eigengap',
    'spectral_embedding_from_similarity',
    
    # Staged merger
    'StagedMerger',
    'EmbeddingSpec',
    'MergeStep',
    'MergePlan',
    'StagedMergeResult',
    'MergeStrategy',
    'quick_staged_merge',
]
