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
Version: 0.2.0
"""

from .embedding_merger import (
    # Main class
    EmbeddingMerger,
    
    # Data classes
    SenseComponent,
    MergerResult,
    MergerBasis,
    SpectralInfo,
    
    # Enums
    ClusteringMethod,
    
    # Integration functions
    create_merger_from_explorers,
    merge_with_ssr,
    
    # Visualization
    plot_merger_dendrogram,
    plot_spectral_analysis,
    
    # Spectral analysis functions
    compute_laplacian_spectrum,
    find_k_by_eigengap,
    spectral_embedding_from_similarity,
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

__version__ = "0.2.0"
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
