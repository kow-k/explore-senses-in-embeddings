"""
Integration code for adding embedding merger to SenseExplorer
=============================================================

Add this code to core.py to enable embedding merger functionality.

Version: 0.9.3
"""

# =============================================================================
# ADD TO IMPORTS SECTION (at top of core.py)
# =============================================================================

# Embedding merger (optional - requires multiple embeddings)
try:
    from .merger import EmbeddingMerger, SenseComponent, MergerResult
    MERGER_AVAILABLE = True
except ImportError:
    MERGER_AVAILABLE = False


# =============================================================================
# ADD TO SenseExplorer CLASS
# =============================================================================

class SenseExplorer:
    """Add these methods to the existing SenseExplorer class."""
    
    # -------------------------------------------------------------------------
    # Method 1: Get merger instance
    # -------------------------------------------------------------------------
    
    def get_merger(self, other_explorers: Dict[str, 'SenseExplorer'] = None) -> 'EmbeddingMerger':
        """
        Get an EmbeddingMerger configured with this explorer's embeddings.
        
        Args:
            other_explorers: Additional SenseExplorer instances to include.
                             Dict mapping names to explorers.
        
        Returns:
            Configured EmbeddingMerger
            
        Example:
            ```python
            se_wiki = SenseExplorer.from_file("wiki.txt")
            se_twitter = SenseExplorer.from_file("twitter.txt")
            
            # Get merger from wiki, adding twitter
            merger = se_wiki.get_merger({"twitter": se_twitter})
            result = merger.merge_senses("bank")
            ```
        """
        if not MERGER_AVAILABLE:
            raise ImportError("merger module not available")
        
        from .merger import EmbeddingMerger
        
        merger = EmbeddingMerger(verbose=self.verbose)
        merger.add_embedding("self", self.embeddings)
        
        if other_explorers:
            for name, se in other_explorers.items():
                merger.add_embedding(name, se.embeddings)
        
        return merger
    
    # -------------------------------------------------------------------------
    # Method 2: Merge senses with another explorer (convenience method)
    # -------------------------------------------------------------------------
    
    def merge_with(
        self,
        other: 'SenseExplorer',
        word: str,
        other_name: str = "other",
        self_name: str = "self",
        n_senses: int = None,
        use_ssr: bool = True,
        distance_threshold: float = 0.05
    ) -> 'MergerResult':
        """
        Merge senses of a word with another embedding.
        
        This is a convenience method for two-embedding merger.
        For more control or more embeddings, use get_merger().
        
        Args:
            other: Another SenseExplorer instance
            word: Word to merge
            other_name: Name for the other embedding (for labeling)
            self_name: Name for this embedding (for labeling)
            n_senses: Number of senses to extract (None = auto)
            use_ssr: If True, use SSR for sense extraction; else use k-means
            distance_threshold: Clustering threshold
            
        Returns:
            MergerResult with convergent and source-specific senses
            
        Example:
            ```python
            se_wiki = SenseExplorer.from_file("wiki.txt")
            se_twitter = SenseExplorer.from_file("twitter.txt")
            
            result = se_wiki.merge_with(se_twitter, "bank")
            print(f"Convergent senses: {result.n_convergent}")
            print(f"Source-specific: {result.n_source_specific}")
            ```
        """
        if not MERGER_AVAILABLE:
            raise ImportError("merger module not available")
        
        from .merger import EmbeddingMerger, SenseComponent, merge_with_ssr
        
        if use_ssr:
            # Use SSR-based extraction
            return merge_with_ssr(
                {self_name: self, other_name: other},
                word,
                n_senses=n_senses,
                distance_threshold=distance_threshold,
                verbose=self.verbose
            )
        else:
            # Use simple k-means extraction
            merger = EmbeddingMerger(verbose=self.verbose)
            merger.add_embedding(self_name, self.embeddings)
            merger.add_embedding(other_name, other.embeddings)
            return merger.merge_senses(word, n_senses=n_senses or 3,
                                       distance_threshold=distance_threshold)
    
    # -------------------------------------------------------------------------
    # Method 3: Extract sense components (for use with external merger)
    # -------------------------------------------------------------------------
    
    def extract_sense_components(
        self,
        word: str,
        source_name: str = None,
        n_senses: int = None
    ) -> List['SenseComponent']:
        """
        Extract sense components in a format suitable for embedding merger.
        
        Uses SSR (induce_senses) for extraction, then wraps results
        in SenseComponent dataclass for use with EmbeddingMerger.
        
        Args:
            word: Target word
            source_name: Name for this embedding source
            n_senses: Number of senses (None = auto via eigengap)
            
        Returns:
            List of SenseComponent objects
            
        Example:
            ```python
            # Extract from multiple explorers
            components = []
            components.extend(se_wiki.extract_sense_components("bank", "wiki"))
            components.extend(se_twitter.extract_sense_components("bank", "twitter"))
            
            # Use with merger
            merger = EmbeddingMerger()
            merger.add_embedding("wiki", se_wiki.embeddings)
            merger.add_embedding("twitter", se_twitter.embeddings)
            result = merger.merge_senses("bank", sense_components=components)
            ```
        """
        if not MERGER_AVAILABLE:
            raise ImportError("merger module not available")
        
        from .merger import SenseComponent
        
        if source_name is None:
            source_name = "source"
        
        # Use SSR to get sense vectors
        senses = self.induce_senses(word, n_senses=n_senses)
        
        components = []
        for sense_name, sense_vec in senses.items():
            # Compute neighbors
            neighbors = []
            for w in self.embeddings:
                if w != word:
                    sim = float(np.dot(sense_vec, self.embeddings[w]))
                    neighbors.append((w, sim))
            neighbors.sort(key=lambda x: -x[1])
            
            components.append(SenseComponent(
                word=word,
                sense_id=f"{source_name}_{sense_name}",
                vector=sense_vec,
                source=source_name,
                top_neighbors=neighbors[:50]
            ))
        
        return components


# =============================================================================
# ADD TO __init__.py EXPORTS
# =============================================================================

"""
Add these to __init__.py:

# Merger module (optional)
try:
    from .merger import (
        EmbeddingMerger,
        SenseComponent,
        MergerResult,
        MergerBasis,
        create_merger_from_explorers,
        merge_with_ssr,
        plot_merger_dendrogram
    )
    __all__.extend([
        'EmbeddingMerger',
        'SenseComponent', 
        'MergerResult',
        'MergerBasis',
        'create_merger_from_explorers',
        'merge_with_ssr',
        'plot_merger_dendrogram'
    ])
except ImportError:
    pass
"""


# =============================================================================
# UPDATED README SECTION
# =============================================================================

README_SECTION = """
## Embedding Merger (v0.9.3)

Combine word embeddings from multiple sources into a unified semantic space.

### Why Merger Requires Sense Separation

Without sense separation, merging embeddings just combines superpositions:
- bank (wiki) = 0.6·financial + 0.3·river + 0.1·other
- bank (twitter) = 0.7·financial + 0.2·slang + 0.1·other
- merged = messier superposition

With sense separation first:
- wiki_financial, wiki_river (separated)
- twitter_financial, twitter_slang (separated)  
- Can now align: wiki_financial ↔ twitter_financial (convergent)
- And identify: wiki_river, twitter_slang (source-specific)

### Quick Start

```python
from sense_explorer import SenseExplorer

# Load two embeddings
se_wiki = SenseExplorer.from_file("glove-wiki-100d.txt")
se_twitter = SenseExplorer.from_file("glove-twitter-100d.txt")

# Simple two-way merge
result = se_wiki.merge_with(se_twitter, "bank")
print(f"Convergent senses: {result.n_convergent}")
print(f"Source-specific: {result.n_source_specific}")

# Or with more control
from sense_explorer.merger import EmbeddingMerger

merger = EmbeddingMerger(verbose=True)
merger.add_embedding("wiki", se_wiki.embeddings)
merger.add_embedding("twitter", se_twitter.embeddings)
merger.add_embedding("news", se_news.embeddings)  # Can add 3+

result = merger.merge_senses("bank", distance_threshold=0.05)
print(merger.report(result))
```

### Threshold Selection

| Threshold | Effect |
|-----------|--------|
| < 0.03 | Fine-grained: preserves source-specific nuances |
| 0.03–0.10 | Balanced: merges equivalent senses |
| > 0.20 | Coarse: collapses most structure |

```python
# Test multiple thresholds
results = merger.merge_senses("bank", return_all_thresholds=True)
for thresh, result in results.items():
    print(f"{thresh}: {result.n_clusters} clusters, {result.n_convergent} convergent")
```

### Visualization

```python
from sense_explorer.merger import plot_merger_dendrogram

result = merger.merge_senses("rock")
plot_merger_dendrogram(result, "rock_dendrogram.png",
                       show_threshold_lines=[0.03, 0.05, 0.10])
```
"""

print("Integration code ready for SenseExplorer v0.9.3")
