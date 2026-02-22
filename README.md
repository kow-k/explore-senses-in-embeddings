# SenseExplorer

**From Sense Discovery to Sense Induction via Simulated Self-Repair**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.9.3-green.svg)](https://github.com/kow-k/sense-explorer)

A lightweight, training-free framework for exploring word sense structure in static embeddings (GloVe, Word2Vec, FastText).

## Key Insight: Meanings are Wave-like

Word senses are **superposed like waves** in embedding space. This is validated by:
- Spectral clustering (wave decomposition) **outperforms** statistical clustering (BIC)
- 90% vs 64% accuracy at 50d — the advantage is strongest where aliasing is worst
- The eigengap criterion answers "how many senses?" like spectral analysis answers "how many frequencies?"

## Key Insight: Self-Repair is Attractor-Following

The self-repair algorithm does not sample the embedding space—it **follows attractors** defined by anchor centroids. This means:
- **Anchor quality determines correctness**, not the number of noisy copies (N)
- N=20–50 suffices regardless of embedding dimensionality (100d, 300d, 1024d)
- Random anchors yield separated but **semantically wrong** senses (100% separation, 6.4% alignment)
- Good anchors yield both separation **and** correctness (100% separation, 99.9% alignment)

## Key Insight: SSR Separates, IVA Distills

Two complementary operations for understanding sense structure:

| Operation | Method | Input | Output | Question Answered |
|-----------|--------|-------|--------|-------------------|
| **Sense Separation** | SSR (Self-Repair) | A polysemous word | Multiple sense vectors | "What senses does this word have?" |
| **Concept Distillation** | IVA (Iterative Vector Averaging) | A set of words | Shared direction | "What do these words share?" |

SSR **discovers** structure within a word. IVA **extracts** structure across words.
Together, they form a complete pipeline: SSR → anchors → IVA → purified sense directions.

## Key Insight: Sense Separation Enables Embedding Merger

**Embedding merger** combines multiple embeddings (Wikipedia, Twitter, News) into a unified semantic space. The critical insight: **sense separation is a prerequisite**.

```
Without sense separation:
  bank (wiki) = 0.6·financial + 0.3·river    ─┐
  bank (twitter) = 0.7·financial + 0.2·slang ─┴→ messier superposition ✗

With sense separation first:
  wiki: bank_financial, bank_river           ─┐
  twitter: bank_financial, bank_slang        ─┴→ meaningful alignment:
                                                  • bank_financial converges ✓
                                                  • bank_river (wiki-specific) ✓
                                                  • bank_slang (twitter-specific) ✓
```

This models how humans build unified lexicons from diverse linguistic experiences.

## The Supervision Continuum

The same attractor-following mechanism operates across the entire supervision spectrum — from fully unsupervised to knowledge-guided — and the comparison between modes is itself informative.

| Capability | Method | Supervision | Description |
|------------|--------|-------------|-------------|
| **Sense Discovery (Auto)** | `discover_senses_auto()` | Unsupervised | Geometry decides k and content (spectral, 90%) |
| **Sense Discovery** | `discover_senses(k)` | Semi-supervised | User decides k, geometry decides content |
| **WordNet-Guided Separation** | `separate_senses_wordnet()` | Knowledge-guided | WordNet provides structure, geometry filters |
| **Sense Induction** | `induce_senses()` | Weakly supervised | User-provided anchors guide separation (88%) |
| **Sense Distillation** | `distill_senses()` | Post-separation | IVA extracts shared essence from anchor sets |
| **Embedding Merger** | `merge_with()` | Cross-embedding | Align senses across multiple embeddings |
| **Polarity Classification** | `get_polarity()` | Supervised | Seed-guided polarity axis (97%) |
| **Sense Geometry** | `localize_senses()` | Post-analysis | Geometric structure of separated senses |

**What's new in v0.9.3**:
- **Embedding merger subpackage**: `sense_explorer.merger` provides clean separation between basic (sense separation) and advanced (cross-embedding merger) functionality
- **Spectral-hierarchical clustering**: Hybrid clustering method for merger combines wave-aware spectral embedding with hierarchical dendrogram structure
- **Staged merger for large embeddings**: `StagedMerger` class with affinity-based merge ordering for memory-efficient processing
- **Complete test suite**: Comprehensive tests for all modules with toy data and real embedding support
- **IVA distillation module**: `distillation.py` implements sense distillation via Iterative Vector Averaging

**What's new in v0.9.2**:
- **Sense distillation via IVA**: `distill_senses()` applies Iterative Vector Averaging to extract the shared semantic essence from anchor sets
- **Two distillation modes**: Global IVA (may drift to vocabulary attractors) and Constrained IVA (stays cluster-specific, recommended)
- **Anchor coherence measurement**: `measure_anchor_coherence()` validates anchor set quality before distillation

**What was new in v0.9.1**:
- **WordNet-guided sense separation**: `separate_senses_wordnet()` automatically derives anchors from WordNet synsets
- **Synset merging**: Iteratively merges the most similar synset pair until geometric separation is sufficient

**What was new in v0.9.0**:
- **Sense geometry analysis**: `localize_senses()` decomposes word vectors into sense components
- **Key finding**: Inter-sense angles cluster at ~48° (median, 100d), formally analogous to molecular bond geometry

## Installation

```bash
pip install sense-explorer

# For full functionality (WordNet-guided separation + gloss extraction):
pip install sense-explorer[full]

# WordNet data (required for separate_senses_wordnet):
python -c "import nltk; nltk.download('wordnet')"
```

Or from source:

```bash
git clone https://github.com/kow-k/sense-explorer.git
cd sense-explorer
pip install -e .
```

## Quick Start

```python
from sense_explorer import SenseExplorer

# Load embeddings (spectral clustering is now default!)
se = SenseExplorer.from_glove("glove.6B.300d.txt")

# UNSUPERVISED: Sense discovery with spectral clustering
senses = se.discover_senses("bank", n_senses=2)
print(senses.keys())  # dict_keys(['sense_0', 'sense_1'])

# PARAMETER-FREE: Eigengap auto-discovers optimal sense count
senses = se.discover_senses_auto("bank")  # No n_senses needed!
print(f"Found {len(senses)} senses")  # Automatically determined via eigengap

# WORDNET-GUIDED: Lexicographic structure + geometric filtering
senses = se.separate_senses_wordnet("bank")
print(senses.keys())  # Synset names as keys

# WEAKLY SUPERVISED: Anchor-guided induction (88% accuracy)
senses = se.induce_senses("bank")
print(senses.keys())  # dict_keys(['financial', 'river'])

# SENSE DISTILLATION: Extract shared essence via IVA
results = se.distill_senses("bank")
for sense, result in results.items():
    print(f"{sense}: coherence={result.coherence:.3f}, exemplars={result.exemplars}")

# EMBEDDING MERGER: Combine multiple embeddings
from sense_explorer.merger import EmbeddingMerger
se_twitter = SenseExplorer.from_file("glove.twitter.100d.txt")
result = se.merge_with(se_twitter, "bank")
print(f"Convergent: {result.n_convergent}, Source-specific: {result.n_source_specific}")

# SUPERVISED: Polarity classification (97% accuracy)
polarity = se.get_polarity("excellent")
print(polarity)  # {'polarity': 'positive', 'score': 0.82, ...}
```

## Package Structure

```
sense_explorer/
├── __init__.py              # Package exports (basic functionality)
├── core.py                  # SenseExplorer main class (SSR)
├── spectral.py              # Spectral clustering + eigengap
├── geometry.py              # Sense decomposition + visualization
├── anchor_extractor.py      # FrameNet/WordNet anchor extraction
├── polarity.py              # Polarity classification
├── distillation.py          # IVA concept distillation
│
├── merger/                  # Advanced: Cross-embedding merger (subpackage)
│   ├── __init__.py          # Merger exports
│   ├── embedding_merger.py  # Core merger with spectral-hierarchical clustering
│   └── staged_embedding_merger.py  # Memory-efficient staged merger
│
└── tests/                   # Comprehensive test suite
    ├── __init__.py
    ├── run_all_tests.py     # Test runner
    ├── test_core.py         # Core SenseExplorer tests
    ├── test_spectral.py     # Spectral clustering tests
    ├── test_geometry.py     # Sense geometry tests
    ├── test_polarity.py     # Polarity classification tests
    ├── test_distillation.py # IVA distillation tests
    ├── test_merger.py       # Embedding merger tests
    └── test_staged_merger.py # Staged merger tests
```

### Architectural Design

The package separates **basic** and **advanced** functionality:

| Layer | Location | Purpose | Dependency |
|-------|----------|---------|------------|
| **Basic** | `sense_explorer/` | Sense separation from ONE embedding | Standalone |
| **Advanced** | `sense_explorer/merger/` | Combine senses from MULTIPLE embeddings | Requires basic |

This ensures that:
- Basic functionality works without merger dependencies
- Advanced functionality explicitly depends on sense separation
- Import paths clearly indicate capability level

## Testing

Run the comprehensive test suite:

```bash
# Run all tests with toy data (no external files needed)
python tests/run_all_tests.py

# Run individual module test
python tests/test_core.py

# Run with real embeddings
python tests/test_core.py --glove path/to/glove.txt

# Run merger tests with multiple embeddings
python tests/run_all_tests.py --wiki wiki.txt --twitter twitter.txt

# Skip specific tests
python tests/run_all_tests.py --skip merger staged_merger
```

### Test Coverage

| Test File | Module | Key Tests |
|-----------|--------|-----------|
| `test_core.py` | `core.py` | Initialization, discover_senses, induce_senses, similarity |
| `test_spectral.py` | `spectral.py` | Spectral clustering, eigengap k-selection |
| `test_geometry.py` | `geometry.py` | Decomposition, angle computation, coefficient ratio |
| `test_polarity.py` | `polarity.py` | Classification, batch classify, polar opposites |
| `test_distillation.py` | `distillation.py` | IVA distillation, global vs constrained, coherence |
| `test_merger.py` | `embedding_merger.py` | Clustering methods, spectral analysis, thresholds |
| `test_staged_merger.py` | `staged_embedding_merger.py` | Staging, affinity, memory management |

## Embedding Merger

Combine word embeddings from multiple sources into a unified semantic space.

### Import Patterns

```python
# Basic functionality (from main package)
from sense_explorer import SenseExplorer

# Advanced functionality (from merger subpackage)
from sense_explorer.merger import (
    EmbeddingMerger,
    SenseComponent,
    MergerResult,
    SpectralInfo,
    plot_merger_dendrogram,
    plot_spectral_analysis,
)

# Staged merger for large embeddings
from sense_explorer.merger import StagedMerger, MergeStrategy
```

### Quick Start

```python
from sense_explorer import SenseExplorer
from sense_explorer.merger import EmbeddingMerger

# Load embeddings
se_wiki = SenseExplorer.from_file("glove-wiki-100d.txt")
se_twitter = SenseExplorer.from_file("glove-twitter-100d.txt")

# Simple two-way merge (convenience method)
result = se_wiki.merge_with(se_twitter, "bank")
print(f"Convergent: {result.n_convergent}")
print(f"Source-specific: {result.n_source_specific}")

# Or use EmbeddingMerger directly for more control
merger = EmbeddingMerger(
    clustering_method='spectral_hierarchical',  # Wave-aware + dendrogram
    verbose=True
)
merger.add_embedding("wiki", se_wiki.embeddings)
merger.add_embedding("twitter", se_twitter.embeddings)

result = merger.merge_senses("bank", n_senses=3)
print(merger.report(result))
```

### Clustering Methods

| Method | Description | Output |
|--------|-------------|--------|
| `hierarchical` | Standard agglomerative | Dendrogram, threshold-based |
| `spectral` | Pure spectral with eigengap | Flat clusters, auto-k |
| `spectral_hierarchical` | **Hybrid (recommended)** | Wave-aware + dendrogram |

### Staged Merger for Large Embeddings

```python
from sense_explorer.merger import StagedMerger, MergeStrategy

specs = {
    "wiki": {"path": "glove-wiki-300d.txt", "format": "glove"},
    "twitter": {"path": "glove-twitter-200d.txt", "format": "glove"},
    "news": {"path": "word2vec-news-300d.bin", "format": "word2vec"},
}

merger = StagedMerger(
    specs,
    max_concurrent=2,  # Memory constraint
    strategy=MergeStrategy.AFFINITY
)

plan = merger.plan_merge_order(sample_words=["bank", "rock"])
result = merger.merge_staged("bank", n_senses=3)
```

## Sense Distillation via IVA

IVA (Iterative Vector Averaging) **distills** the shared semantic essence from a word set:

```python
# Via SenseExplorer
results = se.distill_senses("bank")
for sense, result in results.items():
    print(f"{sense}: coherence={result.coherence:.3f}")

# Check anchor quality
coherences = se.measure_anchor_coherence("bank")

# Compare SSR vs IVA
stats = se.distill_and_compare("bank")
print(stats['ssr_iva_angles'])

# Standalone distiller
from sense_explorer.distillation import IVADistiller, distill_concept

distiller = IVADistiller(embeddings)
result = distiller.distill_constrained(['money', 'loan', 'account'])
```

### Coherence Reference Values

| Word Set Type | Coherence |
|---------------|-----------|
| Random words | ~0.15 |
| Topically related | ~0.30-0.40 |
| Sense-coherent groups | ~0.45-0.55 |
| Near-synonyms | ~0.60-0.80 |

## Why Spectral Clustering?

Spectral clustering validates the **wave superposition view** of meaning:

| Method | 50d | 100d | 200d | 300d | k Selection |
|--------|-----|------|------|------|-------------|
| **Spectral** | **90%** | **80%** | 70% | **80%** | Eigengap |
| X-means | 64% | 76% | **80%** | 76% | BIC |

**Key finding**: Spectral wins 3/4 dimensions, with +26% advantage at 50d — confirming meanings behave like waves requiring frequency decomposition.

## Polarity Classification (97% Accuracy)

```python
# Simple polarity
polarity = se.get_polarity("wonderful")

# Batch classification
result = se.classify_polarity(['good', 'bad', 'table'])

# Domain-specific
pf = se.get_polarity_finder(domain='quality')
pf.get_polarity("excellent")
```

### Available Domains

| Domain | Positive | Negative |
|--------|----------|----------|
| `sentiment` | happy, joy, love | sad, angry, hate |
| `quality` | excellent, superior | poor, inferior |
| `morality` | good, virtuous | evil, wicked |

## Sense Geometry Analysis

```python
decomp = se.localize_senses("bank")
print(decomp.variance_explained_total)  # R²
print(decomp.coefficients)              # Mixing weights
print(decomp.angle_pairs)               # Inter-sense angles

# Batch analysis
results = se.analyze_geometry(["bank", "cell", "crane"])
```

**Key finding**: Inter-sense angles cluster around ~48° (median at 100d), suggesting a characteristic "packing angle" for meanings.

## How It Works

The self-repair mechanism was inspired by DNA repair:

```
1. START:    word embedding w (polysemous: river-bank + money-bank superposed)
2. NOISE:    create N noisy copies: w₁, w₂, ..., wₙ
3. SEED:     bias subsets toward different senses via anchor centroids
4. ITERATE:  let copies self-organize toward stable attractors
5. RESULT:   copies cluster around distinct sense embeddings
```

| Anchor Type | Separation | True Alignment | Explanation |
|-------------|-----------|----------------|-------------|
| Curated     | 100%      | 99.9%          | Correct attractors → correct senses |
| Random      | 100%      | 6.4%           | Arbitrary attractors → wrong senses |

## API Reference

### SenseExplorer

```python
SenseExplorer(
    embeddings,                    # Dict[str, np.ndarray]
    dim=None,                      # Auto-detected
    default_n_senses=2,            # Default sense count
    n_copies=30,                   # Noisy copies
    noise_level=0.5,               # Granularity control (0.1-0.8)
    top_k=50,                      # Neighbors for clustering
    clustering_method='spectral',  # 'spectral', 'xmeans', 'kmeans'
    use_hybrid_anchors=True,       # Enable hybrid extraction
    verbose=True
)
```

### Key Methods

| Method | Mode | Description |
|--------|------|-------------|
| `discover_senses(word, n_senses)` | Unsupervised | Sense discovery (spectral) |
| `discover_senses_auto(word)` | Unsupervised | Parameter-free (eigengap) |
| `separate_senses_wordnet(word)` | Knowledge-guided | WordNet synset-guided |
| `induce_senses(word)` | Weakly supervised | Anchor-guided induction |
| `distill_senses(word)` | Distillation | IVA distillation |
| `merge_with(other, word)` | Cross-embedding | Merge with another embedding |
| `get_merger(others)` | Cross-embedding | Get EmbeddingMerger |
| `localize_senses(word)` | Geometry | Decompose into sense components |
| `get_polarity(word)` | Supervised | Polarity classification |

### Merger API

```python
from sense_explorer.merger import (
    EmbeddingMerger,
    SenseComponent,
    MergerResult,
    SpectralInfo,
    ClusteringMethod,
    create_merger_from_explorers,
    merge_with_ssr,
    plot_merger_dendrogram,
    plot_spectral_analysis,
)

# Staged merger
from sense_explorer.merger import (
    StagedMerger,
    MergeStrategy,
    MergePlan,
    StagedMergeResult,
)
```

### Distillation API

```python
from sense_explorer.distillation import (
    IVADistiller,
    DistillationResult,
    distill_concept,
    measure_set_coherence,
    validate_distillation,
)
```

## Version History

- **v0.9.3**: Merger subpackage (`sense_explorer.merger`), spectral-hierarchical clustering, staged merger, comprehensive test suite, distillation module
- **v0.9.2**: Sense distillation via IVA, constrained/global modes, anchor coherence measurement
- **v0.9.1**: WordNet-guided sense separation, synset merging
- **v0.9.0**: Sense geometry module, SenseDecomposition dataclass, molecular diagrams
- **v0.8.0**: Attractor-following insight, anchor validation, vectorized self-repair
- **v0.7.0**: Spectral clustering default (90% at 50d), eigengap k selection
- **v0.6.0**: X-means for auto k, sense-loyal induction fix
- **v0.5.0**: Polarity classification (97% accuracy)
- **v0.4.0**: FrameNet anchor extraction (88% accuracy)

## Citation

```bibtex
@article{kuroda2026sense,
  title={Sense separation: From sense mining to sense induction via simulated self-repair over word embeddings},
  author={Kuroda, Kow and Claude},
  journal={arXiv preprint},
  year={2026}
}

@article{kuroda2026merger,
  title={Merging embeddings through sense separation: Constructing unified semantic space from multiple word embeddings via sense alignment},
  author={Kuroda, Kow and Claude},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- **Kow Kuroda** - Kyorin University Medical School
- **Claude** - Anthropic (AI Research Assistant)
