# SenseExplorer

**From Sense Discovery to Sense Induction via Simulated Self-Repair**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.9.1-green.svg)](https://github.com/kow-k/sense-explorer)

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

## The Supervision Continuum

The same attractor-following mechanism operates across the entire supervision spectrum — from fully unsupervised to knowledge-guided — and the comparison between modes is itself informative.

| Capability | Method | Supervision | Description |
|------------|--------|-------------|-------------|
| **Sense Discovery (Auto)** | `discover_senses_auto()` | Unsupervised | Geometry decides k and content (spectral, 90%) |
| **Sense Discovery** | `discover_senses(k)` | Semi-supervised | User decides k, geometry decides content |
| **WordNet-Guided Separation** | `separate_senses_wordnet()` | Knowledge-guided | WordNet provides structure, geometry filters |
| **Sense Induction** | `induce_senses()` | Weakly supervised | User-provided anchors guide separation (88%) |
| **Polarity Classification** | `get_polarity()` | Supervised | Seed-guided polarity axis (97%) |
| **Sense Geometry** | `localize_senses()` | Post-analysis | Geometric structure of separated senses |

**What's new in v0.9.1**:
- **WordNet-guided sense separation**: `separate_senses_wordnet()` automatically derives anchors from WordNet synsets — lemmas, hyponyms, hypernyms, glosses — filters for vocabulary presence, and merges synsets whose anchor centroids overlap in embedding space
- **Synset merging**: Iteratively merges the most similar synset pair until geometric separation is sufficient, so WordNet's 18 synsets for "bank" collapse to however many the embedding actually distinguishes
- **Supervision continuum**: The gap between unsupervised and WordNet-guided results directly measures the mismatch between corpus-encoded sense structure and lexicographic sense inventories
- **`explore_senses()` updated**: New `mode='wordnet'` option alongside existing modes
- **CLI**: `--method wordnet` flag in run_synset_mapping.py for WordNet-guided separation with full synset mapping pipeline

**What was new in v0.9.0**:
- **Sense geometry analysis**: `localize_senses()` decomposes word vectors into sense components and reveals the molecular-like angular structure of polysemous embeddings
- **`SenseDecomposition` dataclass**: Rich result object with angles, coefficients, R², dimensional territories, and interference patterns
- **Molecular diagrams**: Publication-quality visualizations of sense geometry
- **Batch analysis**: `analyze_geometry()` runs cross-word comparisons with automatic statistical summaries
- **Key finding**: Inter-sense angles cluster at ~48° (median, 100d), formally analogous to molecular bond geometry

**What was new in v0.8.0**:
- **Attractor-following insight**: Anchor quality, not N, determines sense correctness
- **Anchor validation** (`_validate_anchors()`): Warns before bad anchors silently produce wrong senses
- **n_copies default reduced** from 100 → 30 (~70% computation savings, no accuracy loss)
- **Vectorized self-repair**: Fully NumPy-vectorized noise generation and iteration loops
- **Standalone quality assessment** (`assess_quality()`): Check anchor quality before induction

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
print(senses.keys())  # Synset names, e.g. dict_keys(['depository_financial_institution.n.01', 'bank.n.01'])

# With full diagnostics
senses, details = se.separate_senses_wordnet("bank", return_details=True)
print(f"Merged {details['n_synsets_total']} synsets → {details['n_groups_after_merge']} sense groups")

# WEAKLY SUPERVISED: Anchor-guided induction (88% accuracy)
senses = se.induce_senses("bank")
print(senses.keys())  # dict_keys(['financial', 'river'])

# SUPERVISED: Polarity classification (97% accuracy)
polarity = se.get_polarity("excellent")
print(polarity)  # {'polarity': 'positive', 'score': 0.82, ...}

# Compare any mode via the convenience wrapper
senses_auto = se.explore_senses("bank", mode='discover_auto')
senses_wn   = se.explore_senses("bank", mode='wordnet')
senses_ind  = se.explore_senses("bank", mode='induce')
```

## Why Spectral Clustering?

Spectral clustering validates the **wave superposition view** of meaning:

| Method | 50d | 100d | 200d | 300d | k Selection |
|--------|-----|------|------|------|-------------|
| **Spectral** | **90%** | **80%** | 70% | **80%** | Eigengap |
| X-means | 64% | 76% | **80%** | 76% | BIC |

**Key findings**:
- Spectral wins 3/4 dimensions, with +26% advantage at 50d
- The advantage is strongest where senses are most aliased (compressed)
- This confirms meanings behave like waves requiring frequency decomposition

```
If meanings were points → k-means should suffice
If meanings are waves   → spectral should excel ✓ CONFIRMED
```

## Polarity Classification (97% Accuracy)

Detect positive/negative valence within semantic categories:

```python
# Simple polarity check
polarity = se.get_polarity("wonderful")
# {'polarity': 'positive', 'score': 0.78, 'confidence': 0.92}

# Classify multiple words
result = se.classify_polarity(['good', 'bad', 'happy', 'sad', 'table'])
# {'positive': ['good', 'happy'], 'negative': ['bad', 'sad'], 'neutral': ['table']}

# Use domain-specific polarity (quality, morality, health, etc.)
pf = se.get_polarity_finder(domain='quality')
pf.get_polarity("excellent")  # Uses quality-specific seeds

# Advanced: PolarityFinder directly
from sense_explorer import PolarityFinder
pf = PolarityFinder(se.embeddings)
pf.most_polar_words(top_k=10)
pf.find_polar_opposites("happy")
```

### Available Polarity Domains

| Domain | Positive Pole | Negative Pole |
|--------|---------------|---------------|
| `sentiment` | happy, joy, love | sad, angry, hate |
| `quality` | excellent, superior | poor, inferior |
| `morality` | good, virtuous | evil, wicked |
| `health` | healthy, strong | sick, weak |
| `size` | big, large, huge | small, tiny, little |
| `temperature` | hot, warm | cold, freezing |

## Sense Geometry Analysis (NEW in v0.9.0)

Analyze the geometric structure of separated senses — how sense vectors are arranged around a polysemous word vector:

```python
# Single word: decompose and analyze
decomp = se.localize_senses("bank")
print(f"R² = {decomp.variance_explained_total:.3f}")
print(f"Dominant sense: {decomp.dominant_sense}")
print(f"Coefficient ratio: {decomp.coefficient_ratio:.1f}:1")

# Inter-sense angles
for s1, s2, angle in decomp.angle_pairs:
    print(f"  ∠({s1}, {s2}) = {angle:.1f}°")

# JSON-serializable summary
print(decomp.summary_dict())

# Batch analysis across multiple words
results = se.analyze_geometry(
    ["bank", "cell", "run", "take"],
    save_dir="geometry_output"  # Saves dashboards + summary plots
)

# Work with pre-extracted sense vectors
from sense_explorer.geometry import decompose
decomp = decompose("bank", word_vec, {"financial": vec1, "river": vec2})
```

### Key Finding: Molecular Bond Analogy

Inter-sense angles cluster around a **characteristic scale** (~48° median at 100d), analogous to molecular bond angles:

| Relationship | Typical angle |
|---|---|
| Synonyms / same-sense words | < 30° |
| **Different senses of the same word** | **~35–55° (median ~48°)** |
| Unrelated words | ~90° |

This arises from a **force balance**: context distinctness pushes senses apart (like electron repulsion), while word identity pulls them together (like covalent attraction). The ~48° equilibrium is to distributional semantics what 109.5° is to carbon chemistry.

### Visualizations

`localize_senses()` and `analyze_geometry()` can generate:
- **Molecular diagrams**: Sense vectors radiating from the word vector at true angles
- **Dimension attribution maps**: Which sense "owns" each embedding dimension
- **Interference heatmaps**: Constructive vs. destructive sense interaction
- **Cross-word comparison grids**: Side-by-side geometry across words
- **Angle summary bar charts**: All inter-sense angles at a glance

```python
# Save a full dashboard for one word
from sense_explorer.geometry import plot_word_dashboard
plot_word_dashboard(decomp, "bank_dashboard.png")

# Or save everything at once
results = se.analyze_geometry(["bank", "cell", "run"], save_dir="output/")
```

## The Modes Explained

### 1. Unsupervised Discovery (`discover_senses`)

True sense discovery from distributional data alone:
- **Spectral clustering** (default): 90% accuracy at 50d
- Uses eigengap for automatic k selection
- Generic sense names (`sense_0`, `sense_1`, ...)

```python
# Spectral (default, recommended)
senses = se.discover_senses("bank", n_senses=2)

# Or explicitly specify method
senses = se.discover_senses("bank", n_senses=2, clustering_method='spectral')
senses = se.discover_senses("bank", n_senses=2, clustering_method='xmeans')
senses = se.discover_senses("bank", n_senses=2, clustering_method='kmeans')
```

### 1b. Parameter-Free Discovery (`discover_senses_auto`)

Automatic sense count via eigengap (spectral) or BIC (X-means):
- No need to specify `n_senses`!
- Spectral uses eigengap heuristic
- X-means uses Bayesian Information Criterion

```python
# Spectral + eigengap (default, recommended)
senses = se.discover_senses_auto("bank")
print(f"Found {len(senses)} senses")  # Automatically determined!

# Or use X-means + BIC
senses = se.discover_senses_auto("bank", clustering_method='xmeans')
```

### 2. WordNet-Guided Separation (`separate_senses_wordnet`) — NEW in v0.9.1

Bridges unsupervised discovery and supervised induction. WordNet provides structural guidance (which senses to look for), while embedding geometry determines which senses the corpus actually supports:

- Automatically derives anchors from WordNet synsets (lemmas, hyponyms, hypernyms, glosses)
- Filters for embedding vocabulary presence
- Merges synsets whose anchor centroids overlap in embedding space (cosine > threshold)
- Sense names are WordNet synset names (e.g., `depository_financial_institution.n.01`)
- The gap between unsupervised and WordNet-guided results measures corpus–lexicon mismatch

```python
# Basic usage — let geometry decide what survives
senses = se.separate_senses_wordnet("bank")
print(senses.keys())
# e.g. dict_keys(['depository_financial_institution.n.01', 'bank.n.01'])

# With full diagnostics
senses, details = se.separate_senses_wordnet("bank", return_details=True)
print(f"WordNet has {details['n_synsets_total']} synsets")
print(f"After merging: {details['n_groups_after_merge']} distinct groups")
print(f"Merge history: {details['merge_history']}")  # Which synsets collapsed

# Tune merge aggressiveness
senses = se.separate_senses_wordnet("bank", merge_threshold=0.60)  # More merging
senses = se.separate_senses_wordnet("bank", merge_threshold=0.85)  # Preserve more distinctions

# Restrict to nouns only
senses = se.separate_senses_wordnet("bank", pos_filter='n')
```

**How merging works**: WordNet lists 18 synsets for "bank". Many are too similar in embedding space to separate (e.g., `bank.v.03` "do business with a bank" vs. `bank.v.05` "be in the banking business"). The algorithm iteratively merges the most similar pair until all remaining groups are geometrically distinct. The `merge_threshold` parameter controls this — at 0.70 (default), only groups with cosine similarity < 0.70 survive as separate senses.

### 3. Weakly Supervised Induction (`induce_senses`)

Knowledge-guided sense induction:
- Uses FrameNet frames or WordNet glosses as anchors
- Senses induced toward anchor-defined targets
- Meaningful sense names (`financial`, `river`, ...)
- ~88% accuracy
- Automatic anchor validation warns about low-quality anchors

```python
senses = se.induce_senses("bank")

# Or provide custom anchors
senses = se.induce_senses("bank", anchors={
    "financial": ["money", "account", "loan"],
    "river": ["water", "shore", "stream"]
})

# Check anchor quality before induction
quality = se._validate_anchors("bank", {
    "financial": ["money", "account", "loan"],
    "river": ["water", "shore", "stream"]
})
# Returns per-sense coherence, separation, relevance, and quality rating

# Standalone quality assessment (without full SenseExplorer)
from sense_explorer import HybridAnchorExtractor
extractor = HybridAnchorExtractor(vocab)
anchors, source = extractor.extract("bank")
report = extractor.assess_quality("bank", anchors, embeddings_norm=emb_norm)
print(report['overall'])  # 'good', 'fair', or 'poor'
```

### 4. Supervised Polarity (`get_polarity`)

Polarity classification with seed supervision:
- Requires positive/negative seed words
- Projects words onto polarity axis
- Binary classification with confidence
- ~97% accuracy

```python
polarity = se.get_polarity("terrible")
# {'polarity': 'negative', 'score': -0.71, 'confidence': 0.88}
```

## Theoretical Background

### The Supervision Continuum

All modes share the same attractor-following mechanism but differ in where guidance comes from:

```
Fully Unsupervised    Knowledge-Guided    Weakly Supervised    Fully Supervised    Post-Analysis
       │                     │                    │                   │                  │
  discover_senses()   separate_senses     induce_senses()     get_polarity()   localize_senses()
  discover_senses      _wordnet()                                               analyze_geometry()
    _auto()                  │                    │                   │                  │
       │              WordNet synsets        Anchor targets       Seed labels     Sense vectors →
   No targets         as structural          88% accuracy        97% accuracy    Geometry analysis
   90% accuracy       hints; geometry
   (spectral)         filters what
       │              corpus supports
  Eigengap: auto k          │
  (parameter-free!)   Synset merging:
                      geometry has
                      the final word
```

The comparison between modes is itself informative: the gap between `discover_senses_auto` and `separate_senses_wordnet` for the same word directly measures how well the corpus-encoded sense structure aligns with lexicographic sense inventories.

### Why Spectral Clustering Works

The eigengap criterion answers "how many senses?" the same way spectral analysis answers "how many frequencies?":

```
Eigenvalues:  λ₁ ─ λ₂ ─ λ₃ │ λ₄ ─ λ₅ ─ λ₆
                           │
              connected    GAP    separate
              (same sense)  ↓     (different senses)
                          k = 3
```

X-means (BIC) fails at low dimensions because it assumes Gaussian clusters. Spectral clustering examines graph connectivity, which persists even when geometric separation fails.

### DNA Self-Repair Analogy

All modes on the supervision continuum — discovery, WordNet-guided separation, and induction — use the same mechanism:
1. **Damage** (noise injection): Perturb the embedding
2. **Repair** (self-organization): Settle into stable configurations
3. **Diagnosis** (attractor identification): Observe attractor basins

What differs is the **source of attractors**: spectral clustering discovers them from geometry, WordNet synsets derive them from lexicographic structure, and user-provided anchors define them directly. The mechanism is the same — attractor-following — and the algorithm is agnostic to where the attractors come from.

Critically, the algorithm is **attractor-following**, not space-sampling. Anchor centroids define deterministic attractors, and seeded copies converge to those attractors regardless of how many copies (N) are created. This explains why:

```
N (copies)       → Only reduces variance; even N=3 achieves 100% separation
d (dimensionality) → Does not affect required N; the same N works for 100d and 1024d
Anchor quality    → THE critical factor; determines whether attractors are correct
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
    n_copies=30,                   # Noisy copies (reduced from 100; see below)
    noise_level=0.5,               # Granularity control (0.1-0.8)
    top_k=50,                      # Neighbors for clustering
    clustering_method='spectral',  # 'spectral', 'xmeans', 'kmeans'
    use_hybrid_anchors=True,       # Enable hybrid extraction for induction
    verbose=True
)
```

**Why n_copies=30?** The self-repair algorithm is attractor-following: anchor centroids define the targets, and copies converge regardless of N. Our experiments show N=3 achieves 100% separation across all noise levels and dimensionalities. N=30 provides comfortable variance reduction with ~70% less computation than the previous default of 100.

### Key Methods

| Method | Mode | Description |
|--------|------|-------------|
| `discover_senses(word, n_senses)` | Unsupervised | Sense discovery (spectral default) |
| `discover_senses_auto(word)` | Unsupervised | Parameter-free discovery (eigengap) |
| `separate_senses_wordnet(word)` | Knowledge-guided | WordNet synset-guided separation |
| `induce_senses(word)` | Weakly supervised | Anchor-guided induction |
| `localize_senses(word)` | Geometry | Decompose word vector into sense components |
| `analyze_geometry(words)` | Geometry | Batch cross-word geometry analysis |
| `_validate_anchors(word, anchors)` | Diagnostic | Check anchor quality before induction |
| `explore_senses(word, mode)` | Auto | Convenience wrapper (all modes) |
| `get_polarity(word)` | Supervised | Polarity classification |
| `classify_polarity(words)` | Supervised | Batch polarity |
| `get_polarity_finder(domain)` | Supervised | Advanced polarity ops |
| `similarity(w1, w2)` | - | Sense-aware similarity |
| `disambiguate(word, context)` | - | Context-based disambiguation |

### WordNet-Guided Separation Parameters

```python
se.separate_senses_wordnet(
    word,                        # Target polysemous word
    hyponym_depth=2,             # Levels of hyponym tree to traverse
    merge_threshold=0.70,        # Cosine threshold for synset merging (lower = more merging)
    min_anchors=2,               # Minimum in-vocabulary anchors per synset
    pos_filter=None,             # Restrict to POS: 'n', 'v', 'a', 'r', or None for all
    noise_level=None,            # Override default noise level
    force=False,                 # Force re-separation even if cached
    return_details=False,        # Return (senses, details_dict) tuple
)
```

**`merge_threshold` tuning**: At 0.70 (default), synsets with centroid cosine similarity ≥ 0.70 merge. Lower values (0.50–0.60) produce fewer, coarser senses; higher values (0.80–0.90) preserve more of WordNet's granularity. The comparison across thresholds reveals the granularity structure of the embedding space.

### Spectral Clustering Functions

```python
from sense_explorer import spectral_clustering, find_k_by_eigengap

# Direct spectral clustering
labels, k = spectral_clustering(vectors, k=None, min_k=2, max_k=5)

# Find optimal k via eigengap
k = find_k_by_eigengap(eigenvalues, min_k=2, max_k=5)
```

### HybridAnchorExtractor

```python
from sense_explorer import HybridAnchorExtractor

extractor = HybridAnchorExtractor(vocab)
anchors, source = extractor.extract("bank")       # Extract anchors (manual → FrameNet → WordNet)
quality = extractor.assess_quality(                # Standalone quality check
    "bank", anchors,
    embeddings_norm=emb_norm                       # Optional: enables coherence/separation metrics
)
print(quality['overall'])                          # 'good', 'fair', or 'poor'
print(quality['senses']['financial']['coherence']) # Intra-sense agreement
print(quality['warnings'])                         # Any issues found
```

### PolarityFinder

```python
from sense_explorer import PolarityFinder

pf = PolarityFinder(embeddings, positive_seeds, negative_seeds)
pf.get_polarity(word)           # Single word
pf.classify_words(words)        # Multiple words
pf.find_polar_opposites(word)   # Find antonyms
pf.most_polar_words(top_k=20)   # Extreme words
pf.set_domain('quality')        # Switch domain
pf.evaluate_accuracy(pos, neg)  # Test accuracy
```

### Sense Geometry

```python
from sense_explorer.geometry import (
    decompose,
    SenseDecomposition,
    collect_all_angles,
    plot_word_dashboard,
    plot_molecular_diagram,
    plot_cross_word_comparison,
    plot_angle_summary,
)

# Standalone decomposition (no SenseExplorer needed)
decomp = decompose("bank", word_vector, {"financial": vec1, "river": vec2})
decomp.variance_explained_total   # R²
decomp.coefficients               # Mixing weights α
decomp.angle_pairs                # [(label_i, label_j, angle°), ...]
decomp.coefficient_ratio          # max(|α|) / min(|α|)
decomp.dominant_sense             # Label of strongest sense
decomp.summary_dict()             # JSON-serializable summary

# Cross-word utilities
all_angles = collect_all_angles([decomp1, decomp2, decomp3])

# Visualization (requires matplotlib + scikit-learn)
plot_word_dashboard(decomp, "bank_dashboard.png")
plot_cross_word_comparison([decomp1, decomp2], "comparison.png")
plot_angle_summary([decomp1, decomp2, decomp3], "angles.png")
```

## Version History

- **v0.9.1**: WordNet-guided sense separation (`separate_senses_wordnet`), synset anchor extraction with hyponym drill-down, iterative synset merging by centroid similarity, `explore_senses(mode='wordnet')`, `--method wordnet` CLI support, cache-busting `force` parameter for sweep/oversplit operations
- **v0.9.0**: Sense geometry module (`localize_senses`, `analyze_geometry`), `SenseDecomposition` dataclass, molecular diagrams, cross-word batch analysis
- **v0.8.0**: Attractor-following insight, anchor validation, n_copies 100→30, vectorized self-repair, `assess_quality()`
- **v0.7.0**: Spectral clustering default (90% at 50d), eigengap k selection, `clustering_method` parameter
- **v0.6.0**: X-means for auto k, sense-loyal induction fix, dimensional recovery experiments
- **v0.5.0**: Polarity classification (97% accuracy), domain-specific seeds
- **v0.4.0**: FrameNet anchor extraction (88% accuracy)

## Citation

```bibtex
@article{kuroda2026sense,
  title={Sense separation: From sense mining to sense induction via simulated self-repair over word embeddings},
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
