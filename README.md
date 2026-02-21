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

## Key Insight: Sense Separation Enables Embedding Merger (NEW in v0.9.3)

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
| **Embedding Merger** | `merge_with()` | Cross-embedding | Align senses across multiple embeddings (NEW) |
| **Polarity Classification** | `get_polarity()` | Supervised | Seed-guided polarity axis (97%) |
| **Sense Geometry** | `localize_senses()` | Post-analysis | Geometric structure of separated senses |

**What's new in v0.9.3**:
- **Embedding merger**: `merge_with()` combines sense inventories from multiple embeddings (Wikipedia + Twitter + News), identifying convergent (shared) vs source-specific senses
- **Staged merger for large embeddings**: `StagedMerger` class with affinity-based merge ordering for memory-efficient processing when embeddings are too large to load simultaneously
- **Merge strategies**: AFFINITY (merge similar embeddings first), ANCHOR (start with best quality), HIERARCHICAL (dendrogram over embeddings)
- **Visualization**: Dendrograms showing cross-embedding sense hierarchies with automatic threshold analysis

**What's new in v0.9.2**:
- **Sense distillation via IVA**: `distill_senses()` applies Iterative Vector Averaging to extract the shared semantic essence from anchor sets — producing "purified" sense directions
- **Two distillation modes**: Global IVA (may drift to vocabulary attractors) and Constrained IVA (stays cluster-specific, recommended)
- **Anchor coherence measurement**: `measure_anchor_coherence()` validates anchor set quality before distillation (reference: random ~0.15, coherent ~0.45-0.55)
- **SSR↔IVA comparison**: `distill_and_compare()` measures agreement between SSR sense vectors and IVA distilled directions
- **Standalone distiller**: `get_distiller()` returns an `IVADistiller` for custom word set analysis
- **Validation pipeline**: `validate_sense_distillation()` runs batch analysis across multiple words

**What was new in v0.9.1**:
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

# SENSE DISTILLATION: Extract shared essence via IVA (v0.9.2)
results = se.distill_senses("bank")
for sense, result in results.items():
    print(f"{sense}: coherence={result.coherence:.3f}, exemplars={result.exemplars}")
# financial: coherence=0.456, exemplars=['money', 'credit', 'funds', 'loan', 'capital']
# river: coherence=0.423, exemplars=['shore', 'water', 'stream', 'flow', 'creek']

# EMBEDDING MERGER: Combine multiple embeddings (NEW in v0.9.3)
se_twitter = SenseExplorer.from_glove("glove.twitter.100d.txt")
result = se.merge_with(se_twitter, "bank")
print(f"Convergent senses: {result.n_convergent}")
print(f"Source-specific: {result.n_source_specific}")

# SUPERVISED: Polarity classification (97% accuracy)
polarity = se.get_polarity("excellent")
print(polarity)  # {'polarity': 'positive', 'score': 0.82, ...}

# Compare any mode via the convenience wrapper
senses_auto = se.explore_senses("bank", mode='discover_auto')
senses_wn   = se.explore_senses("bank", mode='wordnet')
senses_ind  = se.explore_senses("bank", mode='induce')
```

## Embedding Merger (NEW in v0.9.3)

Combine word embeddings from multiple sources into a unified semantic space.

### Why Merger Requires Sense Separation

Without sense separation, merging embeddings just combines superpositions into messier superpositions. With sense separation:

| Step | What Happens |
|------|--------------|
| 1. Sense Separation | Extract sense components from each embedding via SSR |
| 2. Shared Basis Construction | Find vocabulary where embeddings agree about each sense |
| 3. Projection & Comparison | Project senses onto shared basis, compute similarity |
| 4. Clustering | Identify convergent (multi-source) vs source-specific clusters |

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

# Get detailed analysis
for cluster_id, info in result.analysis.items():
    if info['is_convergent']:
        print(f"Cluster {cluster_id}: CONVERGENT across {info['sources']}")
    else:
        print(f"Cluster {cluster_id}: SOURCE-SPECIFIC to {info['sources']}")
```

### Three-Way Merge (Wiki + Twitter + News)

```python
from sense_explorer.merger import EmbeddingMerger

# Create merger
merger = EmbeddingMerger(verbose=True)
merger.add_embedding("wiki", se_wiki.embeddings)
merger.add_embedding("twitter", se_twitter.embeddings)
merger.add_embedding("news", se_news.embeddings)

# Merge senses
result = merger.merge_senses("bank", n_senses=3, distance_threshold=0.05)
print(merger.report(result))

# Test multiple thresholds
results = merger.merge_senses("bank", return_all_thresholds=True)
for thresh, res in results.items():
    print(f"{thresh:.2f}: {res.n_clusters} clusters, {res.n_convergent} convergent")
```

### Threshold Selection

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| < 0.03 | Fine-grained | Preserve subtle sense distinctions |
| 0.03–0.10 | Balanced | Merge equivalent senses (recommended) |
| > 0.20 | Coarse | Collapse most structure |

### Staged Merger for Large Embeddings

When embeddings are too large to load simultaneously:

```python
from sense_explorer.staged_merger import StagedMerger, MergeStrategy

# Define embeddings (paths only — not loaded yet)
specs = {
    "wiki": {"path": "glove-wiki-300d.txt", "format": "glove"},
    "twitter": {"path": "glove-twitter-200d.txt", "format": "glove"},
    "news": {"path": "word2vec-news-300d.bin", "format": "word2vec"},
    "crawl": {"path": "fasttext-cc-300d.vec", "format": "fasttext"},
}

# Create merger with memory constraint
merger = StagedMerger(
    specs,
    max_concurrent=2,  # Only 2 embeddings in memory at once
    strategy=MergeStrategy.AFFINITY  # Merge most similar first
)

# Plan optimal order (loads only samples)
plan = merger.plan_merge_order(sample_words=["bank", "rock", "plant"])
print(plan)
# Step 1: Load [wiki, twitter] — Highest affinity pair (0.42)
# Step 2: Load [news] — Best affinity to merged (0.38)
# Step 3: Load [crawl] — Best affinity to merged (0.31)

# Execute staged merge
result = merger.merge_staged("bank", n_senses=3)
print(f"Final: {result.final_result.n_convergent} convergent")
print(f"Convergence history: {result.convergence_history}")
```

### Merge Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `AFFINITY` | Merge highest-affinity pairs first | General use (recommended) |
| `ANCHOR` | Start with best-quality embedding | When one embedding is superior |
| `HIERARCHICAL` | Build dendrogram over embeddings | Research/analysis |
| `SEQUENTIAL` | User-specified order | Manual control |

### Visualization

```python
from sense_explorer.merger import plot_merger_dendrogram

result = merger.merge_senses("rock")
plot_merger_dendrogram(
    result, 
    "rock_dendrogram.png",
    show_threshold_lines=[0.03, 0.05, 0.10]
)
```

### MergerResult

```python
@dataclass
class MergerResult:
    word: str
    sense_components: List[SenseComponent]
    similarity_matrix: np.ndarray
    clusters: Dict[str, int]      # sense_id -> cluster_label
    analysis: Dict[int, Dict]     # cluster_id -> {members, sources, is_convergent, ...}
    pairwise_stats: Dict          # (sense_i, sense_j) -> {similarity, core_words, ...}
    threshold_used: float

    # Properties
    n_clusters: int               # Total clusters
    n_convergent: int             # Clusters with senses from multiple sources
    n_source_specific: int        # Clusters with senses from single source
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

## Sense Distillation via IVA (v0.9.2)

IVA (Iterative Vector Averaging) **distills** the shared semantic essence from a word set:

```python
# Full pipeline: SSR separation → IVA distillation
se = SenseExplorer.from_glove("glove.6B.100d.txt")

# Step 1: Separate senses (SSR)
senses = se.induce_senses("bank")

# Step 2: Distill each sense's anchors (IVA)
results = se.distill_senses("bank")

# Check coherence of anchor sets
coherences = se.measure_anchor_coherence("bank")
print(coherences)  # {'financial': 0.456, 'river': 0.423}

# Compare SSR vectors with IVA directions
stats = se.distill_and_compare("bank")
print(stats['ssr_iva_angles'])      # Angle between SSR and IVA per sense
print(stats['iva_inter_sense_angles'])  # Angles between distilled directions
```

### Why Two Modes?

| Mode | Method | Behavior | Use Case |
|------|--------|----------|----------|
| **Constrained** | `distill_constrained()` | Iterates only within given word set | Stay cluster-specific (recommended) |
| **Global** | `distill()` | Iterates over entire vocabulary | May drift to global attractors |

```python
# Constrained (default, recommended)
results = se.distill_senses("bank", mode='constrained')

# Global (may capture broader patterns)
results = se.distill_senses("bank", mode='global')
```

### Standalone Distiller

For custom word set analysis without SSR:

```python
# Get distiller from explorer
distiller = se.get_distiller()

# Distill any word set
result = distiller.distill_constrained(['money', 'loan', 'account', 'deposit'])
print(result.direction)    # Unit vector: the distilled concept
print(result.exemplars)    # Nearest words to the direction
print(result.coherence)    # Input set coherence (0-1)
print(result.n_iterations) # Convergence speed

# Or use standalone function
from sense_explorer.distillation import distill_concept, measure_set_coherence

result = distill_concept(embeddings, ['money', 'loan', 'account'])
coherence = measure_set_coherence(embeddings, ['money', 'loan', 'account'])
```

### Coherence Reference Values

| Word Set Type | Coherence |
|---------------|-----------|
| Random words | ~0.15 |
| Topically related | ~0.30-0.40 |
| Sense-coherent groups | ~0.45-0.55 |
| Near-synonyms | ~0.60-0.80 |

The coherence metric validates the "garbage in, garbage out" principle:
meaningful input → structured output; random input → noise.

### Batch Validation

```python
# Validate across multiple words
stats = se.validate_sense_distillation(['bank', 'crane', 'bat', 'mouse'])

print(f"Mean coherence: {stats['mean_coherence']:.3f}")
print(f"Mean SSR↔IVA angle: {stats['mean_ssr_iva_angle']:.1f}°")
print(f"Mean inter-sense angle: {stats['mean_inter_sense_angle']:.1f}°")
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

## Sense Geometry Analysis (v0.9.0)

Analyze the geometric structure of separated senses:

```python
# Localize senses and get full decomposition
decomp = se.localize_senses("bank")

print(decomp.variance_explained_total)  # R² — how well senses explain the word
print(decomp.coefficients)              # Mixing weights α per sense
print(decomp.angle_pairs)               # Inter-sense angles

# Generate visualizations
se.plot_sense_dashboard("bank", "bank_analysis.png")

# Batch analysis
results = se.analyze_geometry(["bank", "cell", "crane"])
```

**Key finding**: Inter-sense angles cluster around ~48° (median at 100d), suggesting a characteristic "packing angle" for meanings — formally analogous to molecular bond geometry.

## How It Works

The self-repair mechanism was inspired by DNA repair:

```
1. START:    word embedding w (polysemous: river-bank + money-bank superposed)
2. NOISE:    create N noisy copies: w₁, w₂, ..., wₙ
3. SEED:     bias subsets toward different senses via anchor centroids
4. ITERATE:  let copies self-organize toward stable attractors
5. RESULT:   copies cluster around distinct sense embeddings
```

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
| `distill_senses(word)` | Distillation | IVA distillation of sense anchors |
| `distill_and_compare(word)` | Distillation | Compare SSR↔IVA directions |
| `measure_anchor_coherence(word)` | Distillation | Check anchor set quality |
| `get_distiller()` | Distillation | Get standalone IVADistiller |
| `merge_with(other, word)` | Cross-embedding | Merge senses with another embedding (NEW) |
| `get_merger(others)` | Cross-embedding | Get EmbeddingMerger for N embeddings (NEW) |
| `extract_sense_components(word)` | Cross-embedding | Export senses for external merger (NEW) |
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

### Sense Distillation (IVA)

```python
from sense_explorer.distillation import (
    IVADistiller,
    DistillationResult,
    distill_concept,
    measure_set_coherence,
    validate_distillation,
)

# Create distiller
distiller = IVADistiller(
    embeddings,                    # Dict[str, np.ndarray]
    max_iter=50,                   # Maximum iterations
    convergence_threshold=0.9999,  # Cosine threshold for convergence
    top_k_neighbors=50,            # Neighbors for global mode
    verbose=False
)

# Distill a word set (constrained — recommended)
result = distiller.distill_constrained(words, n_exemplars=5)
# Returns: DistillationResult with direction, exemplars, coherence

# Distill a word set (global — may drift)
result = distiller.distill(words, n_exemplars=5)

# Distill multiple groups at once
results = distiller.distill_multiple(
    {'financial': ['money', 'loan'], 'river': ['water', 'shore']},
    mode='constrained'
)

# Convenience functions
result = distill_concept(embeddings, words, mode='constrained')
coherence = measure_set_coherence(embeddings, words)
```

### DistillationResult

```python
@dataclass
class DistillationResult:
    direction: np.ndarray    # Unit vector: the distilled concept
    exemplars: List[str]     # Nearest words to direction
    coherence: float         # Input set coherence (0-1)
    input_words: List[str]   # Words used for distillation
    n_iterations: int        # Iterations until convergence
    mode: str                # 'global' or 'constrained'
```

### Embedding Merger

```python
from sense_explorer.merger import (
    EmbeddingMerger,
    SenseComponent,
    MergerResult,
    MergerBasis,
    create_merger_from_explorers,
    merge_with_ssr,
    plot_merger_dendrogram,
)

# Create merger
merger = EmbeddingMerger(
    neighbor_k=50,           # Neighbors for basis construction
    max_basis_size=40,       # Maximum merger basis size
    default_threshold=0.05,  # Default clustering threshold
    verbose=True
)

# Add embeddings
merger.add_embedding("wiki", wiki_vectors)
merger.add_embedding("twitter", twitter_vectors)

# Merge senses
result = merger.merge_senses(
    word,
    sense_components=None,        # Optional: pre-extracted senses
    n_senses=3,                   # Senses per embedding (if not provided)
    distance_threshold=0.05,      # Clustering threshold
    return_all_thresholds=False,  # Test multiple thresholds
)

# From SenseExplorer instances
merger = create_merger_from_explorers({"wiki": se_wiki, "twitter": se_twitter})

# With SSR sense extraction
result = merge_with_ssr({"wiki": se_wiki, "twitter": se_twitter}, "bank")
```

### Staged Merger (for large embeddings)

```python
from sense_explorer.staged_merger import (
    StagedMerger,
    MergeStrategy,
    MergePlan,
    StagedMergeResult,
)

# Create staged merger
merger = StagedMerger(
    embedding_specs,          # {name: {"path": ..., "format": ...}}
    max_concurrent=2,         # Memory constraint
    strategy=MergeStrategy.AFFINITY,
    verbose=True
)

# Plan optimal merge order
plan = merger.plan_merge_order(sample_words=["bank", "rock"])

# Execute staged merge
result = merger.merge_staged(word, n_senses=3)
```

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

## Package Structure

```
sense_explorer/
├── __init__.py           # Package exports
├── core.py               # SenseExplorer main class (SSR)
├── spectral.py           # Spectral clustering + eigengap
├── geometry.py           # Sense decomposition + visualization
├── anchor_extractor.py   # FrameNet/WordNet anchor extraction
├── polarity.py           # Polarity classification
├── distillation.py       # IVA concept distillation
├── merger.py             # Embedding merger (NEW)
└── staged_merger.py      # Memory-efficient staged merger (NEW)
```

## Version History

- **v0.9.3**: Embedding merger (`merge_with`, `get_merger`), N-way embedding combination, staged merger for large embeddings (`StagedMerger`), affinity-based merge ordering, convergent vs source-specific sense identification, merger dendrograms
- **v0.9.2**: Sense distillation via IVA (`distill_senses`, `get_distiller`), constrained/global modes, anchor coherence measurement, SSR↔IVA comparison, `DistillationResult` dataclass, `validate_sense_distillation()` batch analysis
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
