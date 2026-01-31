# SenseExplorer

**From Sense Discovery to Sense Induction via Simulated Self-Repair**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.7.0-green.svg)](https://github.com/kow-k/sense-explorer)

A lightweight, training-free framework for exploring word sense structure in static embeddings (GloVe, Word2Vec, FastText).

## Key Insight: Meanings are Wave-like

Word senses are **superposed like waves** in embedding space. This is validated by:
- Spectral clustering (wave decomposition) **outperforms** statistical clustering (BIC)
- 90% vs 64% accuracy at 50d — the advantage is strongest where aliasing is worst
- The eigengap criterion answers "how many senses?" like spectral analysis answers "how many frequencies?"

## Three Capabilities

| Capability | Method | Supervision | Accuracy |
|------------|--------|-------------|----------|
| **Sense Discovery** | `discover_senses()` | Unsupervised | 90% (spectral) |
| **Sense Discovery (Auto)** | `discover_senses_auto()` | Unsupervised + Parameter-free | 90% (spectral) |
| **Sense Induction** | `induce_senses()` | Weakly supervised | 88% |
| **Polarity Classification** | `get_polarity()` | Supervised | 97% |

**What's new in v0.7.0**:
- **Spectral clustering** is now the default method (eigengap-based k selection)
- Dramatically improved unsupervised discovery: 90% vs 64% (X-means) at 50d
- New `clustering_method` parameter: `'spectral'` (default), `'xmeans'`, `'kmeans'`
- Configurable `top_k` parameter for neighbor count (default: 50)

## Installation

```bash
pip install sense-explorer

# For full functionality (WordNet gloss extraction):
pip install sense-explorer[full]
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
se = SenseExplorer.from_glove("glove.6B.100d.txt")
# Output: SenseExplorer v0.7.0 initialized with 400,000 words, dim=100
#         Clustering method: spectral (top_k=50)

# UNSUPERVISED: Sense discovery with spectral clustering
senses = se.discover_senses("bank", n_senses=2)
print(senses.keys())  # dict_keys(['sense_0', 'sense_1'])

# PARAMETER-FREE: Eigengap auto-discovers optimal sense count
senses = se.discover_senses_auto("bank")  # No n_senses needed!
print(f"Found {len(senses)} senses")  # Automatically determined via eigengap

# WEAKLY SUPERVISED: Anchor-guided induction (88% accuracy)
senses = se.induce_senses("bank")
print(senses.keys())  # dict_keys(['financial', 'river'])

# SUPERVISED: Polarity classification (97% accuracy)
polarity = se.get_polarity("excellent")
print(polarity)  # {'polarity': 'positive', 'score': 0.82, ...}

# Compare clustering methods
senses_spectral = se.discover_senses("bank", clustering_method='spectral')  # 90% at 50d
senses_xmeans = se.discover_senses("bank", clustering_method='xmeans')      # 64% at 50d
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

## The Three Modes Explained

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

### 2. Weakly Supervised Induction (`induce_senses`)

Knowledge-guided sense induction:
- Uses FrameNet frames or WordNet glosses as anchors
- Senses induced toward anchor-defined targets
- Meaningful sense names (`financial`, `river`, ...)
- ~88% accuracy

```python
senses = se.induce_senses("bank")

# Or provide custom anchors
senses = se.induce_senses("bank", anchors={
    "financial": ["money", "account", "loan"],
    "river": ["water", "shore", "stream"]
})
```

### 3. Supervised Polarity (`get_polarity`)

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

### The Supervision Spectrum

```
Fully Unsupervised    Weakly Supervised    Fully Supervised
       │                     │                    │
  discover_senses()    induce_senses()     get_polarity()
  discover_senses_auto()
       │                     │                    │
   No targets          Anchor targets        Seed labels
   90% accuracy        88% accuracy          97% accuracy
   (spectral)
       │
  Eigengap: auto k
  (parameter-free!)
```

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

Both sense discovery and induction use the same mechanism:
1. **Damage** (noise injection): Perturb the embedding
2. **Repair** (self-organization): Settle into stable configurations
3. **Diagnosis** (attractor identification): Observe attractor basins

## API Reference

### SenseExplorer

```python
SenseExplorer(
    embeddings,                    # Dict[str, np.ndarray]
    dim=None,                      # Auto-detected
    default_n_senses=2,            # Default sense count
    noise_level=0.5,               # Granularity control (0.1-0.8)
    top_k=50,                      # Neighbors for clustering (NEW)
    clustering_method='spectral',  # 'spectral', 'xmeans', 'kmeans' (NEW)
    use_hybrid_anchors=True,       # Enable hybrid extraction for induction
    verbose=True
)
```

### Key Methods

| Method | Mode | Description |
|--------|------|-------------|
| `discover_senses(word, n_senses)` | Unsupervised | Sense discovery (spectral default) |
| `discover_senses_auto(word)` | Unsupervised | Parameter-free discovery (eigengap) |
| `induce_senses(word)` | Weakly supervised | Anchor-guided induction |
| `explore_senses(word, mode)` | Auto | Convenience wrapper |
| `get_polarity(word)` | Supervised | Polarity classification |
| `classify_polarity(words)` | Supervised | Batch polarity |
| `get_polarity_finder(domain)` | Supervised | Advanced polarity ops |
| `similarity(w1, w2)` | - | Sense-aware similarity |
| `disambiguate(word, context)` | - | Context-based disambiguation |

### Spectral Clustering Functions

```python
from sense_explorer import spectral_clustering, find_k_by_eigengap

# Direct spectral clustering
labels, k = spectral_clustering(vectors, k=None, min_k=2, max_k=5)

# Find optimal k via eigengap
k = find_k_by_eigengap(eigenvalues, min_k=2, max_k=5)
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

## Version History

- **v0.7.0**: Spectral clustering default (90% at 50d), eigengap k selection, `clustering_method` parameter
- **v0.6.0**: X-means for auto k, sense-loyal induction fix, dimensional recovery experiments
- **v0.5.0**: Polarity classification (97% accuracy), domain-specific seeds
- **v0.4.0**: FrameNet anchor extraction (88% accuracy)

## Citation

```bibtex
@article{kuroda2026sense,
  title={From sense mining to sense induction via simulated self-repair:
         Revealing latent semantic attractors in word embeddings},
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
