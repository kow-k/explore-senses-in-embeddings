# SenseRepair

**Discover and disambiguate word senses in static embeddings via simulated self-repair**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## Overview

SenseRepair is a lightweight, training-free method for discovering word senses in static embeddings (GloVe, Word2Vec, FastText). Inspired by **DNA self-repair mechanisms**, it reveals that polysemous words encode multiple senses as **stable attractors** in embedding space.

```python
from sense_repair import SenseRepair

sr = SenseRepair.from_glove("glove.6B.100d.txt")

# Discover senses
senses = sr.discover_senses("bank")
# {'financial': array([...]), 'river': array([...])}

# Sense-aware similarity
sr.similarity("bank", "money")  # 0.827 (vs 0.572 standard)
sr.similarity("bank", "river")  # 0.818 (vs 0.334 standard)
```

## Key Features

- **Zero training required** — Works directly on pre-trained embeddings
- **Automatic anchor discovery** — No manual sense specification needed
- **Noise as granularity control** — Adjust sense resolution like a "semantic zoom"
- **Stability-based sense number selection** — Find the "true" sense count automatically
- **Multiple similarity metrics** — Standard, max-sense, best-match, context-aware
- **Sense-aware analogies** — Better analogy completion
- **Minimal dependencies** — Only NumPy required
- **Fast** — Senses computed once, then cached

## The Insight: Simulated Self-Repair Reveals Sense Structure

Static embeddings encode multiple senses as **stable attractors**—distinct regions in semantic space that resist perturbation. Our method discovers these attractors by simulating a repair process:

1. **Damage**: Add noise to copies of a word's embedding
2. **Repair**: Let copies self-organize toward stable configurations  
3. **Discover**: Different copies settle into different sense attractors

This is analogous to DNA self-repair: damage reveals which states are stable, exposing the underlying structure.

```
        FINANCIAL                          RIVER
            *                                *
          / | \                            / | \
         /  |  \                          /  |  \
        /   |   \                        /   |   \
       /    |    \                      /    |    \
      --------------------------------------------- 
         DEEP BASIN                   SHALLOW BASIN
         (dominant sense)             (minority sense)
         (100% stable)                (100% stable)
```

## Installation

```bash
pip install sense-repair
```

Or install from source:

```bash
git clone https://github.com/kow-k/sense-repair.git
cd sense-repair
pip install -e .
```

## Quick Start

### Load Embeddings

```python
from sense_repair import SenseRepair

# From GloVe
sr = SenseRepair.from_glove("glove.6B.100d.txt")

# From Word2Vec
sr = SenseRepair.from_word2vec("GoogleNews-vectors-negative300.bin")

# From dictionary
sr = SenseRepair.from_dict({"word": [0.1, 0.2, ...], ...})
```

### Discover Senses

```python
# Automatic anchor discovery
senses = sr.discover_senses("bank")
print(senses.keys())  # ['sense_0', 'sense_1']

# With custom anchors (more meaningful sense names)
sr.set_anchors("bank", {
    "financial": ["money", "account", "loan", "deposit"],
    "river": ["river", "shore", "water", "stream"]
})
senses = sr.discover_senses("bank")
print(senses.keys())  # ['financial', 'river']
```

### Compute Similarities

```python
# Standard (sense-blind)
sr.similarity("bank", "river", sense_aware=False)  # 0.334

# Sense-aware (automatic sense selection)
sr.similarity("bank", "river", sense_aware=True)   # 0.818

# With context disambiguation
sr.similarity("bank", "erosion", context=["river", "water", "flood"])  # Uses river sense
sr.similarity("bank", "account", context=["money", "loan", "finance"]) # Uses financial sense
```

### Solve Analogies

```python
# Standard analogy
sr.analogy("bank", "money", "crane", sense_aware=False)
# [('hobby', 0.52), ('travolta', 0.51), ...]  # Nonsense

# Sense-aware analogy
sr.analogy("bank", "money", "crane", sense_aware=True)
# [('building', 0.61), ('heavy', 0.58), ('construction', 0.57), ...]  # Correct!
```

## Noise Level as Granularity Control

A key discovery: **noise level acts as a "semantic zoom" parameter**.

| Noise Level | Granularity | Use Case |
|-------------|-------------|----------|
| 10-20% | Fine-grained | Sub-sense distinctions |
| 30-50% | Standard | Typical WSD-level senses |
| 60-80% | Coarse | Major sense categories |

```python
# Fine-grained discovery
sr_fine = SenseRepair.from_glove("glove.txt", noise_level=0.2)

# Standard discovery (default)
sr_standard = SenseRepair.from_glove("glove.txt", noise_level=0.5)

# Coarse discovery
sr_coarse = SenseRepair.from_glove("glove.txt", noise_level=0.7)

# Or adjust dynamically
sr.set_noise_level(0.3)  # Clears cache and uses new granularity
```

### Why Granularity Matters

Different words require different noise levels:

| Word | Min Noise | Balance | Separation | Interpretation |
|------|-----------|---------|------------|----------------|
| mouse | 5% | 0.965 | 0.575 | Well-separated, balanced senses |
| bank | 50% | 0.555 | 0.592 | Financial sense dominates |
| spring | 70% | 0.686 | 0.369 | Senses are close in space |

**Correlations with minimal noise**:
- Sense separation: r = -0.730 (strong)
- Balance: r = -0.569 (moderate)

## Stability-Based Sense Discovery

The "true" sense count is found where it remains **stable across noise levels**.

```python
# Stability-based discovery (recommended for unknown words)
result = sr.discover_senses_stable("bank")

print(result['stable_k'])      # 2 (stable sense count)
print(result['confidence'])    # 0.45 (45% of noise range is stable)
print(result['optimal_noise']) # 0.35 (best noise level)
print(result['stable_range'])  # (0.2, 0.5)
print(result['senses'])        # {'sense_0': array([...]), 'sense_1': array([...])}
```

### How It Works

1. Run sense discovery at multiple noise levels (10%, 15%, ..., 70%)
2. Find the longest consecutive run with the same sense count
3. Use the middle of the stable range for final discovery

**Principle**: If structure persists under perturbation, it's real.

This is analogous to:
- **Scale-space theory** in image processing
- **Bootstrap methods** in statistics
- **Cross-validation** in machine learning

## Predefined Anchors

For common polysemous words, predefined anchors are available:

```python
from sense_repair import SenseRepair, load_common_polysemous

sr = SenseRepair.from_glove("glove.6B.100d.txt")
load_common_polysemous(sr)  # Loads anchors for common words
```

**Supported words**: `bank`, `bat`, `cell`, `crane`, `mouse`, `plant`, `spring`, `bass`, `match`, `bow`

## API Reference

### SenseRepair

```python
class SenseRepair:
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        dim: int = None,
        default_n_senses: int = 2,
        n_copies: int = 100,
        noise_level: float = 0.5,      # Granularity control
        seed_strength: float = 0.3,
        n_iterations: int = 15,
        anchor_pull: float = 0.2,
        n_anchors: int = 8,
        verbose: bool = True
    )
```

### Core Methods

| Method | Description |
|--------|-------------|
| `discover_senses(word, n_senses=None, anchors=None, noise_level=None)` | Discover sense-specific embeddings |
| `discover_senses_stable(word, n_senses=None, anchors=None)` | Stability-based sense discovery |
| `similarity(w1, w2, sense_aware=True, context=None)` | Compute similarity |
| `max_sense_similarity(w1, w2)` | Max-sense similarity with sense name |
| `best_match_similarity(w1, w2)` | Best sense-pair similarity |
| `context_similarity(w1, w2, context)` | Context-disambiguated similarity |
| `analogy(a, b, c, top_k=10, sense_aware=True)` | Solve analogy |

### Utility Methods

| Method | Description |
|--------|-------------|
| `set_anchors(word, anchors)` | Set custom anchors for a word |
| `set_noise_level(noise_level)` | Change granularity (clears cache) |
| `get_senses(word)` | Get discovered sense names |
| `get_sense_embedding(word, sense)` | Get specific sense embedding |
| `get_anchors(word)` | Get anchors used for a word |
| `get_stability_info(word)` | Get stability analysis results |
| `clear_cache()` | Clear all cached senses |

## Command-Line Interface

```bash
# Discover senses
python -m sense_repair --glove glove.6B.100d.txt --word bank

# Stability-based discovery
python -m sense_repair --glove glove.6B.100d.txt --word bank --stable

# Control granularity
python -m sense_repair --glove glove.6B.100d.txt --word bank --noise 0.3

# Compute similarity
python -m sense_repair --glove glove.6B.100d.txt --similarity bank river

# Solve analogy
python -m sense_repair --glove glove.6B.100d.txt --analogy bank money crane

# Use predefined anchors
python -m sense_repair --glove glove.6B.100d.txt --word bank --use-predefined
```

## Experimental Results

### Similarity Improvement

| Pair | Standard | Sense-Aware | Improvement |
|------|----------|-------------|-------------|
| bank/river | 0.334 | 0.818 | **+0.484** |
| cell/jail | 0.389 | 0.837 | **+0.448** |
| crane/bird | 0.337 | 0.746 | **+0.409** |
| crane/construction | 0.415 | 0.823 | **+0.408** |
| bat/cave | 0.263 | 0.593 | **+0.330** |

**Average improvement: +0.28** across all test pairs.

### Sense Selection Accuracy

- **24/24 (100%)** correct sense selection on polysemous word pairs
- **9/9 (100%)** correct context-based disambiguation

### Stability Analysis

| Word | Stable Range | Stable K | Confidence |
|------|--------------|----------|------------|
| bat | 10%-30% | 3 | High |
| cell | 20%-50% | 5 | High |
| plant | 20%-60% | 6 | High |
| spring | 20%-50% | 4 | Medium |

## How It Works

### The Algorithm

```
1. SEED:   Create N noisy copies, seed subsets toward different sense anchors
2. REPAIR: Iteratively pull copies toward nearest sense centroid
3. SETTLE: Allow copies to find stable positions (decay pull strength)
4. CLUSTER: Group copies by final sense → sense-specific embeddings
```

### Why "Simulated" Self-Repair?

The term "simulated" is important: we artificially create copies and force them through a repair-like process. This is not natural self-repair but a probe that reveals latent structure—like using a stain to reveal cell structure under a microscope.

### Energy Landscape Interpretation

```
    NOISE LEVEL = EXPLORATION ENERGY
    ─────────────────────────────────────────────
      Low   → Explore local texture (may over-split)
      Medium → Cross real sense boundaries
      High  → Random exploration (may under-split)
    ─────────────────────────────────────────────
```

The stable range is where the "true" structure lives.

## Citation

If you use SenseRepair in your research, please cite:

```bibtex
@article{kuroda2026sense,
  title={Sense Discovery via Simulated Self-Repair: 
         Revealing Latent Semantic Attractors in Word Embeddings},
  author={Kuroda, Kow and Claude},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## Related Work

- **Word Sense Disambiguation**: [Navigli, 2009](https://dl.acm.org/doi/10.1145/1459352.1459355)
- **Multi-prototype embeddings**: [Reisinger & Mooney, 2010](https://aclanthology.org/N10-1013/)
- **GloVe**: [Pennington et al., 2014](https://aclanthology.org/D14-1162/)
- **BERT**: [Devlin et al., 2019](https://aclanthology.org/N19-1423/)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Authors

- **Kow Kuroda** — Kyorin University — [kow.k@ks.kyorin-u.ac.jp](mailto:kow.k@ks.kyorin-u.ac.jp)
- **Claude** — Anthropic (AI Research Assistant)

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

Ideas for contribution:
- Support for more embedding formats (FastText, etc.)
- Visualization tools for sense structure
- Integration with downstream NLP tasks
- Benchmarking on standard WSD datasets
- Multi-lingual evaluation
