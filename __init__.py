"""
SenseRepair: Sense Discovery and Disambiguation via Simulated Self-Repair
==========================================================================

A lightweight, training-free method for discovering word senses in static
embeddings (GloVe, Word2Vec, FastText).

Key Features:
    - Zero training required
    - Automatic anchor discovery
    - Noise level as granularity control
    - Stability-based sense number selection
    - Sense-aware similarity metrics

Basic Usage:
    >>> from sense_repair import SenseRepair
    >>> sr = SenseRepair.from_glove("glove.6B.100d.txt")
    >>> senses = sr.discover_senses("bank")
    >>> sr.similarity("bank", "river", sense_aware=True)
    0.818

Stability-based discovery:
    >>> result = sr.discover_senses_stable("bank")
    >>> print(result['stable_k'], result['confidence'])
    2 0.45
"""

from .core import (
    SenseRepair,
    COMMON_POLYSEMOUS,
    load_common_polysemous,
    main,
)

__version__ = "0.2.0"
__author__ = "Kow Kuroda & Claude"
__all__ = [
    "SenseRepair",
    "COMMON_POLYSEMOUS",
    "load_common_polysemous",
]
