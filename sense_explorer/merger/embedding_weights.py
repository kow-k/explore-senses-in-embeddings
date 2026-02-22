#!/usr/bin/env python3
"""
embedding_weights.py - Embedding Quality Weighting for Merger
==============================================================

This module provides methods to compute quality weights for embeddings
and their sense components. These weights are used in the merger to
give more influence to higher-quality sources.

Weighting Philosophy:
    Without weighting, we implicitly treat all embeddings equally.
    This ignores important quality differences:
    
    1. Corpus size: Wikipedia (400K words) vs small domain corpus (10K words)
    2. Sense separation quality: How cleanly senses are separated (R²)
    3. Anchor coherence: How coherent the anchor words are (from IVA)
    4. Domain coverage: Wikipedia covers "river bank"; Twitter covers slang
    5. Vocabulary overlap: More shared words = more reliable comparison

Weight Components:
    - Vocabulary weight: Based on corpus/vocabulary size
    - Coherence weight: Based on anchor/sense coherence (IVA metric)
    - Separation weight: Based on variance explained (R² from geometry)
    - Overlap weight: Based on shared vocabulary with other sources

Usage:
    from sense_explorer.merger.embedding_weights import (
        EmbeddingQualityAssessor,
        compute_sense_weights,
        WeightedSenseComponent
    )
    
    # Assess embedding quality
    assessor = EmbeddingQualityAssessor()
    assessor.add_embedding("wiki", wiki_embeddings, se_wiki)
    assessor.add_embedding("twitter", twitter_embeddings, se_twitter)
    
    # Get weights
    weights = assessor.compute_weights()
    # {'wiki': 0.65, 'twitter': 0.35}
    
    # Use in merger
    result = merger.merge_senses("bank", weights=weights)

Author: Kow Kuroda & Claude (Anthropic)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from enum import Enum
import warnings


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmbeddingQuality:
    """
    Quality assessment for a single embedding source.
    
    All scores are normalized to [0, 1] range.
    """
    name: str
    vocab_size: int
    dimension: int
    
    # Quality metrics (all 0-1)
    vocab_score: float = 0.0       # Based on vocabulary size
    coherence_score: float = 0.0   # Mean anchor coherence across senses
    separation_score: float = 0.0  # Mean R² (variance explained) across senses
    overlap_score: float = 0.0     # Vocabulary overlap with other sources
    
    # Raw metrics for reference
    mean_coherence: float = 0.0    # Raw mean coherence
    mean_r_squared: float = 0.0    # Raw mean R²
    overlap_ratio: float = 0.0     # Raw overlap ratio
    
    # Per-word quality (optional, computed on demand)
    word_qualities: Dict[str, float] = field(default_factory=dict)
    
    @property
    def composite_score(self) -> float:
        """
        Compute weighted composite quality score.
        
        Default weights emphasize coherence and separation (semantic quality)
        over raw vocabulary size.
        """
        weights = {
            'vocab': 0.15,
            'coherence': 0.35,
            'separation': 0.35,
            'overlap': 0.15
        }
        return (
            weights['vocab'] * self.vocab_score +
            weights['coherence'] * self.coherence_score +
            weights['separation'] * self.separation_score +
            weights['overlap'] * self.overlap_score
        )
    
    def composite_score_custom(self, weights: Dict[str, float]) -> float:
        """Compute composite score with custom weights."""
        total = sum(weights.values())
        return (
            weights.get('vocab', 0) * self.vocab_score +
            weights.get('coherence', 0) * self.coherence_score +
            weights.get('separation', 0) * self.separation_score +
            weights.get('overlap', 0) * self.overlap_score
        ) / total if total > 0 else 0.0


@dataclass
class SenseQuality:
    """
    Quality assessment for a specific sense from a specific source.
    """
    sense_id: str
    source: str
    word: str
    
    # Quality metrics
    coherence: float = 0.0      # Anchor coherence for this sense
    separation: float = 0.0     # R² contribution from this sense
    relevance: float = 0.0      # How relevant anchors are to the word
    distinctiveness: float = 0.0  # How distinct from other senses
    
    @property
    def composite_score(self) -> float:
        """Composite sense quality score."""
        return (
            0.3 * self.coherence +
            0.3 * self.separation +
            0.2 * self.relevance +
            0.2 * self.distinctiveness
        )


@dataclass
class WeightedSenseComponent:
    """
    A sense component with quality weight attached.
    
    This extends SenseComponent with weight information for merger.
    """
    sense_id: str
    source: str
    word: str
    vector: np.ndarray
    weight: float  # Quality-based weight (0-1)
    
    # Quality breakdown
    source_weight: float = 0.0   # Weight from embedding quality
    sense_weight: float = 0.0    # Weight from sense-specific quality
    
    # Optional metadata
    top_neighbors: List[Tuple[str, float]] = field(default_factory=list)
    quality_details: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Normalize vector
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm


# =============================================================================
# QUALITY ASSESSMENT
# =============================================================================

class EmbeddingQualityAssessor:
    """
    Assess and compare quality of multiple embedding sources.
    
    This class computes quality scores for each embedding and derives
    weights for use in the merger.
    
    Usage:
        assessor = EmbeddingQualityAssessor()
        assessor.add_embedding("wiki", wiki_emb, se_wiki)
        assessor.add_embedding("twitter", twitter_emb, se_twitter)
        
        # Compute quality for specific words
        assessor.assess_words(["bank", "crane", "bat"])
        
        # Get normalized weights
        weights = assessor.get_weights()
    """
    
    def __init__(
        self,
        vocab_weight: float = 0.15,
        coherence_weight: float = 0.35,
        separation_weight: float = 0.35,
        overlap_weight: float = 0.15,
        verbose: bool = True
    ):
        """
        Initialize assessor with weight configuration.
        
        Args:
            vocab_weight: Weight for vocabulary size score
            coherence_weight: Weight for anchor coherence score
            separation_weight: Weight for sense separation (R²) score
            overlap_weight: Weight for vocabulary overlap score
            verbose: Print progress information
        """
        self.component_weights = {
            'vocab': vocab_weight,
            'coherence': coherence_weight,
            'separation': separation_weight,
            'overlap': overlap_weight
        }
        self.verbose = verbose
        
        # Storage
        self.embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.explorers: Dict[str, Any] = {}  # SenseExplorer instances
        self.qualities: Dict[str, EmbeddingQuality] = {}
        self.sense_qualities: Dict[str, Dict[str, SenseQuality]] = {}  # word -> sense_id -> quality
        
        # Computed values
        self._shared_vocab: set = None
        self._weights: Dict[str, float] = None
    
    def add_embedding(
        self,
        name: str,
        embeddings: Dict[str, np.ndarray],
        sense_explorer: Any = None
    ):
        """
        Add an embedding source for assessment.
        
        Args:
            name: Identifier for this source (e.g., "wikipedia", "twitter")
            embeddings: Dict mapping words to vectors
            sense_explorer: Optional SenseExplorer instance for quality metrics
        """
        self.embeddings[name] = embeddings
        if sense_explorer is not None:
            self.explorers[name] = sense_explorer
        
        # Initialize quality object
        dim = next(iter(embeddings.values())).shape[0] if embeddings else 0
        self.qualities[name] = EmbeddingQuality(
            name=name,
            vocab_size=len(embeddings),
            dimension=dim
        )
        
        # Reset computed values
        self._shared_vocab = None
        self._weights = None
        
        if self.verbose:
            print(f"Added embedding '{name}': {len(embeddings)} words, {dim}d")
    
    def _compute_shared_vocab(self) -> set:
        """Compute vocabulary shared across all embeddings."""
        if len(self.embeddings) == 0:
            return set()
        
        vocabs = [set(emb.keys()) for emb in self.embeddings.values()]
        shared = vocabs[0]
        for v in vocabs[1:]:
            shared = shared.intersection(v)
        
        return shared
    
    @property
    def shared_vocabulary(self) -> set:
        """Get vocabulary shared across all embeddings."""
        if self._shared_vocab is None:
            self._shared_vocab = self._compute_shared_vocab()
        return self._shared_vocab
    
    def assess_vocabulary_scores(self):
        """
        Compute vocabulary-based quality scores.
        
        Uses log-scaled normalization: larger corpora are better,
        but with diminishing returns.
        """
        if len(self.embeddings) == 0:
            return
        
        # Get vocabulary sizes
        sizes = {name: q.vocab_size for name, q in self.qualities.items()}
        
        # Log-scale normalization
        log_sizes = {name: np.log10(max(size, 1)) for name, size in sizes.items()}
        max_log = max(log_sizes.values())
        
        for name, q in self.qualities.items():
            if max_log > 0:
                q.vocab_score = log_sizes[name] / max_log
            else:
                q.vocab_score = 1.0
        
        if self.verbose:
            print("\nVocabulary scores:")
            for name, q in self.qualities.items():
                print(f"  {name}: {q.vocab_size:,} words → score={q.vocab_score:.3f}")
    
    def assess_overlap_scores(self):
        """
        Compute vocabulary overlap scores.
        
        The overlap score reflects how much each embedding contributes to
        the shared vocabulary. Larger embeddings are not penalized for having
        more unique words - what matters is absolute contribution to shared space.
        """
        if len(self.embeddings) < 2:
            for q in self.qualities.values():
                q.overlap_score = 1.0
                q.overlap_ratio = 1.0
            return
        
        shared = self.shared_vocabulary
        n_shared = len(shared)
        
        # Compute union of all vocabularies
        all_vocabs = [set(emb.keys()) for emb in self.embeddings.values()]
        union_vocab = set.union(*all_vocabs)
        n_union = len(union_vocab)
        
        for name, q in self.qualities.items():
            own_vocab = set(self.embeddings[name].keys())
            
            # Raw overlap ratio (for reference)
            q.overlap_ratio = len(shared) / len(own_vocab) if own_vocab else 0.0
            
            # Score based on contribution:
            # - How many shared words does this embedding contain?
            # - Normalized by total embedding size (to favor quality over quantity)
            # 
            # We use: (words_in_shared / total_words) * (own_size / max_size)
            # This balances coverage (how much of own vocab is useful) with scale
            
            # Method: Use shared vocabulary size relative to union
            # All embeddings that contain the shared vocab get high scores
            # This doesn't penalize having extra words
            
            # Simpler approach: All embeddings containing the full shared vocab
            # get score = 1.0. Otherwise proportional to what they contribute.
            own_shared = own_vocab.intersection(shared)
            coverage = len(own_shared) / n_shared if n_shared > 0 else 1.0
            
            # Coverage is typically 1.0 (embedding contains all shared words)
            # So also consider: what fraction of shared vocab comes from this embedding?
            # Larger embeddings contribute more unique context words
            
            # Final score: coverage (typically 1.0) - no penalty for extra words
            q.overlap_score = coverage
        
        if self.verbose:
            print(f"\nOverlap scores (shared vocab: {n_shared:,} words):")
            for name, q in self.qualities.items():
                print(f"  {name}: coverage={q.overlap_score:.1%} (raw ratio={q.overlap_ratio:.1%})")
    
    def assess_coherence_scores(self, words: List[str] = None):
        """
        Compute anchor coherence scores using SenseExplorer.
        
        Requires SenseExplorer instances to be provided via add_embedding().
        
        Args:
            words: Words to assess (defaults to sample of shared vocabulary)
        """
        if not self.explorers:
            warnings.warn("No SenseExplorer instances provided; coherence scores set to default")
            for q in self.qualities.values():
                q.coherence_score = 0.5
                q.mean_coherence = 0.5
            return
        
        # Select words to assess
        if words is None:
            # Sample from shared vocabulary
            shared = list(self.shared_vocabulary)
            words = shared[:min(20, len(shared))]
        
        if not words:
            warnings.warn("No words to assess for coherence")
            return
        
        if self.verbose:
            print(f"\nAssessing coherence on {len(words)} words...")
        
        for name, se in self.explorers.items():
            coherences = []
            
            for word in words:
                if word not in se.vocab:
                    continue
                
                try:
                    # Get anchor coherence if available
                    if hasattr(se, 'measure_anchor_coherence'):
                        coh_dict = se.measure_anchor_coherence(word)
                        if coh_dict:
                            coherences.extend(coh_dict.values())
                    elif hasattr(se, '_validate_anchors'):
                        # Fallback to anchor validation
                        anchors = se.get_anchors(word)
                        if anchors:
                            quality = se._validate_anchors(word, anchors, warn=False)
                            for info in quality.values():
                                if 'coherence' in info:
                                    coherences.append(info['coherence'])
                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: coherence assessment failed for {word}: {e}")
            
            if coherences:
                mean_coh = np.mean(coherences)
                self.qualities[name].mean_coherence = mean_coh
                # Score: coherence is already 0-1, but apply slight boost for high values
                self.qualities[name].coherence_score = min(1.0, mean_coh * 1.2)
            else:
                self.qualities[name].coherence_score = 0.5
                self.qualities[name].mean_coherence = 0.5
        
        if self.verbose:
            print("Coherence scores:")
            for name, q in self.qualities.items():
                print(f"  {name}: mean={q.mean_coherence:.3f} → score={q.coherence_score:.3f}")
    
    def assess_separation_scores(self, words: List[str] = None):
        """
        Compute sense separation (R²) scores using SenseExplorer geometry.
        
        Requires SenseExplorer instances with geometry support.
        
        Args:
            words: Words to assess (defaults to sample of shared vocabulary)
        """
        if not self.explorers:
            warnings.warn("No SenseExplorer instances provided; separation scores set to default")
            for q in self.qualities.values():
                q.separation_score = 0.5
                q.mean_r_squared = 0.5
            return
        
        # Select words to assess
        if words is None:
            shared = list(self.shared_vocabulary)
            words = shared[:min(20, len(shared))]
        
        if not words:
            warnings.warn("No words to assess for separation")
            return
        
        if self.verbose:
            print(f"\nAssessing separation (R²) on {len(words)} words...")
        
        for name, se in self.explorers.items():
            r_squared_values = []
            
            for word in words:
                if word not in se.vocab:
                    continue
                
                try:
                    # Use localize_senses if available (returns SenseDecomposition)
                    if hasattr(se, 'localize_senses'):
                        decomp = se.localize_senses(word)
                        if hasattr(decomp, 'variance_explained_total'):
                            r_squared_values.append(decomp.variance_explained_total)
                        elif hasattr(decomp, 'r_squared'):
                            r_squared_values.append(decomp.r_squared)
                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: separation assessment failed for {word}: {e}")
            
            if r_squared_values:
                mean_r2 = np.mean(r_squared_values)
                self.qualities[name].mean_r_squared = mean_r2
                self.qualities[name].separation_score = mean_r2  # R² is already 0-1
            else:
                self.qualities[name].separation_score = 0.5
                self.qualities[name].mean_r_squared = 0.5
        
        if self.verbose:
            print("Separation (R²) scores:")
            for name, q in self.qualities.items():
                print(f"  {name}: mean R²={q.mean_r_squared:.3f} → score={q.separation_score:.3f}")
    
    def assess_all(self, words: List[str] = None):
        """
        Run all quality assessments.
        
        Args:
            words: Words to use for coherence and separation assessment
        """
        if self.verbose:
            print("=" * 60)
            print("EMBEDDING QUALITY ASSESSMENT")
            print("=" * 60)
        
        self.assess_vocabulary_scores()
        self.assess_overlap_scores()
        self.assess_coherence_scores(words)
        self.assess_separation_scores(words)
        
        # Compute composite scores
        if self.verbose:
            print("\n" + "-" * 40)
            print("COMPOSITE QUALITY SCORES")
            print("-" * 40)
            for name, q in self.qualities.items():
                print(f"  {name}: {q.composite_score:.3f}")
                print(f"    vocab={q.vocab_score:.2f}, coherence={q.coherence_score:.2f}, "
                      f"separation={q.separation_score:.2f}, overlap={q.overlap_score:.2f}")
    
    def get_weights(self, normalize: bool = True) -> Dict[str, float]:
        """
        Get quality-based weights for each embedding.
        
        Args:
            normalize: If True, weights sum to 1.0
            
        Returns:
            Dict mapping embedding name to weight
        """
        if not self.qualities:
            return {}
        
        weights = {name: q.composite_score for name, q in self.qualities.items()}
        
        if normalize:
            total = sum(weights.values())
            if total > 0:
                weights = {name: w / total for name, w in weights.items()}
        
        self._weights = weights
        return weights
    
    def get_sense_weights(
        self,
        word: str,
        sense_components: List[Any]  # List of SenseComponent
    ) -> Dict[str, float]:
        """
        Get weights for specific sense components.
        
        Combines embedding-level weights with sense-specific quality.
        
        Args:
            word: The word being analyzed
            sense_components: List of SenseComponent objects
            
        Returns:
            Dict mapping sense_id to weight
        """
        if self._weights is None:
            self.get_weights()
        
        sense_weights = {}
        
        for sc in sense_components:
            source_weight = self._weights.get(sc.source, 0.5)
            
            # Get sense-specific quality if available
            sense_quality = 1.0
            if word in self.sense_qualities:
                sq = self.sense_qualities[word].get(sc.sense_id)
                if sq:
                    sense_quality = sq.composite_score
            
            # Combine: 70% source weight, 30% sense-specific
            combined = 0.7 * source_weight + 0.3 * sense_quality
            sense_weights[sc.sense_id] = combined
        
        # Normalize
        total = sum(sense_weights.values())
        if total > 0:
            sense_weights = {k: v / total for k, v in sense_weights.items()}
        
        return sense_weights


# =============================================================================
# WEIGHTED SIMILARITY COMPUTATION
# =============================================================================

def compute_weighted_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    weight_a: float,
    weight_b: float,
    method: str = "weighted_cosine"
) -> float:
    """
    Compute weighted similarity between two vectors.
    
    Args:
        vec_a, vec_b: Vectors to compare
        weight_a, weight_b: Quality weights for each vector
        method: Similarity method
            - "weighted_cosine": Standard cosine, then weight the result
            - "weight_adjusted": Adjust similarity based on weight difference
            
    Returns:
        Weighted similarity score
    """
    # Normalize vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    vec_a_norm = vec_a / norm_a
    vec_b_norm = vec_b / norm_b
    
    # Base cosine similarity
    cosine_sim = float(np.dot(vec_a_norm, vec_b_norm))
    
    if method == "weighted_cosine":
        # Simple weighted average doesn't change similarity,
        # but we can use weights to adjust confidence
        return cosine_sim
    
    elif method == "weight_adjusted":
        # Adjust similarity based on weight balance
        # If weights are similar, trust the similarity more
        weight_balance = 1 - abs(weight_a - weight_b)
        return cosine_sim * (0.7 + 0.3 * weight_balance)
    
    else:
        return cosine_sim


def compute_weighted_centroid(
    vectors: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Compute weighted centroid of vectors.
    
    Args:
        vectors: List of vectors
        weights: Corresponding weights
        
    Returns:
        Weighted centroid vector (normalized)
    """
    if not vectors:
        raise ValueError("No vectors provided")
    
    if len(vectors) != len(weights):
        raise ValueError("Number of vectors and weights must match")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / total_weight for w in weights]
    
    # Compute weighted sum
    dim = vectors[0].shape[0]
    centroid = np.zeros(dim)
    
    for vec, weight in zip(vectors, weights):
        centroid += weight * vec
    
    # Normalize result
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    
    return centroid


# =============================================================================
# WEIGHTED CLUSTERING SUPPORT
# =============================================================================

def weight_similarity_matrix(
    similarity_matrix: np.ndarray,
    weights: List[float],
    method: str = "symmetric"
) -> np.ndarray:
    """
    Apply weights to a similarity matrix.
    
    This adjusts pairwise similarities based on the quality weights
    of the sources involved.
    
    Args:
        similarity_matrix: Original similarity matrix (n x n)
        weights: Quality weights for each item
        method:
            - "symmetric": sqrt(w_i * w_j) * sim_ij
            - "receiver": w_j * sim_ij (weight by receiving item)
            - "mean": (w_i + w_j) / 2 * sim_ij
            
    Returns:
        Weighted similarity matrix
    """
    n = similarity_matrix.shape[0]
    weights = np.array(weights)
    
    if len(weights) != n:
        raise ValueError(f"Weight count ({len(weights)}) doesn't match matrix size ({n})")
    
    weighted = similarity_matrix.copy()
    
    if method == "symmetric":
        # sqrt(w_i * w_j) preserves symmetry
        weight_matrix = np.sqrt(np.outer(weights, weights))
        weighted = similarity_matrix * weight_matrix
        
    elif method == "receiver":
        # Each column weighted by receiver's weight
        weighted = similarity_matrix * weights.reshape(1, -1)
        
    elif method == "mean":
        # Average weights
        for i in range(n):
            for j in range(n):
                weighted[i, j] = similarity_matrix[i, j] * (weights[i] + weights[j]) / 2
    
    return weighted


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_assess(
    embeddings_dict: Dict[str, Dict[str, np.ndarray]],
    explorers_dict: Dict[str, Any] = None,
    words: List[str] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Quick assessment and weight computation for multiple embeddings.
    
    Args:
        embeddings_dict: {name: embeddings} for each source
        explorers_dict: {name: SenseExplorer} for quality metrics
        words: Words to assess (optional)
        verbose: Print progress
        
    Returns:
        Normalized weights for each embedding
    """
    assessor = EmbeddingQualityAssessor(verbose=verbose)
    
    for name, emb in embeddings_dict.items():
        se = explorers_dict.get(name) if explorers_dict else None
        assessor.add_embedding(name, emb, se)
    
    assessor.assess_all(words)
    return assessor.get_weights()
