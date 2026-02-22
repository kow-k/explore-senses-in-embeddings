"""
Procrustes Alignment for Cross-Embedding Integration

Uses empirically discovered invariant words (those with sim≈1, drift≈0 in 
register specificity analysis) as anchor points to learn an orthogonal
transformation that aligns embedding spaces.

Theory:
    Given invariant words W = {w1, w2, ..., wn}
    Source vectors: X = [x1, x2, ..., xn]  (from embedding A)
    Target vectors: Y = [y1, y2, ..., yn]  (from embedding B)
    
    Find orthogonal R that minimizes ||Y - X @ R||²
    
    Solution (Orthogonal Procrustes):
        M = X.T @ Y
        U, S, Vt = SVD(M)
        R = U @ Vt
    
    Then for any word w:
        aligned_vec_A(w) = vec_A(w) @ R  ≈  vec_B(w)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import warnings


@dataclass
class ProcrustesAlignment:
    """Result of Procrustes alignment between two embedding spaces."""
    
    # The learned transformation
    rotation_matrix: np.ndarray  # R: (dim, dim)
    scale_factor: float          # Optional scaling
    
    # Alignment quality metrics
    alignment_error: float       # Mean reconstruction error
    max_error: float             # Max reconstruction error
    anchor_cosines: np.ndarray   # Per-anchor cosine after alignment
    
    # Anchor information
    anchor_words: List[str]
    n_anchors: int
    
    # Source/target info
    source_name: str
    target_name: str
    
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Transform vectors from source space to target space."""
        return vectors @ self.rotation_matrix * self.scale_factor
    
    def transform_word(self, word: str, source_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Transform a single word's embedding."""
        vec = source_embeddings[word]
        return self.transform(vec.reshape(1, -1)).flatten()
    
    def report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            f"PROCRUSTES ALIGNMENT: {self.source_name} → {self.target_name}",
            "=" * 60,
            f"Anchors used: {self.n_anchors}",
            f"Scale factor: {self.scale_factor:.4f}",
            f"",
            f"Alignment quality:",
            f"  Mean error: {self.alignment_error:.4f}",
            f"  Max error:  {self.max_error:.4f}",
            f"  Mean cosine (post-alignment): {self.anchor_cosines.mean():.4f}",
            f"  Min cosine:  {self.anchor_cosines.min():.4f}",
            f"",
            f"Rotation matrix:",
            f"  Shape: {self.rotation_matrix.shape}",
            f"  ||R - I||_F: {np.linalg.norm(self.rotation_matrix - np.eye(self.rotation_matrix.shape[0])):.4f}",
            f"  Orthogonality check ||R @ R.T - I||: {np.linalg.norm(self.rotation_matrix @ self.rotation_matrix.T - np.eye(self.rotation_matrix.shape[0])):.6f}",
        ]
        return "\n".join(lines)


def discover_invariants(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    n_samples: int = 500,
    sim_threshold: float = 0.95,
    drift_threshold: float = 0.05,
    verbose: bool = True
) -> List[str]:
    """
    Discover invariant words between two embeddings.
    
    Invariants are words with high cross-register similarity and low drift,
    indicating stable semantic representation across both sources.
    
    Args:
        se1, se2: SenseExplorer instances for the two embeddings
        n_samples: Number of words to sample for analysis
        sim_threshold: Minimum similarity to qualify as invariant
        drift_threshold: Maximum drift to qualify as invariant
        verbose: Print progress
        
    Returns:
        List of invariant words
    """
    from sense_explorer.merger import create_register_profile
    
    # Get shared vocabulary
    shared_vocab = list(set(se1.vocab) & set(se2.vocab))
    
    if verbose:
        print(f"Shared vocabulary: {len(shared_vocab)} words")
    
    # Sample words
    np.random.seed(42)
    sample_words = np.random.choice(shared_vocab, min(n_samples, len(shared_vocab)), replace=False)
    
    if verbose:
        print(f"Sampling {len(sample_words)} words for invariant discovery...")
    
    explorers = {"source1": se1, "source2": se2}
    invariants = []
    
    for i, word in enumerate(sample_words):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(sample_words)}...")
        
        try:
            profile = create_register_profile(word, explorers, n_senses=2, verbose=False)
            
            sim = profile.overall_register_similarity
            drifts = [sd.mean_drift for sd in profile.sense_drifts.values()]
            mean_drift = np.mean(drifts) if drifts else 0.0
            
            if sim >= sim_threshold and mean_drift <= drift_threshold:
                invariants.append(word)
                
        except Exception:
            continue
    
    if verbose:
        print(f"Found {len(invariants)} invariants (sim >= {sim_threshold}, drift <= {drift_threshold})")
    
    return invariants


def compute_procrustes_alignment(
    source_embeddings: Dict[str, np.ndarray],
    target_embeddings: Dict[str, np.ndarray],
    anchor_words: List[str],
    source_name: str = "source",
    target_name: str = "target",
    use_scaling: bool = True,
    center: bool = True,
    verbose: bool = True
) -> ProcrustesAlignment:
    """
    Compute Procrustes alignment from source to target embedding space.
    
    Args:
        source_embeddings: Dict mapping words to vectors (source space)
        target_embeddings: Dict mapping words to vectors (target space)
        anchor_words: List of words to use as alignment anchors
        source_name: Name of source embedding
        target_name: Name of target embedding
        use_scaling: Whether to include uniform scaling
        center: Whether to center vectors before alignment
        verbose: Print progress
        
    Returns:
        ProcrustesAlignment object with rotation matrix and metrics
    """
    # Filter to words in both vocabularies
    valid_anchors = [w for w in anchor_words 
                     if w in source_embeddings and w in target_embeddings]
    
    if len(valid_anchors) < 3:
        raise ValueError(f"Need at least 3 valid anchors, got {len(valid_anchors)}")
    
    if verbose:
        print(f"Using {len(valid_anchors)} anchors for alignment")
    
    # Get anchor vectors
    X = np.array([source_embeddings[w] for w in valid_anchors])  # (n, dim_src)
    Y = np.array([target_embeddings[w] for w in valid_anchors])  # (n, dim_tgt)
    
    # Handle dimension mismatch - use minimum
    min_dim = min(X.shape[1], Y.shape[1])
    X = X[:, :min_dim]
    Y = Y[:, :min_dim]
    
    if verbose:
        print(f"Working dimension: {min_dim}")
    
    # Center the data (optional but often helps)
    if center:
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
    else:
        X_centered = X
        Y_centered = Y
    
    # Normalize rows for rotation computation
    X_norms = np.linalg.norm(X_centered, axis=1, keepdims=True) + 1e-10
    Y_norms = np.linalg.norm(Y_centered, axis=1, keepdims=True) + 1e-10
    X_normed = X_centered / X_norms
    Y_normed = Y_centered / Y_norms
    
    # Compute optimal rotation via SVD
    # R = argmin ||Y_normed - X_normed @ R||² subject to R.T @ R = I
    M = X_normed.T @ Y_normed  # (dim, dim)
    U, S, Vt = np.linalg.svd(M)
    
    # Optimal orthogonal matrix
    R = U @ Vt
    
    # Ensure proper rotation (det = 1, not -1)
    if np.linalg.det(R) < 0:
        # Flip sign of last column of U
        U[:, -1] *= -1
        R = U @ Vt
    
    # Compute scaling factor if requested
    if use_scaling:
        # Optimal scale: s = trace(Y.T @ X @ R) / trace(X.T @ X)
        X_rotated = X_normed @ R
        scale = np.sum(Y_normed * X_rotated) / (np.sum(X_normed * X_normed) + 1e-10)
        
        # Also account for norm differences
        mean_Y_norm = Y_norms.mean()
        mean_X_norm = X_norms.mean()
        scale *= mean_Y_norm / mean_X_norm
    else:
        scale = 1.0
    
    # Compute alignment quality
    X_aligned = X_normed @ R
    
    # Per-anchor cosine similarity after alignment
    cosines = np.sum(X_aligned * Y_normed, axis=1)
    
    # Reconstruction error
    errors = np.linalg.norm(Y_normed - X_aligned, axis=1)
    
    if verbose:
        print(f"Alignment complete:")
        print(f"  Mean cosine (post-alignment): {cosines.mean():.4f}")
        print(f"  Mean error: {errors.mean():.4f}")
        print(f"  Scale factor: {scale:.4f}")
    
    return ProcrustesAlignment(
        rotation_matrix=R,
        scale_factor=scale,
        alignment_error=errors.mean(),
        max_error=errors.max(),
        anchor_cosines=cosines,
        anchor_words=valid_anchors,
        n_anchors=len(valid_anchors),
        source_name=source_name,
        target_name=target_name
    )


def align_embeddings(
    source_embeddings: Dict[str, np.ndarray],
    alignment: ProcrustesAlignment,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Apply Procrustes alignment to transform all source embeddings.
    
    Args:
        source_embeddings: Dict mapping words to vectors
        alignment: ProcrustesAlignment object
        normalize: Whether to L2-normalize output vectors
        
    Returns:
        Dict mapping words to aligned vectors
    """
    R = alignment.rotation_matrix
    scale = alignment.scale_factor
    dim = R.shape[0]
    
    aligned = {}
    for word, vec in source_embeddings.items():
        # Handle dimension mismatch
        if len(vec) > dim:
            vec = vec[:dim]
        elif len(vec) < dim:
            vec = np.concatenate([vec, np.zeros(dim - len(vec))])
        
        # Apply transformation
        aligned_vec = vec @ R * scale
        
        if normalize:
            norm = np.linalg.norm(aligned_vec)
            if norm > 1e-10:
                aligned_vec = aligned_vec / norm
        
        aligned[word] = aligned_vec
    
    return aligned


def evaluate_alignment(
    aligned_embeddings: Dict[str, np.ndarray],
    target_embeddings: Dict[str, np.ndarray],
    test_words: List[str] = None,
    n_test: int = 1000
) -> Dict:
    """
    Evaluate alignment quality on test words (not used for training).
    
    Args:
        aligned_embeddings: Aligned source embeddings
        target_embeddings: Target embeddings
        test_words: Specific words to test (if None, samples randomly)
        n_test: Number of words to test
        
    Returns:
        Dict with evaluation metrics
    """
    # Get test words
    if test_words is None:
        shared = list(set(aligned_embeddings.keys()) & set(target_embeddings.keys()))
        np.random.seed(123)
        test_words = np.random.choice(shared, min(n_test, len(shared)), replace=False)
    
    # Filter to valid words
    test_words = [w for w in test_words 
                  if w in aligned_embeddings and w in target_embeddings]
    
    # Compute metrics
    cosines = []
    l2_distances = []
    
    for word in test_words:
        v_aligned = aligned_embeddings[word]
        v_target = target_embeddings[word]
        
        # Handle dimension mismatch
        min_dim = min(len(v_aligned), len(v_target))
        v_aligned = v_aligned[:min_dim]
        v_target = v_target[:min_dim]
        
        # Normalize
        v_aligned = v_aligned / (np.linalg.norm(v_aligned) + 1e-10)
        v_target = v_target / (np.linalg.norm(v_target) + 1e-10)
        
        cos = np.dot(v_aligned, v_target)
        l2 = np.linalg.norm(v_aligned - v_target)
        
        cosines.append(cos)
        l2_distances.append(l2)
    
    cosines = np.array(cosines)
    l2_distances = np.array(l2_distances)
    
    return {
        'n_test': len(test_words),
        'mean_cosine': cosines.mean(),
        'std_cosine': cosines.std(),
        'median_cosine': np.median(cosines),
        'min_cosine': cosines.min(),
        'max_cosine': cosines.max(),
        'mean_l2': l2_distances.mean(),
        'pct_high_sim': np.mean(cosines > 0.5) * 100,
        'pct_very_high_sim': np.mean(cosines > 0.8) * 100,
    }


# =============================================================================
# INTEGRATED ALIGNMENT WORKFLOW
# =============================================================================

def align_embedding_spaces(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    source_name: str = "source",
    target_name: str = "target",
    anchor_words: List[str] = None,
    discover_anchors: bool = True,
    n_discovery_samples: int = 500,
    sim_threshold: float = 0.90,
    drift_threshold: float = 0.10,
    verbose: bool = True
) -> Tuple[ProcrustesAlignment, Dict[str, np.ndarray]]:
    """
    Full workflow to align two embedding spaces using Procrustes analysis.
    
    Args:
        se1: Source SenseExplorer
        se2: Target SenseExplorer
        source_name: Name of source embedding
        target_name: Name of target embedding
        anchor_words: Pre-specified anchor words (if None, discovers them)
        discover_anchors: Whether to discover anchors if not provided
        n_discovery_samples: Samples for anchor discovery
        sim_threshold: Similarity threshold for invariant detection
        drift_threshold: Drift threshold for invariant detection
        verbose: Print progress
        
    Returns:
        Tuple of (alignment, aligned_embeddings)
    """
    if verbose:
        print("=" * 60)
        print(f"ALIGNING: {source_name} → {target_name}")
        print("=" * 60)
    
    # Step 1: Get or discover anchors
    if anchor_words is None and discover_anchors:
        if verbose:
            print("\nStep 1: Discovering invariant anchors...")
        anchor_words = discover_invariants(
            se1, se2,
            n_samples=n_discovery_samples,
            sim_threshold=sim_threshold,
            drift_threshold=drift_threshold,
            verbose=verbose
        )
    elif anchor_words is None:
        raise ValueError("Either provide anchor_words or set discover_anchors=True")
    
    if len(anchor_words) < 10:
        warnings.warn(f"Only {len(anchor_words)} anchors found. Alignment may be unstable.")
    
    # Step 2: Compute Procrustes alignment
    if verbose:
        print(f"\nStep 2: Computing Procrustes alignment...")
    
    alignment = compute_procrustes_alignment(
        se1.embeddings,
        se2.embeddings,
        anchor_words,
        source_name=source_name,
        target_name=target_name,
        verbose=verbose
    )
    
    # Step 3: Transform all source embeddings
    if verbose:
        print(f"\nStep 3: Transforming {len(se1.embeddings)} embeddings...")
    
    aligned_embeddings = align_embeddings(se1.embeddings, alignment)
    
    # Step 4: Evaluate on held-out words
    if verbose:
        print(f"\nStep 4: Evaluating alignment...")
    
    # Use non-anchor words for evaluation
    test_words = [w for w in se1.vocab if w in se2.vocab and w not in anchor_words]
    eval_results = evaluate_alignment(aligned_embeddings, se2.embeddings, test_words[:1000])
    
    if verbose:
        print(f"  Test words: {eval_results['n_test']}")
        print(f"  Mean cosine: {eval_results['mean_cosine']:.4f}")
        print(f"  % with cosine > 0.5: {eval_results['pct_high_sim']:.1f}%")
        print(f"  % with cosine > 0.8: {eval_results['pct_very_high_sim']:.1f}%")
    
    return alignment, aligned_embeddings


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Procrustes alignment of embedding spaces")
    parser.add_argument("--emb1", required=True, help="Path to source embedding")
    parser.add_argument("--name1", default="source", help="Name of source embedding")
    parser.add_argument("--emb2", required=True, help="Path to target embedding")
    parser.add_argument("--name2", default="target", help="Name of target embedding")
    parser.add_argument("--max-words", type=int, default=50000, help="Max words to load")
    parser.add_argument("--n-samples", type=int, default=500, help="Samples for anchor discovery")
    parser.add_argument("--sim-threshold", type=float, default=0.90, help="Similarity threshold")
    parser.add_argument("--drift-threshold", type=float, default=0.10, help="Drift threshold")
    parser.add_argument("--output", default=None, help="Output file for aligned embeddings")
    
    args = parser.parse_args()
    
    from sense_explorer.core import SenseExplorer
    
    print("Loading embeddings...")
    se1 = SenseExplorer.from_file(args.emb1, max_words=args.max_words, verbose=True)
    se2 = SenseExplorer.from_file(args.emb2, max_words=args.max_words, 
                                   target_dim=se1.dim, verbose=True)
    
    alignment, aligned = align_embedding_spaces(
        se1, se2,
        source_name=args.name1,
        target_name=args.name2,
        n_discovery_samples=args.n_samples,
        sim_threshold=args.sim_threshold,
        drift_threshold=args.drift_threshold,
        verbose=True
    )
    
    print("\n" + alignment.report())
    
    if args.output:
        print(f"\nSaving aligned embeddings to {args.output}...")
        np.savez(args.output, 
                 words=list(aligned.keys()),
                 vectors=np.array(list(aligned.values())),
                 rotation=alignment.rotation_matrix,
                 scale=alignment.scale_factor)
        print("Done!")
