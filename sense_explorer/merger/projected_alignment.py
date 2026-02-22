"""
Projected-Space Procrustes Alignment

Instead of aligning raw embedding vectors (which live in different spaces),
first project both embeddings onto a shared vocabulary basis, then perform
Procrustes alignment in that projected space.

Theory:
    Raw vectors: vec_A(w) and vec_B(w) are in DIFFERENT spaces
    
    Projection to shared basis B = {b1, b2, ..., bn}:
        proj_A(w) = [cos(vec_A(w), vec_A(b1)), cos(vec_A(w), vec_A(b2)), ...]
        proj_B(w) = [cos(vec_B(w), vec_B(b1)), cos(vec_B(w), vec_B(b2)), ...]
    
    Now proj_A(w) and proj_B(w) are in the SAME space (R^n)!
    
    For invariant words: proj_A(w) ≈ proj_B(w)  (this is what we measured!)
    
    Procrustes in projected space:
        R = argmin ||proj_B - proj_A @ R||²
        
    This R should be SMALL (near identity) since invariants already align.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import warnings


@dataclass
class ProjectedAlignment:
    """Result of alignment in projected space."""
    
    # Projection basis
    basis_words: List[str]
    basis_dim: int
    
    # Transformation in projected space
    rotation_matrix: np.ndarray  # R: (basis_dim, basis_dim)
    scale_factor: float
    
    # Alignment quality
    anchor_cosines_pre: np.ndarray   # Before alignment
    anchor_cosines_post: np.ndarray  # After alignment
    alignment_error: float
    
    # Anchor info
    anchor_words: List[str]
    n_anchors: int
    
    # Source/target info
    source_name: str
    target_name: str
    
    def report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            f"PROJECTED-SPACE ALIGNMENT: {self.source_name} → {self.target_name}",
            "=" * 60,
            f"Basis dimension: {self.basis_dim} words",
            f"Anchors used: {self.n_anchors}",
            f"Scale factor: {self.scale_factor:.4f}",
            f"",
            f"Alignment quality:",
            f"  Mean cosine BEFORE: {self.anchor_cosines_pre.mean():.4f}",
            f"  Mean cosine AFTER:  {self.anchor_cosines_post.mean():.4f}",
            f"  Improvement: {self.anchor_cosines_post.mean() - self.anchor_cosines_pre.mean():.4f}",
            f"  Min cosine (post): {self.anchor_cosines_post.min():.4f}",
            f"",
            f"Rotation matrix:",
            f"  ||R - I||_F: {np.linalg.norm(self.rotation_matrix - np.eye(self.basis_dim)):.4f}",
        ]
        return "\n".join(lines)


def build_projection_basis(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    max_basis_size: int = 1000,
    min_frequency_rank: int = 100,
    verbose: bool = True
) -> List[str]:
    """
    Build a shared vocabulary basis for projection.
    
    Selects words that:
    1. Exist in both vocabularies
    2. Are common (high frequency rank)
    3. Are diverse (not too similar to each other)
    
    Args:
        se1, se2: SenseExplorer instances
        max_basis_size: Maximum basis dimension
        min_frequency_rank: Skip top N words (often function words)
        verbose: Print progress
        
    Returns:
        List of basis words
    """
    # Get shared vocabulary
    shared = list(set(se1.vocab) & set(se2.vocab))
    
    if verbose:
        print(f"Shared vocabulary: {len(shared)} words")
    
    # Sort by position in vocabulary (proxy for frequency)
    # Words earlier in vocab files are typically more frequent
    word_ranks = {}
    for i, w in enumerate(se1.vocab):
        if w in shared:
            word_ranks[w] = i
    
    # Sort by rank, skip very top (function words) and filter
    ranked_words = sorted(shared, key=lambda w: word_ranks.get(w, float('inf')))
    
    # Filter: skip top N, require alphabetic
    filtered = [w for w in ranked_words[min_frequency_rank:] 
                if w.isalpha() and len(w) >= 3]
    
    # Take top words up to max_basis_size
    basis = filtered[:max_basis_size]
    
    if verbose:
        print(f"Basis size: {len(basis)} words")
        print(f"Sample basis words: {basis[:10]}")
    
    return basis


def project_to_basis(
    embeddings: Dict[str, np.ndarray],
    basis_words: List[str],
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Project embeddings onto the vocabulary basis.
    
    For each word w, compute:
        proj(w) = [cos(w, b1), cos(w, b2), ..., cos(w, bn)]
    
    Args:
        embeddings: Dict mapping words to vectors
        basis_words: List of basis words
        normalize: Whether to L2-normalize input vectors
        
    Returns:
        Dict mapping words to projected vectors (dimension = len(basis_words))
    """
    # Get basis vectors
    basis_vecs = []
    valid_basis = []
    for b in basis_words:
        if b in embeddings:
            vec = embeddings[b]
            if normalize:
                norm = np.linalg.norm(vec)
                vec = vec / norm if norm > 1e-10 else vec
            basis_vecs.append(vec)
            valid_basis.append(b)
    
    if len(basis_vecs) == 0:
        raise ValueError("No basis words found in embeddings")
    
    basis_matrix = np.array(basis_vecs)  # (n_basis, dim)
    
    # Project each word
    projected = {}
    for word, vec in embeddings.items():
        if normalize:
            norm = np.linalg.norm(vec)
            vec_norm = vec / norm if norm > 1e-10 else vec
        else:
            vec_norm = vec
        
        # Handle dimension mismatch
        min_dim = min(len(vec_norm), basis_matrix.shape[1])
        
        # Compute cosines with all basis words
        proj = basis_matrix[:, :min_dim] @ vec_norm[:min_dim]
        
        # Normalize projection
        proj_norm = np.linalg.norm(proj)
        if proj_norm > 1e-10:
            proj = proj / proj_norm
        
        projected[word] = proj
    
    return projected


def compute_projected_alignment(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    anchor_words: List[str],
    basis_words: List[str] = None,
    max_basis_size: int = 500,
    source_name: str = "source",
    target_name: str = "target",
    verbose: bool = True
) -> ProjectedAlignment:
    """
    Compute Procrustes alignment in projected space.
    
    Args:
        se1: Source SenseExplorer
        se2: Target SenseExplorer
        anchor_words: Invariant words for alignment
        basis_words: Projection basis (computed if None)
        max_basis_size: Max basis dimension
        source_name: Name of source
        target_name: Name of target
        verbose: Print progress
        
    Returns:
        ProjectedAlignment object
    """
    if verbose:
        print("=" * 60)
        print(f"PROJECTED-SPACE ALIGNMENT: {source_name} → {target_name}")
        print("=" * 60)
    
    # Step 1: Build or validate basis
    if basis_words is None:
        if verbose:
            print("\nStep 1: Building projection basis...")
        basis_words = build_projection_basis(se1, se2, max_basis_size, verbose=verbose)
    
    # Step 2: Project both embeddings to basis space
    if verbose:
        print(f"\nStep 2: Projecting embeddings to {len(basis_words)}-dimensional basis...")
    
    proj1 = project_to_basis(se1.embeddings, basis_words)
    proj2 = project_to_basis(se2.embeddings, basis_words)
    
    if verbose:
        print(f"  Projected {len(proj1)} words from {source_name}")
        print(f"  Projected {len(proj2)} words from {target_name}")
    
    # Step 3: Get anchor vectors in projected space
    valid_anchors = [w for w in anchor_words if w in proj1 and w in proj2]
    
    if len(valid_anchors) < 3:
        raise ValueError(f"Need at least 3 valid anchors, got {len(valid_anchors)}")
    
    if verbose:
        print(f"\nStep 3: Using {len(valid_anchors)} anchors for alignment...")
    
    X = np.array([proj1[w] for w in valid_anchors])  # (n_anchors, basis_dim)
    Y = np.array([proj2[w] for w in valid_anchors])
    
    # Compute cosines BEFORE alignment
    cosines_pre = np.sum(X * Y, axis=1)
    
    if verbose:
        print(f"  Mean cosine BEFORE alignment: {cosines_pre.mean():.4f}")
    
    # Step 4: Compute Procrustes rotation
    if verbose:
        print(f"\nStep 4: Computing Procrustes rotation...")
    
    # SVD of X.T @ Y
    M = X.T @ Y
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    # Compute scale
    X_rotated = X @ R
    scale = np.sum(Y * X_rotated) / (np.sum(X_rotated * X_rotated) + 1e-10)
    
    # Apply transformation
    X_aligned = X @ R * scale
    
    # Normalize for cosine
    X_aligned_norm = X_aligned / (np.linalg.norm(X_aligned, axis=1, keepdims=True) + 1e-10)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosines AFTER alignment
    cosines_post = np.sum(X_aligned_norm * Y_norm, axis=1)
    
    # Compute error
    errors = np.linalg.norm(Y_norm - X_aligned_norm, axis=1)
    
    if verbose:
        print(f"  Mean cosine AFTER alignment: {cosines_post.mean():.4f}")
        print(f"  Improvement: {cosines_post.mean() - cosines_pre.mean():.4f}")
        print(f"  ||R - I||_F: {np.linalg.norm(R - np.eye(len(basis_words))):.4f}")
    
    return ProjectedAlignment(
        basis_words=basis_words,
        basis_dim=len(basis_words),
        rotation_matrix=R,
        scale_factor=scale,
        anchor_cosines_pre=cosines_pre,
        anchor_cosines_post=cosines_post,
        alignment_error=errors.mean(),
        anchor_words=valid_anchors,
        n_anchors=len(valid_anchors),
        source_name=source_name,
        target_name=target_name
    )


def align_projected_embeddings(
    source_projected: Dict[str, np.ndarray],
    alignment: ProjectedAlignment
) -> Dict[str, np.ndarray]:
    """
    Apply alignment transformation to projected embeddings.
    
    Args:
        source_projected: Projected source embeddings
        alignment: ProjectedAlignment object
        
    Returns:
        Aligned projected embeddings
    """
    R = alignment.rotation_matrix
    scale = alignment.scale_factor
    
    aligned = {}
    for word, vec in source_projected.items():
        aligned_vec = vec @ R * scale
        # Normalize
        norm = np.linalg.norm(aligned_vec)
        if norm > 1e-10:
            aligned_vec = aligned_vec / norm
        aligned[word] = aligned_vec
    
    return aligned


def evaluate_projected_alignment(
    source_projected: Dict[str, np.ndarray],
    target_projected: Dict[str, np.ndarray],
    alignment: ProjectedAlignment,
    test_words: List[str] = None,
    n_test: int = 1000
) -> Dict:
    """
    Evaluate alignment on test words.
    
    Args:
        source_projected: Projected source embeddings
        target_projected: Projected target embeddings
        alignment: ProjectedAlignment object
        test_words: Words to test (samples if None)
        n_test: Number of test words
        
    Returns:
        Dict with evaluation metrics
    """
    R = alignment.rotation_matrix
    scale = alignment.scale_factor
    anchor_set = set(alignment.anchor_words)
    
    # Get test words (excluding anchors)
    if test_words is None:
        shared = [w for w in source_projected.keys() 
                  if w in target_projected and w not in anchor_set]
        np.random.seed(123)
        test_words = list(np.random.choice(shared, min(n_test, len(shared)), replace=False))
    
    # Compute metrics
    cosines_pre = []
    cosines_post = []
    
    for word in test_words:
        if word not in source_projected or word not in target_projected:
            continue
        
        x = source_projected[word]
        y = target_projected[word]
        
        # Normalize
        x_norm = x / (np.linalg.norm(x) + 1e-10)
        y_norm = y / (np.linalg.norm(y) + 1e-10)
        
        # Before alignment
        cos_pre = np.dot(x_norm, y_norm)
        cosines_pre.append(cos_pre)
        
        # After alignment
        x_aligned = x @ R * scale
        x_aligned_norm = x_aligned / (np.linalg.norm(x_aligned) + 1e-10)
        cos_post = np.dot(x_aligned_norm, y_norm)
        cosines_post.append(cos_post)
    
    cosines_pre = np.array(cosines_pre)
    cosines_post = np.array(cosines_post)
    
    return {
        'n_test': len(cosines_pre),
        'mean_cosine_pre': cosines_pre.mean(),
        'mean_cosine_post': cosines_post.mean(),
        'improvement': cosines_post.mean() - cosines_pre.mean(),
        'std_cosine_post': cosines_post.std(),
        'min_cosine_post': cosines_post.min(),
        'max_cosine_post': cosines_post.max(),
        'pct_improved': np.mean(cosines_post > cosines_pre) * 100,
        'pct_high_sim': np.mean(cosines_post > 0.5) * 100,
        'pct_very_high_sim': np.mean(cosines_post > 0.8) * 100,
    }


# =============================================================================
# FULL WORKFLOW
# =============================================================================

def align_in_projected_space(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    source_name: str = "source",
    target_name: str = "target",
    anchor_words: List[str] = None,
    max_basis_size: int = 500,
    discover_anchors: bool = True,
    n_discovery_samples: int = 500,
    sim_threshold: float = 0.90,
    drift_threshold: float = 0.10,
    verbose: bool = True
) -> Tuple[ProjectedAlignment, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict]:
    """
    Full workflow for projected-space alignment.
    
    Args:
        se1, se2: SenseExplorer instances
        source_name, target_name: Names for reporting
        anchor_words: Pre-specified anchors (discovers if None)
        max_basis_size: Projection basis dimension
        discover_anchors: Whether to discover anchors
        n_discovery_samples: Samples for discovery
        sim_threshold, drift_threshold: Thresholds for invariant detection
        verbose: Print progress
        
    Returns:
        Tuple of (alignment, source_projected, target_projected, eval_results)
    """
    # Step 1: Discover anchors if needed
    if anchor_words is None and discover_anchors:
        if verbose:
            print("\n" + "=" * 60)
            print("DISCOVERING INVARIANT ANCHORS")
            print("=" * 60)
        
        # Import discover_invariants - handle both module and script contexts
        try:
            from .procrustes_alignment import discover_invariants
        except ImportError:
            from procrustes_alignment import discover_invariants
        
        anchor_words = discover_invariants(
            se1, se2,
            n_samples=n_discovery_samples,
            sim_threshold=sim_threshold,
            drift_threshold=drift_threshold,
            verbose=verbose
        )
    
    if anchor_words is None or len(anchor_words) < 3:
        raise ValueError("Need anchor words for alignment")
    
    # Step 2: Build basis and project
    basis_words = build_projection_basis(se1, se2, max_basis_size, verbose=verbose)
    
    if verbose:
        print(f"\nProjecting embeddings...")
    
    proj1 = project_to_basis(se1.embeddings, basis_words)
    proj2 = project_to_basis(se2.embeddings, basis_words)
    
    # Step 3: Compute alignment
    alignment = compute_projected_alignment(
        se1, se2, anchor_words, basis_words,
        source_name=source_name, target_name=target_name,
        verbose=verbose
    )
    
    # Step 4: Evaluate on test words
    if verbose:
        print(f"\nStep 5: Evaluating on test words...")
    
    eval_results = evaluate_projected_alignment(
        proj1, proj2, alignment, n_test=1000
    )
    
    if verbose:
        print(f"  Test words: {eval_results['n_test']}")
        print(f"  Mean cosine BEFORE: {eval_results['mean_cosine_pre']:.4f}")
        print(f"  Mean cosine AFTER:  {eval_results['mean_cosine_post']:.4f}")
        print(f"  Improvement: {eval_results['improvement']:.4f}")
        print(f"  % improved: {eval_results['pct_improved']:.1f}%")
        print(f"  % with cosine > 0.5: {eval_results['pct_high_sim']:.1f}%")
        print(f"  % with cosine > 0.8: {eval_results['pct_very_high_sim']:.1f}%")
    
    return alignment, proj1, proj2, eval_results


# =============================================================================
# COMMAND LINE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Projected-space Procrustes alignment")
    parser.add_argument("--emb1", required=True, help="Path to source embedding")
    parser.add_argument("--name1", default="source", help="Name of source")
    parser.add_argument("--emb2", required=True, help="Path to target embedding")
    parser.add_argument("--name2", default="target", help="Name of target")
    parser.add_argument("--max-words", type=int, default=50000)
    parser.add_argument("--basis-size", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--sim-threshold", type=float, default=0.90)
    parser.add_argument("--drift-threshold", type=float, default=0.10)
    
    args = parser.parse_args()
    
    from sense_explorer.core import SenseExplorer
    
    print("Loading embeddings...")
    se1 = SenseExplorer.from_file(args.emb1, max_words=args.max_words, verbose=True)
    se2 = SenseExplorer.from_file(args.emb2, max_words=args.max_words,
                                   target_dim=se1.dim, verbose=True)
    
    alignment, proj1, proj2, eval_results = align_in_projected_space(
        se1, se2,
        source_name=args.name1,
        target_name=args.name2,
        max_basis_size=args.basis_size,
        n_discovery_samples=args.n_samples,
        sim_threshold=args.sim_threshold,
        drift_threshold=args.drift_threshold,
        verbose=True
    )
    
    print("\n" + alignment.report())
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
