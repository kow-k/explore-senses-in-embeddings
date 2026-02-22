"""
Analyze numerical relationships between invariant vectors across embeddings.

Check for:
1. Scaling/amplification factors
2. Rotation patterns
3. Dimension-wise correlations
4. Norm ratios
"""

import sys
import argparse
import numpy as np
from typing import List, Dict, Tuple


def analyze_invariant_relationships(
    emb1_path: str,
    emb2_path: str,
    emb1_name: str = "source1",
    emb2_name: str = "source2",
    invariants: List[str] = None,
    max_words: int = 50000
):
    """Analyze numerical relationships between invariant vectors."""
    
    from sense_explorer.core import SenseExplorer
    
    print("=" * 70)
    print(f"INVARIANT VECTOR ANALYSIS: {emb1_name} vs {emb2_name}")
    print("=" * 70)
    
    # Load embeddings
    print(f"\nLoading {emb1_name}...")
    se1 = SenseExplorer.from_file(emb1_path, max_words=max_words, verbose=False)
    
    print(f"Loading {emb2_name}...")
    se2 = SenseExplorer.from_file(emb2_path, max_words=max_words, target_dim=se1.dim, verbose=False)
    
    print(f"Dimensions: {emb1_name}={se1.dim}d, {emb2_name}={se2.dim}d")
    
    # Default invariants if not provided
    if invariants is None:
        invariants = [
            'mine', 'pen', 'pot', 'pride', 'ride', 'safe', 'shoot', 'smoke',
            'surprise', 'tank', 'weapon', 'welcome', 'worth', 'wrap', 'whale',
            'grab', 'academy', 'bolt', 'pan', 'segment', 'chunk', 'comb',
            'brakes', 'sixth', 'polish', 'ape', 'boats', 'courtesy', 'excess',
            'turtle', 'archives', 'general', 'freedom', 'bouquet'
        ]
    
    # Filter to words in both vocabularies
    valid_invariants = [w for w in invariants if w in se1.vocab and w in se2.vocab]
    print(f"\nValid invariants: {len(valid_invariants)}/{len(invariants)}")
    
    if len(valid_invariants) < 5:
        print("Not enough invariants to analyze!")
        return
    
    # Get vectors
    vecs1 = np.array([se1.embeddings[w] for w in valid_invariants])
    vecs2 = np.array([se2.embeddings[w] for w in valid_invariants])
    
    # Ensure same dimensions
    min_dim = min(vecs1.shape[1], vecs2.shape[1])
    vecs1 = vecs1[:, :min_dim]
    vecs2 = vecs2[:, :min_dim]
    
    print(f"Vector dimensions: {vecs1.shape[1]}")
    
    # =====================================================================
    # 1. NORM ANALYSIS
    # =====================================================================
    print("\n" + "=" * 50)
    print("1. NORM ANALYSIS")
    print("=" * 50)
    
    norms1 = np.linalg.norm(vecs1, axis=1)
    norms2 = np.linalg.norm(vecs2, axis=1)
    
    print(f"\n{emb1_name} norms: mean={norms1.mean():.4f}, std={norms1.std():.4f}")
    print(f"{emb2_name} norms: mean={norms2.mean():.4f}, std={norms2.std():.4f}")
    
    # Norm ratios
    ratios = norms2 / (norms1 + 1e-10)
    print(f"\nNorm ratio ({emb2_name}/{emb1_name}):")
    print(f"  Mean: {ratios.mean():.4f}")
    print(f"  Std:  {ratios.std():.4f}")
    print(f"  Min:  {ratios.min():.4f}")
    print(f"  Max:  {ratios.max():.4f}")
    
    if ratios.std() / ratios.mean() < 0.1:
        print(f"\n  *** UNIFORM SCALING DETECTED: ~{ratios.mean():.3f}x ***")
    
    # =====================================================================
    # 2. COSINE SIMILARITY (should be ~1.0 for invariants)
    # =====================================================================
    print("\n" + "=" * 50)
    print("2. COSINE SIMILARITY")
    print("=" * 50)
    
    vecs1_norm = vecs1 / (norms1[:, np.newaxis] + 1e-10)
    vecs2_norm = vecs2 / (norms2[:, np.newaxis] + 1e-10)
    
    cosines = np.sum(vecs1_norm * vecs2_norm, axis=1)
    
    print(f"\nCosine similarities:")
    print(f"  Mean: {cosines.mean():.4f}")
    print(f"  Std:  {cosines.std():.4f}")
    print(f"  Min:  {cosines.min():.4f}")
    print(f"  Max:  {cosines.max():.4f}")
    
    # =====================================================================
    # 3. DIMENSION-WISE SCALING
    # =====================================================================
    print("\n" + "=" * 50)
    print("3. DIMENSION-WISE SCALING")
    print("=" * 50)
    
    # For each dimension, compute the ratio across all invariants
    dim_ratios = vecs2 / (vecs1 + 1e-10)
    
    # Mean ratio per dimension
    dim_mean_ratios = np.mean(dim_ratios, axis=0)
    dim_std_ratios = np.std(dim_ratios, axis=0)
    
    print(f"\nPer-dimension ratio statistics:")
    print(f"  Mean of means: {dim_mean_ratios.mean():.4f}")
    print(f"  Std of means:  {dim_mean_ratios.std():.4f}")
    print(f"  Mean of stds:  {dim_std_ratios.mean():.4f}")
    
    # Check if ratios are consistent per dimension
    consistent_dims = np.sum(dim_std_ratios < 0.5)
    print(f"\nDimensions with consistent ratio (std < 0.5): {consistent_dims}/{min_dim}")
    
    # =====================================================================
    # 4. LINEAR TRANSFORMATION ANALYSIS
    # =====================================================================
    print("\n" + "=" * 50)
    print("4. LINEAR TRANSFORMATION ANALYSIS")
    print("=" * 50)
    
    # Check if vecs2 ≈ A @ vecs1 for some matrix A
    # Using least squares: A = vecs2.T @ pinv(vecs1.T)
    
    # Actually, let's check simpler relationships first:
    
    # 4a. Pure scaling: vecs2 ≈ c * vecs1
    scale_factors = []
    for i in range(len(valid_invariants)):
        # Optimal scale: c = (v1 · v2) / (v1 · v1)
        c = np.dot(vecs1[i], vecs2[i]) / (np.dot(vecs1[i], vecs1[i]) + 1e-10)
        scale_factors.append(c)
    
    scale_factors = np.array(scale_factors)
    print(f"\n4a. Pure scaling (v2 ≈ c * v1):")
    print(f"  Scale factors: mean={scale_factors.mean():.4f}, std={scale_factors.std():.4f}")
    
    # Reconstruction error with pure scaling
    scaled_vecs1 = vecs1 * scale_factors[:, np.newaxis]
    scale_errors = np.linalg.norm(vecs2 - scaled_vecs1, axis=1)
    rel_scale_errors = scale_errors / (norms2 + 1e-10)
    print(f"  Relative reconstruction error: mean={rel_scale_errors.mean():.4f}")
    
    # 4b. Affine: vecs2 ≈ c * vecs1 + b
    print(f"\n4b. Affine transformation (v2 ≈ c * v1 + b):")
    mean1 = vecs1.mean(axis=0)
    mean2 = vecs2.mean(axis=0)
    centered1 = vecs1 - mean1
    centered2 = vecs2 - mean2
    
    # Global scale on centered data
    global_scale = np.sum(centered1 * centered2) / (np.sum(centered1 * centered1) + 1e-10)
    print(f"  Global scale: {global_scale:.4f}")
    print(f"  Bias norm: {np.linalg.norm(mean2 - global_scale * mean1):.4f}")
    
    # 4c. Orthogonal Procrustes: find rotation R such that vecs2 ≈ vecs1 @ R
    print(f"\n4c. Orthogonal Procrustes (v2 ≈ v1 @ R):")
    
    # SVD of vecs1.T @ vecs2
    U, S, Vt = np.linalg.svd(vecs1_norm.T @ vecs2_norm)
    R = U @ Vt
    
    # Reconstruction
    rotated_vecs1 = vecs1_norm @ R
    procrustes_errors = np.linalg.norm(vecs2_norm - rotated_vecs1, axis=1)
    print(f"  Reconstruction error: mean={procrustes_errors.mean():.4f}, max={procrustes_errors.max():.4f}")
    
    # Check if R is nearly identity
    identity_dist = np.linalg.norm(R - np.eye(min_dim), 'fro')
    print(f"  ||R - I||_F = {identity_dist:.4f}")
    
    if identity_dist < 1.0:
        print("  *** NEAR-IDENTITY ROTATION ***")
    
    # =====================================================================
    # 5. COMPONENT CORRELATION
    # =====================================================================
    print("\n" + "=" * 50)
    print("5. COMPONENT-WISE CORRELATION")
    print("=" * 50)
    
    # Correlation between corresponding dimensions
    dim_correlations = []
    for d in range(min_dim):
        corr = np.corrcoef(vecs1[:, d], vecs2[:, d])[0, 1]
        dim_correlations.append(corr)
    
    dim_correlations = np.array(dim_correlations)
    valid_corrs = dim_correlations[~np.isnan(dim_correlations)]
    
    print(f"\nPer-dimension correlations:")
    print(f"  Mean: {valid_corrs.mean():.4f}")
    print(f"  Std:  {valid_corrs.std():.4f}")
    print(f"  Highly correlated (>0.8): {np.sum(valid_corrs > 0.8)}/{len(valid_corrs)}")
    print(f"  Anti-correlated (<-0.8): {np.sum(valid_corrs < -0.8)}/{len(valid_corrs)}")
    
    # =====================================================================
    # 6. SUMMARY
    # =====================================================================
    print("\n" + "=" * 50)
    print("6. SUMMARY")
    print("=" * 50)
    
    print(f"""
Invariant vectors between {emb1_name} and {emb2_name}:

1. Norm ratio: {ratios.mean():.3f} ± {ratios.std():.3f}
2. Cosine similarity: {cosines.mean():.3f} ± {cosines.std():.3f}
3. Pure scaling error: {rel_scale_errors.mean():.3f}
4. Procrustes error: {procrustes_errors.mean():.3f}
5. Dimension correlations: {valid_corrs.mean():.3f}
""")
    
    if ratios.std() / ratios.mean() < 0.1:
        print(f"FINDING: Uniform scaling by factor ~{ratios.mean():.3f}")
    
    if procrustes_errors.mean() < 0.1:
        print("FINDING: Low Procrustes error - good alignment possible!")
    
    if identity_dist < 1.0:
        print("FINDING: Near-identity rotation - embeddings already aligned!")
    
    return {
        'invariants': valid_invariants,
        'norm_ratios': ratios,
        'cosines': cosines,
        'scale_factors': scale_factors,
        'procrustes_R': R,
        'procrustes_error': procrustes_errors.mean()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze invariant vector relationships")
    parser.add_argument("--emb1", required=True, help="Path to first embedding")
    parser.add_argument("--name1", default="source1", help="Name for first embedding")
    parser.add_argument("--emb2", required=True, help="Path to second embedding")
    parser.add_argument("--name2", default="source2", help="Name for second embedding")
    parser.add_argument("--max-words", type=int, default=50000, help="Max words to load")
    
    args = parser.parse_args()
    
    results = analyze_invariant_relationships(
        args.emb1, args.emb2,
        emb1_name=args.name1,
        emb2_name=args.name2,
        max_words=args.max_words
    )
