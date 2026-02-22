"""
Dual-Population Projected Alignment

Recognizes that words fall into two distinct populations with different
cross-register behaviors, and aligns each population separately.

Population Structure:
    Group 0: Invariants at (sim≈1, drift≈0) - shared by both populations
    Group 1: Main line (drift ≈ 1 - similarity)
    Group 2: Secondary line (drift ≈ 0.3 - similarity, or drift << 1 - sim)

Alignment Strategy:
    Subset A = Group 0 ∪ Group 1  →  Rotation R_A
    Subset B = Group 0 ∪ Group 2  →  Rotation R_B
    
    For new word w:
        1. Classify w as main-line or secondary-line
        2. Apply appropriate rotation (R_A or R_B)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import warnings


@dataclass
class PopulationClassification:
    """Classification of words into populations."""
    
    group0_invariants: List[str]   # (sim≈1, drift≈0)
    group1_main_line: List[str]    # drift ≈ 1 - sim
    group2_secondary_line: List[str]  # drift << 1 - sim
    
    # Thresholds used
    invariant_sim_threshold: float
    invariant_drift_threshold: float
    secondary_residual_threshold: float
    
    def report(self) -> str:
        lines = [
            "=" * 60,
            "POPULATION CLASSIFICATION",
            "=" * 60,
            f"Group 0 (invariants): {len(self.group0_invariants)} words",
            f"Group 1 (main line): {len(self.group1_main_line)} words",
            f"Group 2 (secondary line): {len(self.group2_secondary_line)} words",
            "",
            f"Subset A (Group 0 + 1): {len(self.group0_invariants) + len(self.group1_main_line)} words",
            f"Subset B (Group 0 + 2): {len(self.group0_invariants) + len(self.group2_secondary_line)} words",
        ]
        return "\n".join(lines)


@dataclass
class DualAlignment:
    """Result of dual-population alignment."""
    
    # Population classification
    classification: PopulationClassification
    
    # Alignment for Subset A (main line)
    rotation_A: np.ndarray
    scale_A: float
    anchors_A: List[str]
    cosine_pre_A: float
    cosine_post_A: float
    
    # Alignment for Subset B (secondary line)
    rotation_B: np.ndarray
    scale_B: float
    anchors_B: List[str]
    cosine_pre_B: float
    cosine_post_B: float
    
    # Basis info
    basis_words: List[str]
    basis_dim: int
    
    # Source/target info
    source_name: str
    target_name: str
    
    def classify_word(
        self, 
        word: str, 
        source_projected: Dict[str, np.ndarray],
        target_projected: Dict[str, np.ndarray]
    ) -> str:
        """
        Classify a word as 'main' or 'secondary' based on its behavior.
        
        Returns 'main', 'secondary', or 'invariant'
        """
        if word in self.classification.group0_invariants:
            return 'invariant'
        elif word in self.classification.group2_secondary_line:
            return 'secondary'
        elif word in self.classification.group1_main_line:
            return 'main'
        else:
            # Unknown word - classify by similarity/drift pattern
            if word in source_projected and word in target_projected:
                x = source_projected[word]
                y = target_projected[word]
                x_norm = x / (np.linalg.norm(x) + 1e-10)
                y_norm = y / (np.linalg.norm(y) + 1e-10)
                sim = np.dot(x_norm, y_norm)
                # Estimate drift (we don't have it directly, use 1-sim as proxy)
                drift_proxy = 1 - sim
                residual = drift_proxy + sim - 1  # Should be ~0 for main line
                
                if residual < -0.05:
                    return 'secondary'
                else:
                    return 'main'
            return 'main'  # Default to main line
    
    def transform(
        self, 
        word: str,
        source_projected: Dict[str, np.ndarray],
        target_projected: Dict[str, np.ndarray] = None,
        population: str = None
    ) -> np.ndarray:
        """
        Transform a word using the appropriate rotation.
        
        Args:
            word: Word to transform
            source_projected: Projected source embeddings
            target_projected: Projected target embeddings (for classification)
            population: Override classification ('main', 'secondary', or None for auto)
            
        Returns:
            Transformed vector
        """
        if word not in source_projected:
            raise KeyError(f"Word '{word}' not in source embeddings")
        
        vec = source_projected[word]
        
        # Classify if not specified
        if population is None:
            if target_projected is not None:
                population = self.classify_word(word, source_projected, target_projected)
            else:
                population = 'main'  # Default
        
        # Apply appropriate transformation
        if population == 'secondary':
            R, scale = self.rotation_B, self.scale_B
        else:  # 'main' or 'invariant'
            R, scale = self.rotation_A, self.scale_A
        
        aligned = vec @ R * scale
        aligned = aligned / (np.linalg.norm(aligned) + 1e-10)
        
        return aligned
    
    def report(self) -> str:
        lines = [
            "=" * 60,
            f"DUAL-POPULATION ALIGNMENT: {self.source_name} → {self.target_name}",
            "=" * 60,
            "",
            self.classification.report(),
            "",
            "-" * 40,
            "SUBSET A ALIGNMENT (Main Line)",
            "-" * 40,
            f"Anchors: {len(self.anchors_A)}",
            f"Scale: {self.scale_A:.4f}",
            f"Cosine BEFORE: {self.cosine_pre_A:.4f}",
            f"Cosine AFTER:  {self.cosine_post_A:.4f}",
            f"||R_A - I||: {np.linalg.norm(self.rotation_A - np.eye(self.basis_dim)):.4f}",
            "",
            "-" * 40,
            "SUBSET B ALIGNMENT (Secondary Line)",
            "-" * 40,
            f"Anchors: {len(self.anchors_B)}",
            f"Scale: {self.scale_B:.4f}",
            f"Cosine BEFORE: {self.cosine_pre_B:.4f}",
            f"Cosine AFTER:  {self.cosine_post_B:.4f}",
            f"||R_B - I||: {np.linalg.norm(self.rotation_B - np.eye(self.basis_dim)):.4f}",
        ]
        return "\n".join(lines)


def classify_populations(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    n_samples: int = 1000,
    invariant_sim: float = 0.95,
    invariant_drift: float = 0.05,
    secondary_residual: float = -0.05,
    verbose: bool = True
) -> Tuple[PopulationClassification, Dict[str, Tuple[float, float, float]]]:
    """
    Classify words into three groups based on register behavior.
    
    Args:
        se1, se2: SenseExplorer instances
        n_samples: Number of words to sample
        invariant_sim: Similarity threshold for Group 0
        invariant_drift: Drift threshold for Group 0
        secondary_residual: Residual threshold for Group 2 (negative)
        verbose: Print progress
        
    Returns:
        Tuple of (PopulationClassification, word_stats dict)
    """
    try:
        from .register_profiles import create_register_profile
    except ImportError:
        from register_profiles import create_register_profile
    
    # Get shared vocabulary
    shared = list(set(se1.vocab) & set(se2.vocab))
    np.random.seed(42)
    sample_words = list(np.random.choice(shared, min(n_samples, len(shared)), replace=False))
    
    if verbose:
        print(f"Classifying {len(sample_words)} words into populations...")
    
    explorers = {"source": se1, "target": se2}
    
    word_stats = {}  # word -> (similarity, drift, residual)
    
    for i, word in enumerate(sample_words):
        if verbose and (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(sample_words)}...")
        
        try:
            profile = create_register_profile(word, explorers, n_senses=2, verbose=False)
            sim = profile.overall_register_similarity
            drifts = [sd.mean_drift for sd in profile.sense_drifts.values()]
            mean_drift = np.mean(drifts) if drifts else 0.0
            residual = mean_drift + sim - 1  # residual from main line
            
            word_stats[word] = (sim, mean_drift, residual)
        except Exception:
            continue
    
    # Classify into groups
    group0 = []  # invariants
    group1 = []  # main line
    group2 = []  # secondary line
    
    for word, (sim, drift, residual) in word_stats.items():
        if sim >= invariant_sim and drift <= invariant_drift:
            group0.append(word)
        elif residual < secondary_residual:
            group2.append(word)
        else:
            group1.append(word)
    
    if verbose:
        print(f"\nClassification results:")
        print(f"  Group 0 (invariants): {len(group0)}")
        print(f"  Group 1 (main line): {len(group1)}")
        print(f"  Group 2 (secondary line): {len(group2)}")
    
    classification = PopulationClassification(
        group0_invariants=group0,
        group1_main_line=group1,
        group2_secondary_line=group2,
        invariant_sim_threshold=invariant_sim,
        invariant_drift_threshold=invariant_drift,
        secondary_residual_threshold=secondary_residual
    )
    
    return classification, word_stats


def compute_single_alignment(
    proj1: Dict[str, np.ndarray],
    proj2: Dict[str, np.ndarray],
    anchor_words: List[str],
    basis_dim: int,
    verbose: bool = True,
    label: str = ""
) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    """
    Compute Procrustes alignment for a single population.
    
    Returns:
        Tuple of (R, scale, cosine_pre, cosine_post, per_anchor_cosines_post)
    """
    # Filter valid anchors
    valid = [w for w in anchor_words if w in proj1 and w in proj2]
    
    if len(valid) < 3:
        warnings.warn(f"{label}: Only {len(valid)} valid anchors")
        # Return identity
        return np.eye(basis_dim), 1.0, 0.0, 0.0, np.array([])
    
    X = np.array([proj1[w] for w in valid])
    Y = np.array([proj2[w] for w in valid])
    
    # Normalize
    X_norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    Y_norms = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-10
    X_norm = X / X_norms
    Y_norm = Y / Y_norms
    
    # Cosine before
    cosines_pre = np.sum(X_norm * Y_norm, axis=1)
    
    # Procrustes
    M = X_norm.T @ Y_norm
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    # Scale
    X_rot = X_norm @ R
    scale = np.sum(Y_norm * X_rot) / (np.sum(X_rot * X_rot) + 1e-10)
    
    # Cosine after
    X_aligned = X_norm @ R * scale
    X_aligned_norm = X_aligned / (np.linalg.norm(X_aligned, axis=1, keepdims=True) + 1e-10)
    cosines_post = np.sum(X_aligned_norm * Y_norm, axis=1)
    
    if verbose:
        print(f"  {label}: {len(valid)} anchors, cosine {cosines_pre.mean():.4f} → {cosines_post.mean():.4f}")
    
    return R, scale, cosines_pre.mean(), cosines_post.mean(), cosines_post


def build_projection_basis(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    max_basis_size: int = 500,
    min_frequency_rank: int = 100,
    verbose: bool = True
) -> List[str]:
    """Build shared vocabulary basis."""
    shared = list(set(se1.vocab) & set(se2.vocab))
    
    # Sort by position in vocab (frequency proxy)
    word_ranks = {w: i for i, w in enumerate(se1.vocab) if w in shared}
    ranked = sorted(shared, key=lambda w: word_ranks.get(w, float('inf')))
    
    # Filter
    filtered = [w for w in ranked[min_frequency_rank:] if w.isalpha() and len(w) >= 3]
    basis = filtered[:max_basis_size]
    
    if verbose:
        print(f"Basis: {len(basis)} words")
    
    return basis


def project_to_basis(
    embeddings: Dict[str, np.ndarray],
    basis_words: List[str]
) -> Dict[str, np.ndarray]:
    """Project embeddings onto vocabulary basis."""
    # Get basis vectors
    basis_vecs = []
    for b in basis_words:
        if b in embeddings:
            vec = embeddings[b]
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            basis_vecs.append(vec)
    
    basis_matrix = np.array(basis_vecs)
    
    projected = {}
    for word, vec in embeddings.items():
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        min_dim = min(len(vec_norm), basis_matrix.shape[1])
        proj = basis_matrix[:, :min_dim] @ vec_norm[:min_dim]
        proj = proj / (np.linalg.norm(proj) + 1e-10)
        projected[word] = proj
    
    return projected


def compute_dual_alignment(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    source_name: str = "source",
    target_name: str = "target",
    n_classification_samples: int = 1000,
    max_basis_size: int = 300,
    invariant_sim: float = 0.95,
    invariant_drift: float = 0.05,
    secondary_residual: float = -0.05,
    expand_anchors_A: bool = True,
    expand_anchors_B: bool = True,
    anchor_expansion_sim: float = 0.7,
    anchor_expansion_drift: float = 0.3,
    holdout_ratio: float = 0.2,
    verbose: bool = True
) -> Tuple[DualAlignment, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Tuple[float, float, float]]]:
    """
    Compute dual-population alignment.
    
    Args:
        se1, se2: SenseExplorer instances
        source_name, target_name: Names for reporting
        n_classification_samples: Samples for population classification
        max_basis_size: Projection basis dimension
        invariant_sim, invariant_drift: Thresholds for Group 0
        secondary_residual: Residual threshold for Group 2
        expand_anchors_A, expand_anchors_B: Whether to expand anchor sets
        anchor_expansion_sim, anchor_expansion_drift: Thresholds for expansion
        holdout_ratio: Fraction of each group to hold out for testing
        verbose: Print progress
        
    Returns:
        Tuple of (DualAlignment, proj1, proj2, word_stats)
    """
    if verbose:
        print("=" * 60)
        print(f"DUAL-POPULATION ALIGNMENT: {source_name} → {target_name}")
        print("=" * 60)
    
    # Step 1: Classify populations
    if verbose:
        print("\nStep 1: Classifying populations...")
    
    classification, word_stats = classify_populations(
        se1, se2,
        n_samples=n_classification_samples,
        invariant_sim=invariant_sim,
        invariant_drift=invariant_drift,
        secondary_residual=secondary_residual,
        verbose=verbose
    )
    
    # Step 2: Build basis and project
    if verbose:
        print(f"\nStep 2: Building projection basis...")
    
    basis_words = build_projection_basis(se1, se2, max_basis_size, verbose=verbose)
    
    if verbose:
        print("Projecting embeddings...")
    
    proj1 = project_to_basis(se1.embeddings, basis_words)
    proj2 = project_to_basis(se2.embeddings, basis_words)
    
    # Step 3: Build anchor sets with holdout
    if verbose:
        print(f"\nStep 3: Building anchor sets (holdout={holdout_ratio:.0%})...")
    
    np.random.seed(42)
    
    def split_holdout(words, ratio):
        """Split words into anchors and holdout."""
        n_holdout = max(1, int(len(words) * ratio))
        shuffled = list(words)
        np.random.shuffle(shuffled)
        return shuffled[n_holdout:], shuffled[:n_holdout]
    
    # Split Group 0 (invariants)
    g0_anchors, g0_holdout = split_holdout(classification.group0_invariants, holdout_ratio)
    
    # Split Group 1 (main line) 
    g1_anchors, g1_holdout = split_holdout(classification.group1_main_line, holdout_ratio)
    
    # Split Group 2 (secondary line)
    g2_anchors, g2_holdout = split_holdout(classification.group2_secondary_line, holdout_ratio)
    
    if verbose:
        print(f"  Group 0: {len(g0_anchors)} anchors, {len(g0_holdout)} holdout")
        print(f"  Group 1: {len(g1_anchors)} anchors, {len(g1_holdout)} holdout")
        print(f"  Group 2: {len(g2_anchors)} anchors, {len(g2_holdout)} holdout")
    
    # Subset A anchors: Group 0 + Group 1 (anchors only)
    anchors_A = g0_anchors.copy()
    if expand_anchors_A:
        # Add main line words with good similarity
        for word in g1_anchors:
            if word in word_stats:
                sim, drift, residual = word_stats[word]
                if sim >= anchor_expansion_sim and drift <= anchor_expansion_drift:
                    anchors_A.append(word)
    
    # Subset B anchors: Group 0 + Group 2 (anchors only)
    anchors_B = g0_anchors.copy()
    anchors_B.extend(g2_anchors)
    
    if verbose:
        print(f"  Subset A anchors: {len(anchors_A)} (Group 0 + expanded Group 1)")
        print(f"  Subset B anchors: {len(anchors_B)} (Group 0 + Group 2)")
    
    # Step 4: Compute alignments
    if verbose:
        print(f"\nStep 4: Computing alignments...")
    
    R_A, scale_A, pre_A, post_A, _ = compute_single_alignment(
        proj1, proj2, anchors_A, len(basis_words), verbose, "Subset A"
    )
    
    R_B, scale_B, pre_B, post_B, _ = compute_single_alignment(
        proj1, proj2, anchors_B, len(basis_words), verbose, "Subset B"
    )
    
    # Update classification to track holdout words
    # (Store original lists for evaluation)
    classification.group0_invariants = classification.group0_invariants  # Keep all for eval
    classification.group1_main_line = classification.group1_main_line
    classification.group2_secondary_line = classification.group2_secondary_line
    
    # Create result
    alignment = DualAlignment(
        classification=classification,
        rotation_A=R_A,
        scale_A=scale_A,
        anchors_A=anchors_A,
        cosine_pre_A=pre_A,
        cosine_post_A=post_A,
        rotation_B=R_B,
        scale_B=scale_B,
        anchors_B=anchors_B,
        cosine_pre_B=pre_B,
        cosine_post_B=post_B,
        basis_words=basis_words,
        basis_dim=len(basis_words),
        source_name=source_name,
        target_name=target_name
    )
    
    return alignment, proj1, proj2, word_stats


def evaluate_dual_alignment(
    alignment: DualAlignment,
    proj1: Dict[str, np.ndarray],
    proj2: Dict[str, np.ndarray],
    word_stats: Dict[str, Tuple[float, float, float]] = None,
    n_test: int = 500,
    holdout_ratio: float = 0.2,
    verbose: bool = True
) -> Dict:
    """
    Evaluate dual alignment on test words.
    
    Tests each word using its appropriate transformation.
    Also compares what happens when using the "wrong" rotation.
    """
    # Get test words (excluding anchors)
    anchor_set = set(alignment.anchors_A) | set(alignment.anchors_B)
    test_candidates = [w for w in proj1.keys() if w in proj2 and w not in anchor_set]
    
    # Separate by known classification
    main_test = [w for w in test_candidates if w in alignment.classification.group1_main_line]
    secondary_test = [w for w in test_candidates if w in alignment.classification.group2_secondary_line]
    invariant_test = [w for w in test_candidates if w in alignment.classification.group0_invariants]
    unknown_test = [w for w in test_candidates 
                    if w not in alignment.classification.group1_main_line 
                    and w not in alignment.classification.group2_secondary_line
                    and w not in alignment.classification.group0_invariants]
    
    if verbose:
        print(f"\nTest set breakdown:")
        print(f"  Main line (Group 1): {len(main_test)}")
        print(f"  Secondary line (Group 2): {len(secondary_test)}")
        print(f"  Invariants (Group 0): {len(invariant_test)}")
        print(f"  Unknown (not classified): {len(unknown_test)}")
    
    # Sample from each group
    np.random.seed(789)
    
    def sample_group(words, n):
        if len(words) <= n:
            return words
        return list(np.random.choice(words, n, replace=False))
    
    test_main = sample_group(main_test, min(300, len(main_test)))
    test_secondary = sample_group(secondary_test, len(secondary_test))  # Use all
    test_invariant = sample_group(invariant_test, len(invariant_test))  # Use all
    test_unknown = sample_group(unknown_test, min(200, len(unknown_test)))
    
    results = {
        'main': {'correct_R': [], 'wrong_R': [], 'no_R': []},
        'secondary': {'correct_R': [], 'wrong_R': [], 'no_R': []},
        'invariant': {'correct_R': [], 'wrong_R': [], 'no_R': []},
        'unknown': {'auto_R': [], 'R_A': [], 'R_B': [], 'no_R': []},
    }
    
    R_A, scale_A = alignment.rotation_A, alignment.scale_A
    R_B, scale_B = alignment.rotation_B, alignment.scale_B
    
    def compute_cosines(word, R, scale):
        x = proj1[word]
        y = proj2[word]
        x_norm = x / (np.linalg.norm(x) + 1e-10)
        y_norm = y / (np.linalg.norm(y) + 1e-10)
        
        # Before (no rotation)
        cos_pre = np.dot(x_norm, y_norm)
        
        # After rotation
        x_rot = x @ R * scale
        x_rot_norm = x_rot / (np.linalg.norm(x_rot) + 1e-10)
        cos_post = np.dot(x_rot_norm, y_norm)
        
        return cos_pre, cos_post
    
    # Test main line words
    for word in test_main:
        cos_pre, cos_A = compute_cosines(word, R_A, scale_A)
        _, cos_B = compute_cosines(word, R_B, scale_B)
        results['main']['correct_R'].append(cos_A)  # R_A is correct
        results['main']['wrong_R'].append(cos_B)    # R_B is wrong
        results['main']['no_R'].append(cos_pre)     # No rotation
    
    # Test secondary line words
    for word in test_secondary:
        cos_pre, cos_A = compute_cosines(word, R_A, scale_A)
        _, cos_B = compute_cosines(word, R_B, scale_B)
        results['secondary']['correct_R'].append(cos_B)  # R_B is correct
        results['secondary']['wrong_R'].append(cos_A)    # R_A is wrong
        results['secondary']['no_R'].append(cos_pre)     # No rotation
    
    # Test invariant words (both rotations should work)
    for word in test_invariant:
        cos_pre, cos_A = compute_cosines(word, R_A, scale_A)
        _, cos_B = compute_cosines(word, R_B, scale_B)
        results['invariant']['correct_R'].append(max(cos_A, cos_B))  # Either works
        results['invariant']['wrong_R'].append(min(cos_A, cos_B))    # Worse one
        results['invariant']['no_R'].append(cos_pre)
    
    # Test unknown words (use auto-classification)
    for word in test_unknown:
        cos_pre, cos_A = compute_cosines(word, R_A, scale_A)
        _, cos_B = compute_cosines(word, R_B, scale_B)
        
        # Auto-classify
        pop = alignment.classify_word(word, proj1, proj2)
        cos_auto = cos_B if pop == 'secondary' else cos_A
        
        results['unknown']['auto_R'].append(cos_auto)
        results['unknown']['R_A'].append(cos_A)
        results['unknown']['R_B'].append(cos_B)
        results['unknown']['no_R'].append(cos_pre)
    
    # Compute summary statistics
    def summarize(arr):
        if len(arr) == 0:
            return {'n': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'pct_gt_0.5': 0, 'pct_gt_0.8': 0}
        arr = np.array(arr)
        return {
            'n': len(arr),
            'mean': arr.mean(),
            'std': arr.std(),
            'min': arr.min(),
            'max': arr.max(),
            'pct_gt_0.5': np.mean(arr > 0.5) * 100,
            'pct_gt_0.8': np.mean(arr > 0.8) * 100,
        }
    
    summary = {}
    
    for group in ['main', 'secondary', 'invariant']:
        if results[group]['no_R']:
            no_R = np.array(results[group]['no_R'])
            correct = np.array(results[group]['correct_R'])
            wrong = np.array(results[group]['wrong_R'])
            
            summary[group] = {
                'n': len(no_R),
                'no_rotation': summarize(no_R),
                'correct_rotation': summarize(correct),
                'wrong_rotation': summarize(wrong),
                'improvement_correct': correct.mean() - no_R.mean() if len(correct) else 0,
                'improvement_wrong': wrong.mean() - no_R.mean() if len(wrong) else 0,
                'pct_correct_better': np.mean(correct > wrong) * 100 if len(correct) else 0,
            }
    
    if results['unknown']['no_R']:
        no_R = np.array(results['unknown']['no_R'])
        auto = np.array(results['unknown']['auto_R'])
        R_A_arr = np.array(results['unknown']['R_A'])
        R_B_arr = np.array(results['unknown']['R_B'])
        
        summary['unknown'] = {
            'n': len(no_R),
            'no_rotation': summarize(no_R),
            'auto_rotation': summarize(auto),
            'R_A': summarize(R_A_arr),
            'R_B': summarize(R_B_arr),
            'improvement_auto': auto.mean() - no_R.mean(),
        }
    
    # Print report
    if verbose:
        print("\n" + "=" * 70)
        print("DETAILED EVALUATION RESULTS")
        print("=" * 70)
        
        for group in ['main', 'secondary', 'invariant']:
            if group in summary and summary[group]['n'] > 0:
                s = summary[group]
                print(f"\n{'─' * 50}")
                print(f"{group.upper()} LINE ({s['n']} words)")
                print(f"{'─' * 50}")
                print(f"  No rotation:      mean={s['no_rotation']['mean']:.4f}")
                print(f"  Correct rotation: mean={s['correct_rotation']['mean']:.4f} (Δ={s['improvement_correct']:+.4f})")
                print(f"  Wrong rotation:   mean={s['wrong_rotation']['mean']:.4f} (Δ={s['improvement_wrong']:+.4f})")
                print(f"  % where correct > wrong: {s['pct_correct_better']:.1f}%")
                print(f"  % > 0.5 (correct): {s['correct_rotation']['pct_gt_0.5']:.1f}%")
                print(f"  % > 0.8 (correct): {s['correct_rotation']['pct_gt_0.8']:.1f}%")
        
        if 'unknown' in summary and summary['unknown']['n'] > 0:
            s = summary['unknown']
            print(f"\n{'─' * 50}")
            print(f"UNKNOWN (auto-classified) ({s['n']} words)")
            print(f"{'─' * 50}")
            print(f"  No rotation:   mean={s['no_rotation']['mean']:.4f}")
            print(f"  Auto rotation: mean={s['auto_rotation']['mean']:.4f} (Δ={s['improvement_auto']:+.4f})")
            print(f"  R_A only:      mean={s['R_A']['mean']:.4f}")
            print(f"  R_B only:      mean={s['R_B']['mean']:.4f}")
            print(f"  % > 0.5 (auto): {s['auto_rotation']['pct_gt_0.5']:.1f}%")
        
        # Overall summary
        print(f"\n{'=' * 70}")
        print("SUMMARY")
        print("=" * 70)
        
        all_correct = []
        all_wrong = []
        all_no_R = []
        for group in ['main', 'secondary', 'invariant']:
            if results[group]['correct_R']:
                all_correct.extend(results[group]['correct_R'])
                all_wrong.extend(results[group]['wrong_R'])
                all_no_R.extend(results[group]['no_R'])
        
        if all_correct:
            all_correct = np.array(all_correct)
            all_wrong = np.array(all_wrong)
            all_no_R = np.array(all_no_R)
            
            print(f"\nAcross all classified words ({len(all_correct)}):")
            print(f"  No rotation:      {all_no_R.mean():.4f}")
            print(f"  Correct rotation: {all_correct.mean():.4f} (Δ={all_correct.mean() - all_no_R.mean():+.4f})")
            print(f"  Wrong rotation:   {all_wrong.mean():.4f} (Δ={all_wrong.mean() - all_no_R.mean():+.4f})")
            print(f"  % where correct > wrong: {np.mean(all_correct > all_wrong) * 100:.1f}%")
            print(f"  % improved by correct R: {np.mean(all_correct > all_no_R) * 100:.1f}%")
    
    return summary


# =============================================================================
# COMMAND LINE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual-population alignment")
    parser.add_argument("--emb1", required=True, help="Path to source embedding")
    parser.add_argument("--name1", default="source", help="Name of source")
    parser.add_argument("--emb2", required=True, help="Path to target embedding")
    parser.add_argument("--name2", default="target", help="Name of target")
    parser.add_argument("--max-words", type=int, default=50000)
    parser.add_argument("--basis-size", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--expand-sim", type=float, default=0.7)
    parser.add_argument("--expand-drift", type=float, default=0.3)
    
    args = parser.parse_args()
    
    from sense_explorer.core import SenseExplorer
    
    print("Loading embeddings...")
    se1 = SenseExplorer.from_file(args.emb1, max_words=args.max_words, verbose=True)
    se2 = SenseExplorer.from_file(args.emb2, max_words=args.max_words,
                                   target_dim=se1.dim, verbose=True)
    
    alignment, proj1, proj2, word_stats = compute_dual_alignment(
        se1, se2,
        source_name=args.name1,
        target_name=args.name2,
        n_classification_samples=args.n_samples,
        max_basis_size=args.basis_size,
        anchor_expansion_sim=args.expand_sim,
        anchor_expansion_drift=args.expand_drift,
        holdout_ratio=0.2,
        verbose=True
    )
    
    print("\n" + alignment.report())
    
    # Evaluate
    summary = evaluate_dual_alignment(alignment, proj1, proj2, word_stats, verbose=True)
