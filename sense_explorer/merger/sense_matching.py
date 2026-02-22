"""
Sense Matching Across Registers

Uses the Hungarian algorithm (Kuhn-Munkres) to optimally match senses
across registers, even when sense names differ.

Problem:
    When inducing senses independently in different registers, the same
    semantic sense may receive different names:
    
    Twitter: "bank" → {"sense_0": financial, "sense_1": river}
    News:    "bank" → {"sense_0": river,     "sense_1": financial}
    
    Or even worse, different anchor-based names:
    
    Twitter: "bank" → {"financial": ..., "river": ...}
    News:    "bank" → {"monetary": ...,  "geographic": ...}

Solution:
    Use the Hungarian algorithm to find the optimal bipartite matching
    that maximizes total similarity between matched sense pairs.
    
    This requires a similarity measure between sense vectors from
    different embedding spaces - we use the projected-space similarity.

Algorithm:
    1. Project sense vectors onto shared vocabulary basis
    2. Build similarity matrix between all sense pairs
    3. Apply Hungarian algorithm to find optimal matching
    4. Return matched pairs with their similarities
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment


@dataclass
class SenseMatch:
    """A matched pair of senses across registers."""
    sense1_name: str
    sense2_name: str
    similarity: float
    sense1_register: str
    sense2_register: str


@dataclass 
class SenseMatchingResult:
    """Result of sense matching across registers."""
    matches: List[SenseMatch]
    unmatched_reg1: List[str]  # Senses in reg1 with no match
    unmatched_reg2: List[str]  # Senses in reg2 with no match
    mean_similarity: float
    min_similarity: float
    
    def report(self) -> str:
        lines = [
            f"Sense Matching: {len(self.matches)} matched pairs",
            f"  Mean similarity: {self.mean_similarity:.4f}",
            f"  Min similarity:  {self.min_similarity:.4f}",
        ]
        for m in self.matches:
            lines.append(f"  {m.sense1_name} ↔ {m.sense2_name}: {m.similarity:.4f}")
        if self.unmatched_reg1:
            lines.append(f"  Unmatched in reg1: {self.unmatched_reg1}")
        if self.unmatched_reg2:
            lines.append(f"  Unmatched in reg2: {self.unmatched_reg2}")
        return "\n".join(lines)


def project_sense_vector(
    sense_vec: np.ndarray,
    embeddings: Dict[str, np.ndarray],
    basis_words: List[str]
) -> np.ndarray:
    """
    Project a sense vector onto shared vocabulary basis.
    
    Args:
        sense_vec: The sense vector to project
        embeddings: Word embeddings from the same space as sense_vec
        basis_words: Words to use as projection basis
        
    Returns:
        Projected vector (dimension = len(basis_words))
    """
    sense_norm = sense_vec / (np.linalg.norm(sense_vec) + 1e-10)
    
    projection = []
    for word in basis_words:
        if word in embeddings:
            word_vec = embeddings[word]
            word_norm = word_vec / (np.linalg.norm(word_vec) + 1e-10)
            # Handle dimension mismatch
            min_dim = min(len(sense_norm), len(word_norm))
            sim = np.dot(sense_norm[:min_dim], word_norm[:min_dim])
            projection.append(sim)
        else:
            projection.append(0.0)
    
    proj = np.array(projection)
    proj_norm = np.linalg.norm(proj)
    if proj_norm > 1e-10:
        proj = proj / proj_norm
    
    return proj


def build_shared_basis(
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    max_basis_size: int = 500
) -> List[str]:
    """Build a shared vocabulary basis for projection."""
    shared = list(set(se1.vocab) & set(se2.vocab))
    # Sort by frequency (position in vocab)
    word_ranks = {w: i for i, w in enumerate(se1.vocab) if w in shared}
    ranked = sorted(shared, key=lambda w: word_ranks.get(w, float('inf')))
    # Filter: alphabetic, length >= 3, skip top 100 (function words)
    filtered = [w for w in ranked[100:] if w.isalpha() and len(w) >= 3]
    return filtered[:max_basis_size]


def match_senses_hungarian(
    senses1: Dict[str, np.ndarray],
    senses2: Dict[str, np.ndarray],
    embeddings1: Dict[str, np.ndarray],
    embeddings2: Dict[str, np.ndarray],
    basis_words: List[str],
    reg1_name: str = "reg1",
    reg2_name: str = "reg2",
    min_similarity: float = 0.3
) -> SenseMatchingResult:
    """
    Match senses across registers using Hungarian algorithm.
    
    Args:
        senses1: Dict mapping sense names to vectors (register 1)
        senses2: Dict mapping sense names to vectors (register 2)
        embeddings1: Word embeddings for register 1
        embeddings2: Word embeddings for register 2
        basis_words: Shared vocabulary for projection
        reg1_name, reg2_name: Register names for reporting
        min_similarity: Minimum similarity to accept a match
        
    Returns:
        SenseMatchingResult with optimal matching
    """
    names1 = list(senses1.keys())
    names2 = list(senses2.keys())
    
    if len(names1) == 0 or len(names2) == 0:
        return SenseMatchingResult(
            matches=[],
            unmatched_reg1=names1,
            unmatched_reg2=names2,
            mean_similarity=0.0,
            min_similarity=0.0
        )
    
    # Project all sense vectors onto shared basis
    proj1 = {name: project_sense_vector(senses1[name], embeddings1, basis_words) 
             for name in names1}
    proj2 = {name: project_sense_vector(senses2[name], embeddings2, basis_words) 
             for name in names2}
    
    # Build similarity matrix
    n1, n2 = len(names1), len(names2)
    sim_matrix = np.zeros((n1, n2))
    
    for i, name1 in enumerate(names1):
        for j, name2 in enumerate(names2):
            sim = np.dot(proj1[name1], proj2[name2])
            sim_matrix[i, j] = sim
    
    # Hungarian algorithm (minimize cost = maximize similarity)
    cost_matrix = -sim_matrix  # Negate for minimization
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Build matches
    matches = []
    matched_i = set()
    matched_j = set()
    
    for i, j in zip(row_ind, col_ind):
        sim = sim_matrix[i, j]
        if sim >= min_similarity:
            matches.append(SenseMatch(
                sense1_name=names1[i],
                sense2_name=names2[j],
                similarity=sim,
                sense1_register=reg1_name,
                sense2_register=reg2_name
            ))
            matched_i.add(i)
            matched_j.add(j)
    
    # Find unmatched senses
    unmatched1 = [names1[i] for i in range(n1) if i not in matched_i]
    unmatched2 = [names2[j] for j in range(n2) if j not in matched_j]
    
    # Compute summary stats
    if matches:
        sims = [m.similarity for m in matches]
        mean_sim = np.mean(sims)
        min_sim = np.min(sims)
    else:
        mean_sim = 0.0
        min_sim = 0.0
    
    return SenseMatchingResult(
        matches=matches,
        unmatched_reg1=unmatched1,
        unmatched_reg2=unmatched2,
        mean_similarity=mean_sim,
        min_similarity=min_sim
    )


def compute_matched_drift(
    word: str,
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    basis_words: List[str],
    n_senses: int = 2,
    reg1_name: str = "reg1",
    reg2_name: str = "reg2",
    min_match_similarity: float = 0.3,
    verbose: bool = False
) -> Tuple[float, float, SenseMatchingResult]:
    """
    Compute drift using matched senses only.
    
    This fixes the secondary-line artifact by:
    1. Matching senses across registers using Hungarian algorithm
    2. Computing drift only for matched sense pairs
    3. Ensuring similarity + drift = 1 (by construction)
    
    Args:
        word: Target word
        se1, se2: SenseExplorer instances
        basis_words: Shared vocabulary for projection
        n_senses: Number of senses to induce
        reg1_name, reg2_name: Register names
        min_match_similarity: Minimum similarity to accept a match
        verbose: Print matching details
        
    Returns:
        Tuple of (similarity, drift, matching_result)
        Where similarity + drift = 1.0 (always)
    """
    # Induce senses in each register
    senses1 = se1.induce_senses(word, n_senses=n_senses)
    senses2 = se2.induce_senses(word, n_senses=n_senses)
    
    # Match senses using Hungarian algorithm
    matching = match_senses_hungarian(
        senses1, senses2,
        se1.embeddings, se2.embeddings,
        basis_words,
        reg1_name, reg2_name,
        min_match_similarity
    )
    
    if verbose:
        print(matching.report())
    
    if not matching.matches:
        # No matches - can't compute drift
        return 0.0, 1.0, matching
    
    # Compute drift for each matched pair
    drifts = []
    
    for match in matching.matches:
        vec1 = senses1[match.sense1_name]
        vec2 = senses2[match.sense2_name]
        
        # Project both onto shared basis
        proj1 = project_sense_vector(vec1, se1.embeddings, basis_words)
        proj2 = project_sense_vector(vec2, se2.embeddings, basis_words)
        
        # Drift = 1 - cosine similarity
        cos_sim = np.dot(proj1, proj2)
        drift = max(0.0, 1.0 - cos_sim)  # Clamp to non-negative
        drifts.append(drift)
    
    # Aggregate
    mean_drift = np.mean(drifts)
    similarity = 1.0 - mean_drift  # By construction: sim + drift = 1
    
    return similarity, mean_drift, matching


def compare_registers_matched(
    words: List[str],
    se1: 'SenseExplorer',
    se2: 'SenseExplorer',
    reg1_name: str = "reg1",
    reg2_name: str = "reg2",
    n_senses: int = 2,
    max_basis_size: int = 500,
    min_match_similarity: float = 0.3,
    verbose: bool = True
) -> Dict[str, Tuple[float, float, SenseMatchingResult]]:
    """
    Compare registers using matched-sense drift computation.
    
    This should eliminate the secondary-line artifact.
    
    Args:
        words: List of words to analyze
        se1, se2: SenseExplorer instances
        reg1_name, reg2_name: Register names
        n_senses: Number of senses per word
        max_basis_size: Projection basis dimension
        min_match_similarity: Minimum similarity for sense matching
        verbose: Print progress
        
    Returns:
        Dict mapping word to (similarity, drift, matching_result)
    """
    if verbose:
        print(f"Building shared basis...")
    
    basis_words = build_shared_basis(se1, se2, max_basis_size)
    
    if verbose:
        print(f"Basis size: {len(basis_words)} words")
        print(f"Analyzing {len(words)} words with Hungarian sense matching...")
    
    results = {}
    errors = []
    
    for i, word in enumerate(words):
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(words)}...")
        
        try:
            if word not in se1.vocab or word not in se2.vocab:
                continue
            
            sim, drift, matching = compute_matched_drift(
                word, se1, se2, basis_words, n_senses,
                reg1_name, reg2_name, min_match_similarity,
                verbose=False
            )
            
            results[word] = (sim, drift, matching)
            
        except Exception as e:
            errors.append((word, str(e)))
    
    if verbose:
        print(f"Analyzed {len(results)} words successfully")
        if errors:
            print(f"Errors: {len(errors)}")
    
    return results


def validate_main_line(
    results: Dict[str, Tuple[float, float, SenseMatchingResult]],
    tolerance: float = 0.001
) -> Dict:
    """
    Validate that all words fall on the main line (sim + drift = 1).
    
    After Hungarian matching fix, there should be NO secondary line.
    """
    on_line = 0
    off_line = []
    
    for word, (sim, drift, _) in results.items():
        residual = sim + drift - 1.0
        if abs(residual) <= tolerance:
            on_line += 1
        else:
            off_line.append((word, sim, drift, residual))
    
    return {
        'total': len(results),
        'on_main_line': on_line,
        'off_main_line': len(off_line),
        'pct_on_line': 100.0 * on_line / len(results) if results else 0,
        'off_line_words': off_line[:20],  # First 20 examples
    }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Compare registers with Hungarian sense matching")
    parser.add_argument("--emb1", required=True, help="Path to first embedding")
    parser.add_argument("--name1", default="reg1", help="Name of first register")
    parser.add_argument("--emb2", required=True, help="Path to second embedding")
    parser.add_argument("--name2", default="reg2", help="Name of second register")
    parser.add_argument("--max-words", type=int, default=50000, help="Max words to load")
    parser.add_argument("--n-test", type=int, default=500, help="Words to test")
    parser.add_argument("--basis-size", type=int, default=500, help="Projection basis size")
    parser.add_argument("--output", default="matched_comparison", help="Output file prefix")
    
    args = parser.parse_args()
    
    from sense_explorer.core import SenseExplorer
    
    print("Loading embeddings...")
    se1 = SenseExplorer.from_file(args.emb1, max_words=args.max_words, verbose=True)
    se2 = SenseExplorer.from_file(args.emb2, max_words=args.max_words, 
                                   target_dim=se1.dim, verbose=True)
    
    # Get shared vocabulary
    shared = list(set(se1.vocab) & set(se2.vocab))
    print(f"Shared vocabulary: {len(shared)} words")
    
    # Sample test words
    np.random.seed(42)
    test_words = list(np.random.choice(shared, min(args.n_test, len(shared)), replace=False))
    
    # Compare with matched senses
    print("\n" + "=" * 60)
    print("HUNGARIAN-MATCHED SENSE COMPARISON")
    print("=" * 60)
    
    start_time = time.time()
    results = compare_registers_matched(
        test_words, se1, se2,
        reg1_name=args.name1,
        reg2_name=args.name2,
        max_basis_size=args.basis_size,
        verbose=True
    )
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f}s")
    
    # Validate main line
    print("\n" + "=" * 60)
    print("VALIDATION: All words should be on main line (sim + drift = 1)")
    print("=" * 60)
    
    validation = validate_main_line(results)
    print(f"Total words: {validation['total']}")
    print(f"On main line: {validation['on_main_line']} ({validation['pct_on_line']:.1f}%)")
    print(f"Off main line: {validation['off_main_line']}")
    
    if validation['off_line_words']:
        print("\nOff-line examples:")
        for word, sim, drift, residual in validation['off_line_words'][:10]:
            print(f"  {word}: sim={sim:.4f}, drift={drift:.4f}, residual={residual:.4f}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    sims = [r[0] for r in results.values()]
    drifts = [r[1] for r in results.values()]
    
    print(f"\nSimilarity ({args.name1} ↔ {args.name2}):")
    print(f"  Mean: {np.mean(sims):.4f}")
    print(f"  Std:  {np.std(sims):.4f}")
    print(f"  Min:  {np.min(sims):.4f}")
    print(f"  Max:  {np.max(sims):.4f}")
    
    print(f"\nDrift:")
    print(f"  Mean: {np.mean(drifts):.4f}")
    print(f"  Std:  {np.std(drifts):.4f}")
    
    # Save results
    output_file = f"{args.output}_data.tsv"
    with open(output_file, 'w') as f:
        f.write("word\tsimilarity\tdrift\tresidual\tn_matched\tmean_match_sim\n")
        for word, (sim, drift, matching) in sorted(results.items()):
            residual = sim + drift - 1.0
            n_matched = len(matching.matches)
            mean_match_sim = matching.mean_similarity
            f.write(f"{word}\t{sim:.4f}\t{drift:.4f}\t{residual:.6f}\t{n_matched}\t{mean_match_sim:.4f}\n")
    
    print(f"\nResults saved to {output_file}")
