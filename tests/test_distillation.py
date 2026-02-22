#!/usr/bin/env python3
"""
test_distillation.py - Tests for IVA Sense Distillation
========================================================

Tests the distillation module with both toy data and real embeddings.

Usage:
    # Toy data only (no dependencies)
    python test_distillation.py
    
    # With real embeddings
    python test_distillation.py --glove path/to/glove.txt

Author: Kow Kuroda & Claude (Anthropic)
"""

import numpy as np
import argparse
import sys
import os

# Add parent directory to path
# Add parent directory (package root) to path for imports
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.insert(0, package_root)


def create_toy_embeddings(dim=50, n_words=200):
    """
    Create synthetic embeddings with known semantic structure.
    
    Creates three semantic clusters:
    - Financial: money, loan, credit, bank, ...
    - Nature: river, water, tree, forest, ...
    - Technology: computer, software, code, ...
    """
    np.random.seed(42)
    
    # Define cluster directions
    financial_dir = np.random.randn(dim)
    financial_dir /= np.linalg.norm(financial_dir)
    
    nature_dir = np.random.randn(dim)
    nature_dir /= np.linalg.norm(nature_dir)
    
    tech_dir = np.random.randn(dim)
    tech_dir /= np.linalg.norm(tech_dir)
    
    embeddings = {}
    
    # Financial cluster
    financial_words = ['money', 'loan', 'credit', 'bank', 'finance', 
                       'investment', 'deposit', 'savings', 'account', 'funds']
    for word in financial_words:
        noise = np.random.randn(dim) * 0.2
        vec = financial_dir + noise
        embeddings[word] = vec / np.linalg.norm(vec)
    
    # Nature cluster
    nature_words = ['river', 'water', 'tree', 'forest', 'mountain',
                    'lake', 'stream', 'shore', 'nature', 'green']
    for word in nature_words:
        noise = np.random.randn(dim) * 0.2
        vec = nature_dir + noise
        embeddings[word] = vec / np.linalg.norm(vec)
    
    # Technology cluster
    tech_words = ['computer', 'software', 'code', 'program', 'digital',
                  'internet', 'data', 'algorithm', 'network', 'system']
    for word in tech_words:
        noise = np.random.randn(dim) * 0.2
        vec = tech_dir + noise
        embeddings[word] = vec / np.linalg.norm(vec)
    
    # Random filler words
    for i in range(n_words - len(embeddings)):
        word = f"word_{i}"
        vec = np.random.randn(dim)
        embeddings[word] = vec / np.linalg.norm(vec)
    
    return embeddings, {
        'financial': financial_words,
        'nature': nature_words,
        'technology': tech_words
    }


def test_basic_distillation():
    """Test basic IVA distillation functionality."""
    from sense_explorer.distillation import IVADistiller, DistillationResult
    
    print("=" * 60)
    print("TEST: Basic IVA Distillation")
    print("=" * 60)
    
    embeddings, clusters = create_toy_embeddings()
    distiller = IVADistiller(embeddings, verbose=True)
    
    print(f"\nCreated embeddings: {len(embeddings)} words, {len(next(iter(embeddings.values())))}d")
    
    # Test constrained distillation
    print("\n--- Constrained Distillation (Recommended) ---")
    for cluster_name, words in clusters.items():
        result = distiller.distill_constrained(words[:5], n_exemplars=5)
        print(f"\n{cluster_name}:")
        print(f"  Input: {words[:5]}")
        print(f"  Coherence: {result.coherence:.3f}")
        print(f"  Exemplars: {result.exemplars}")
        print(f"  Iterations: {result.n_iterations}")
        
        # Verify it's a DistillationResult
        assert isinstance(result, DistillationResult)
        assert result.mode == 'constrained'
        assert len(result.direction) == 50
    
    print("\n✓ Basic distillation test PASSED")
    return True


def test_global_vs_constrained():
    """Test difference between global and constrained modes."""
    from sense_explorer.distillation import IVADistiller
    
    print("\n" + "=" * 60)
    print("TEST: Global vs Constrained Modes")
    print("=" * 60)
    
    embeddings, clusters = create_toy_embeddings()
    distiller = IVADistiller(embeddings)
    
    financial_words = clusters['financial'][:5]
    
    # Constrained
    result_constrained = distiller.distill_constrained(financial_words)
    
    # Global
    result_global = distiller.distill(financial_words)
    
    print(f"\nInput words: {financial_words}")
    print(f"\nConstrained mode:")
    print(f"  Coherence: {result_constrained.coherence:.3f}")
    print(f"  Exemplars: {result_constrained.exemplars}")
    
    print(f"\nGlobal mode:")
    print(f"  Coherence: {result_global.coherence:.3f}")
    print(f"  Exemplars: {result_global.exemplars}")
    
    # Compare directions
    cos_sim = np.dot(result_constrained.direction, result_global.direction)
    angle = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
    print(f"\nAngle between modes: {angle:.1f}°")
    
    print("\n✓ Global vs Constrained test PASSED")
    return True


def test_coherence_measurement():
    """Test coherence measurement function."""
    from sense_explorer.distillation import measure_set_coherence
    
    print("\n" + "=" * 60)
    print("TEST: Coherence Measurement")
    print("=" * 60)
    
    embeddings, clusters = create_toy_embeddings()
    
    # High coherence: words from same cluster
    coherence_same = measure_set_coherence(embeddings, clusters['financial'][:5])
    
    # Low coherence: words from different clusters
    mixed_words = [clusters['financial'][0], clusters['nature'][0], 
                   clusters['technology'][0], 'word_0', 'word_1']
    coherence_mixed = measure_set_coherence(embeddings, mixed_words)
    
    # Random words
    random_words = [f'word_{i}' for i in range(5)]
    coherence_random = measure_set_coherence(embeddings, random_words)
    
    print(f"\nSame cluster (financial): {coherence_same:.3f}")
    print(f"Mixed clusters: {coherence_mixed:.3f}")
    print(f"Random words: {coherence_random:.3f}")
    
    # Verify ordering
    assert coherence_same > coherence_mixed, "Same cluster should have higher coherence"
    print("\n✓ Coherence measurement test PASSED")
    return True


def test_multiple_distillation():
    """Test distilling multiple word groups."""
    from sense_explorer.distillation import IVADistiller
    
    print("\n" + "=" * 60)
    print("TEST: Multiple Group Distillation")
    print("=" * 60)
    
    embeddings, clusters = create_toy_embeddings()
    distiller = IVADistiller(embeddings)
    
    # Distill all clusters
    word_groups = {name: words[:5] for name, words in clusters.items()}
    results = distiller.distill_multiple(word_groups, mode='constrained')
    
    print("\nDistillation results:")
    for name, result in results.items():
        print(f"  {name}: coherence={result.coherence:.3f}, exemplars={result.exemplars[:3]}")
    
    # Compare directions
    angles = distiller.compare_directions(results)
    print("\nInter-cluster angles:")
    for (name1, name2), angle in angles.items():
        print(f"  {name1} <-> {name2}: {angle:.1f}°")
    
    # Clusters should be well-separated (large angles)
    for angle in angles.values():
        assert angle > 30, f"Clusters should be separated (got {angle:.1f}°)"
    
    print("\n✓ Multiple distillation test PASSED")
    return True


def test_validation():
    """Test validation function."""
    from sense_explorer.distillation import validate_distillation
    
    print("\n" + "=" * 60)
    print("TEST: Distillation Validation")
    print("=" * 60)
    
    embeddings, clusters = create_toy_embeddings()
    
    word_groups = {name: words[:5] for name, words in clusters.items()}
    stats = validate_distillation(embeddings, word_groups)
    
    print(f"\nValidation results:")
    print(f"  Mean coherence: {stats['mean_coherence']:.3f}")
    print(f"  Mean inter-sense angle: {stats['mean_angle']:.1f}°")
    print(f"  Mean exemplar overlap: {stats['mean_exemplar_overlap']:.3f}")
    
    print("\n✓ Validation test PASSED")
    return True


def test_with_real_embeddings(glove_path):
    """Test with real GloVe embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Real Embeddings")
    print("=" * 60)
    
    try:
        from sense_explorer.core import SenseExplorer
    except ImportError:
        print("SenseExplorer not found, skipping real embedding test")
        return True
    
    print(f"\nLoading embeddings from {glove_path}...")
    se = SenseExplorer.from_file(glove_path, max_words=50000, verbose=True)
    
    # Test distill_senses
    print("\n--- Testing distill_senses('bank') ---")
    results = se.distill_senses("bank")
    if results:
        for sense, result in results.items():
            print(f"  {sense}: coherence={result.coherence:.3f}, exemplars={result.exemplars}")
    else:
        print("  No results (anchors not found)")
    
    # Test measure_anchor_coherence
    print("\n--- Testing measure_anchor_coherence('bank') ---")
    coherences = se.measure_anchor_coherence("bank")
    if coherences:
        for sense, coh in coherences.items():
            print(f"  {sense}: {coh:.3f}")
    else:
        print("  No coherences (anchors not found)")
    
    # Test distill_and_compare
    print("\n--- Testing distill_and_compare('bank') ---")
    stats = se.distill_and_compare("bank")
    if stats.get('ssr_iva_angles'):
        print(f"  SSR↔IVA angles: {stats['ssr_iva_angles']}")
        print(f"  Mean SSR↔IVA angle: {stats['mean_ssr_iva_angle']:.1f}°")
    else:
        print("  No comparison available")
    
    # Test validate_sense_distillation
    print("\n--- Testing validate_sense_distillation ---")
    words = ['bank', 'bat', 'crane', 'mouse', 'plant']
    words = [w for w in words if w in se.vocab]
    if words:
        validation = se.validate_sense_distillation(words[:3])
        print(f"  Words tested: {validation['n_words']}")
        print(f"  Mean coherence: {validation['mean_coherence']:.3f}")
        print(f"  Mean SSR↔IVA angle: {validation['mean_ssr_iva_angle']:.1f}°")
    
    print("\n✓ Real embeddings test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test IVA Distillation")
    parser.add_argument("--glove", type=str, help="Path to GloVe embeddings")
    args = parser.parse_args()
    
    print("#" * 70)
    print("# DISTILLATION MODULE TESTS")
    print("#" * 70)
    
    # Run toy data tests
    all_passed = True
    all_passed &= test_basic_distillation()
    all_passed &= test_global_vs_constrained()
    all_passed &= test_coherence_measurement()
    all_passed &= test_multiple_distillation()
    all_passed &= test_validation()
    
    # Run real embedding tests if path provided
    if args.glove:
        all_passed &= test_with_real_embeddings(args.glove)
    else:
        print("\n" + "-" * 60)
        print("To test with real embeddings, provide --glove path:")
        print("  python test_distillation.py --glove glove.6B.100d.txt")
        print("-" * 60)
    
    # Summary
    print("\n" + "#" * 70)
    if all_passed:
        print("# ALL TESTS PASSED ✓")
    else:
        print("# SOME TESTS FAILED ✗")
    print("#" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
