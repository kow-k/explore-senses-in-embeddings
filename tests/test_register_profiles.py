"""
Tests for Register-Specific Sense Profiles

Tests the analysis of how word senses differ across registers
(e.g., formal Wikipedia vs informal Twitter).
"""

import sys
import numpy as np
from typing import Dict, List
import argparse


def create_toy_register_embeddings(dim: int = 50):
    """
    Create toy embeddings simulating register differences.
    
    - 'formal' register: more technical/academic vocabulary
    - 'informal' register: more casual/colloquial vocabulary
    
    Words like 'bank' will have:
    - Similar financial sense in both
    - Different associations (formal: institution, informal: broke/payday)
    """
    np.random.seed(42)
    
    # Base vocabulary (shared)
    common_words = [f"word_{i}" for i in range(200)]
    
    formal = {}
    informal = {}
    
    # Generate correlated embeddings for common words
    for word in common_words:
        base = np.random.randn(dim)
        formal[word] = base + np.random.randn(dim) * 0.3
        informal[word] = base + np.random.randn(dim) * 0.3
    
    # === Financial sense (similar in both, different neighbors) ===
    financial_base = np.random.randn(dim)
    financial_base /= np.linalg.norm(financial_base)
    
    # Formal financial words
    formal_financial = ['institution', 'deposit', 'investment', 'portfolio', 
                        'securities', 'dividend', 'equity', 'capital']
    for word in formal_financial:
        formal[word] = financial_base + np.random.randn(dim) * 0.2
        # Different vector in informal (same word, different context)
        informal[word] = financial_base + np.random.randn(dim) * 0.4
    
    # Informal financial words
    informal_financial = ['money', 'broke', 'payday', 'cash', 'bucks', 
                          'loaded', 'wallet', 'atm']
    for word in informal_financial:
        informal[word] = financial_base + np.random.randn(dim) * 0.2
        formal[word] = financial_base + np.random.randn(dim) * 0.5
    
    # === River sense (more prominent in formal) ===
    river_base = np.random.randn(dim)
    river_base /= np.linalg.norm(river_base)
    
    formal_river = ['riverbank', 'embankment', 'tributary', 'watershed',
                    'erosion', 'sediment', 'floodplain', 'estuary']
    for word in formal_river:
        formal[word] = river_base + np.random.randn(dim) * 0.2
        informal[word] = np.random.randn(dim)  # Random in informal
    
    informal_river = ['river', 'stream', 'water', 'fishing']
    for word in informal_river:
        informal[word] = river_base + np.random.randn(dim) * 0.3
        formal[word] = river_base + np.random.randn(dim) * 0.3
    
    # === Slang sense (informal only) ===
    slang_base = np.random.randn(dim)
    slang_base /= np.linalg.norm(slang_base)
    
    informal_slang = ['trust', 'rely', 'count', 'depend', 'bet']
    for word in informal_slang:
        informal[word] = slang_base + np.random.randn(dim) * 0.2
        formal[word] = np.random.randn(dim)  # Random in formal
    
    # === Target word 'bank' ===
    # In formal: mix of financial (0.6) and river (0.4)
    formal['bank'] = 0.6 * financial_base + 0.4 * river_base + np.random.randn(dim) * 0.1
    
    # In informal: mix of financial (0.5), river (0.2), slang (0.3)
    informal['bank'] = 0.5 * financial_base + 0.2 * river_base + 0.3 * slang_base + np.random.randn(dim) * 0.1
    
    # Normalize all
    for word in formal:
        formal[word] = formal[word] / np.linalg.norm(formal[word])
    for word in informal:
        informal[word] = informal[word] / np.linalg.norm(informal[word])
    
    return formal, informal


def test_sense_prevalence():
    """Test sense prevalence computation."""
    print("\n" + "=" * 60)
    print("TEST: Sense Prevalence")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import compute_sense_prevalence
    
    formal, informal = create_toy_register_embeddings(dim=50)
    
    se_formal = SenseExplorer.from_dict(formal, verbose=False)
    se_informal = SenseExplorer.from_dict(informal, verbose=False)
    
    explorers = {"formal": se_formal, "informal": se_informal}
    
    prevalences = compute_sense_prevalence("bank", explorers, n_senses=2)
    
    print(f"Found {len(prevalences)} sense prevalences:")
    for sp in prevalences:
        print(f"  {sp}")
    
    assert len(prevalences) > 0, "Expected at least one sense"
    
    print("\n✓ Sense prevalence test PASSED")
    return True


def test_register_neighbors():
    """Test register-specific neighbor computation."""
    print("\n" + "=" * 60)
    print("TEST: Register Neighbors")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import compute_register_neighbors
    
    formal, informal = create_toy_register_embeddings(dim=50)
    
    se_formal = SenseExplorer.from_dict(formal, verbose=False)
    se_informal = SenseExplorer.from_dict(informal, verbose=False)
    
    explorers = {"formal": se_formal, "informal": se_informal}
    
    reg_neighbors = compute_register_neighbors("bank", explorers, n_senses=2)
    
    print(f"Found neighbors for {len(reg_neighbors)} senses:")
    for sense_name, rn in reg_neighbors.items():
        print(f"\n  {sense_name}:")
        print(f"    Overlap score: {rn.overlap_score:.3f}")
        print(f"    Shared: {rn.shared_neighbors[:5]}")
        for reg, unique in rn.unique_neighbors.items():
            print(f"    {reg} only: {unique[:5]}")
    
    assert len(reg_neighbors) > 0, "Expected at least one sense"
    
    print("\n✓ Register neighbors test PASSED")
    return True


def test_sense_drift():
    """Test sense drift computation."""
    print("\n" + "=" * 60)
    print("TEST: Sense Drift")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import compute_sense_drift
    
    formal, informal = create_toy_register_embeddings(dim=50)
    
    se_formal = SenseExplorer.from_dict(formal, verbose=False)
    se_informal = SenseExplorer.from_dict(informal, verbose=False)
    
    explorers = {"formal": se_formal, "informal": se_informal}
    
    drifts = compute_sense_drift("bank", explorers, n_senses=2)
    
    print(f"Found drift for {len(drifts)} senses:")
    for sense_name, sd in drifts.items():
        print(f"  {sense_name}: mean={sd.mean_drift:.3f}, max={sd.max_drift:.3f}")
    
    assert len(drifts) > 0, "Expected at least one sense"
    
    print("\n✓ Sense drift test PASSED")
    return True


def test_full_register_profile():
    """Test full register profile creation."""
    print("\n" + "=" * 60)
    print("TEST: Full Register Profile")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import create_register_profile
    
    formal, informal = create_toy_register_embeddings(dim=50)
    
    se_formal = SenseExplorer.from_dict(formal, verbose=False)
    se_informal = SenseExplorer.from_dict(informal, verbose=False)
    
    explorers = {"formal": se_formal, "informal": se_informal}
    
    profile = create_register_profile("bank", explorers, n_senses=2, verbose=True)
    
    print(f"\n{profile}")
    print(f"\nPrevalence dict: {profile.prevalence}")
    print(f"Sense drift dict: {profile.sense_drift}")
    
    # Print full report
    print("\n" + profile.report(verbose=True))
    
    assert profile.overall_register_similarity > 0
    
    print("\n✓ Full register profile test PASSED")
    return True


def test_compare_registers():
    """Test multi-word register comparison."""
    print("\n" + "=" * 60)
    print("TEST: Compare Registers (Multiple Words)")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import compare_registers, summarize_register_comparison
    
    formal, informal = create_toy_register_embeddings(dim=50)
    
    # Add more test words
    np.random.seed(123)
    for word in ['money', 'river', 'trust']:
        if word not in formal:
            formal[word] = np.random.randn(50)
            formal[word] /= np.linalg.norm(formal[word])
        if word not in informal:
            informal[word] = np.random.randn(50)
            informal[word] /= np.linalg.norm(informal[word])
    
    se_formal = SenseExplorer.from_dict(formal, verbose=False)
    se_informal = SenseExplorer.from_dict(informal, verbose=False)
    
    explorers = {"formal": se_formal, "informal": se_informal}
    
    words = ['bank', 'money', 'river']
    profiles = compare_registers(words, explorers, verbose=True)
    
    print(f"\nProfiles created: {len(profiles)}")
    
    summary = summarize_register_comparison(profiles)
    print(f"\n{summary}")
    
    assert len(profiles) > 0
    
    print("\n✓ Compare registers test PASSED")
    return True


def test_with_real_embeddings(wiki_path, twitter_path):
    """Test with real GloVe embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Real Embeddings Register Profiles")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import (
        create_register_profile, 
        compare_registers,
        summarize_register_comparison,
        analyze_register_specificity
    )
    
    # Load embeddings
    print(f"\nLoading Wikipedia embeddings from {wiki_path}...")
    se_wiki = SenseExplorer.from_file(wiki_path, max_words=50000, verbose=True)
    wiki_dim = se_wiki.dim
    
    print(f"\nLoading Twitter embeddings from {twitter_path}...")
    se_twitter = SenseExplorer.from_file(
        twitter_path,
        max_words=50000,
        target_dim=wiki_dim,
        verbose=True
    )
    
    explorers = {"wiki": se_wiki, "twitter": se_twitter}
    
    # Single word profile
    print("\n" + "=" * 40)
    print("Single Word Profile: 'bank'")
    print("=" * 40)
    
    profile = create_register_profile("bank", explorers, n_senses=2, verbose=True)
    print(profile.report(verbose=True))
    
    # Multi-word comparison
    print("\n" + "=" * 40)
    print("Multi-Word Comparison")
    print("=" * 40)
    
    test_words = ['bank', 'rock', 'apple', 'spring', 'bat', 'crane']
    profiles = compare_registers(test_words, explorers, verbose=True)
    
    print("\n" + summarize_register_comparison(profiles))
    
    # Register specificity clustering
    print("\n" + "=" * 40)
    print("Register Specificity Clustering")
    print("=" * 40)
    
    # Use a larger word list for better clustering
    cluster_words = [
        'bank', 'rock', 'apple', 'spring', 'bat', 'crane',
        'light', 'run', 'play', 'break', 'fall', 'match',
        'cold', 'fair', 'fine', 'mean', 'sound', 'watch'
    ]
    
    analysis = analyze_register_specificity(
        cluster_words,
        explorers,
        n_senses=2,
        method="threshold",
        verbose=True
    )
    
    print("\n" + analysis.report())
    
    # Generate plot
    print("\n" + "=" * 40)
    print("Generating Register Specificity Plot")
    print("=" * 40)
    
    try:
        from sense_explorer.merger import plot_register_specificity
        
        fig = plot_register_specificity(
            analysis.profiles,
            output_path="register_specificity.png",
            title="Word Register Specificity (Wiki vs Twitter)"
        )
        if fig:
            print("Plot saved to register_specificity.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    # Show detailed profiles for most divergent
    print("\n" + "-" * 40)
    print("DETAILED PROFILES (most divergent)")
    print("-" * 40)
    
    sorted_profiles = sorted(profiles.items(), 
                             key=lambda x: x[1].overall_register_similarity)
    
    for word, prof in sorted_profiles[:3]:
        print(f"\n{prof.report(verbose=False)}")
    
    print("\n✓ Real embeddings register profile test PASSED")
    return True


def run_all_tests(wiki_path=None, twitter_path=None):
    """Run all register profile tests."""
    print("#" * 70)
    print("# REGISTER-SPECIFIC SENSE PROFILE TESTS")
    print("#" * 70)
    
    tests = [
        ("Sense Prevalence", test_sense_prevalence),
        ("Register Neighbors", test_register_neighbors),
        ("Sense Drift", test_sense_drift),
        ("Full Register Profile", test_full_register_profile),
        ("Compare Registers", test_compare_registers),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Real embeddings test (optional)
    if wiki_path and twitter_path:
        try:
            test_with_real_embeddings(wiki_path, twitter_path)
            passed += 1
        except Exception as e:
            print(f"\n✗ Real Embeddings test FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#" * 70)
    if failed == 0:
        print(f"# ALL {passed} TESTS PASSED ✓")
    else:
        print(f"# {passed} PASSED, {failed} FAILED")
    print("#" * 70)
    
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test register profiles")
    parser.add_argument("--wiki", help="Path to Wikipedia GloVe embeddings")
    parser.add_argument("--twitter", help="Path to Twitter GloVe embeddings")
    args = parser.parse_args()
    
    success = run_all_tests(args.wiki, args.twitter)
    sys.exit(0 if success else 1)
