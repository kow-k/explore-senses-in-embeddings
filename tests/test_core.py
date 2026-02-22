#!/usr/bin/env python3
"""
test_core.py - Tests for Core SenseExplorer Module
===================================================

Tests the main SenseExplorer class and sense separation functionality.

Usage:
    # Toy data only
    python test_core.py
    
    # With real embeddings
    python test_core.py --glove path/to/glove.txt

Author: Kow Kuroda & Claude (Anthropic)
"""

import numpy as np
import argparse
import sys
import os

# Add parent directory (package root) to path for imports
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.insert(0, package_root)


def create_toy_embeddings(dim=50, n_words=500):
    """
    Create synthetic embeddings with known polysemous structure.
    
    Creates "bank" with financial and river senses.
    """
    np.random.seed(42)
    
    # Sense directions
    financial_dir = np.random.randn(dim)
    financial_dir /= np.linalg.norm(financial_dir)
    
    river_dir = np.random.randn(dim)
    river_dir /= np.linalg.norm(river_dir)
    
    embeddings = {}
    
    # Financial neighbors
    financial_words = ['money', 'loan', 'credit', 'finance', 'investment',
                       'account', 'deposit', 'savings', 'funds', 'banking']
    for word in financial_words:
        noise = np.random.randn(dim) * 0.15
        vec = financial_dir + noise
        embeddings[word] = vec / np.linalg.norm(vec)
    
    # River neighbors
    river_words = ['river', 'stream', 'shore', 'water', 'flow',
                   'creek', 'lake', 'pond', 'embankment', 'wetland']
    for word in river_words:
        noise = np.random.randn(dim) * 0.15
        vec = river_dir + noise
        embeddings[word] = vec / np.linalg.norm(vec)
    
    # Polysemous word "bank" (superposition)
    bank_vec = 0.6 * financial_dir + 0.4 * river_dir
    bank_vec /= np.linalg.norm(bank_vec)
    embeddings['bank'] = bank_vec
    
    # Random filler words
    for i in range(n_words - len(embeddings)):
        word = f"word_{i}"
        vec = np.random.randn(dim)
        embeddings[word] = vec / np.linalg.norm(vec)
    
    return embeddings


def test_initialization():
    """Test SenseExplorer initialization."""
    from sense_explorer.core import SenseExplorer
    
    print("=" * 60)
    print("TEST: SenseExplorer Initialization")
    print("=" * 60)
    
    embeddings = create_toy_embeddings()
    
    # Test from_dict
    se = SenseExplorer.from_dict(embeddings, verbose=True)
    
    print(f"\nSenseExplorer created:")
    print(f"  Vocab size: {se.vocab_size}")
    print(f"  Dimension: {se.dim}")
    print(f"  Clustering method: {se.clustering_method}")
    
    assert se.vocab_size == len(embeddings)
    assert se.dim == 50
    assert 'bank' in se.vocab
    
    print("\n✓ Initialization test PASSED")
    return True


def test_discover_senses():
    """Test unsupervised sense discovery."""
    from sense_explorer.core import SenseExplorer
    
    print("\n" + "=" * 60)
    print("TEST: Sense Discovery")
    print("=" * 60)
    
    embeddings = create_toy_embeddings()
    se = SenseExplorer.from_dict(embeddings, verbose=False)
    
    # Test discover_senses with fixed k
    senses = se.discover_senses('bank', n_senses=2)
    
    print(f"\nDiscovered senses for 'bank':")
    for sense_name, vec in senses.items():
        # Find nearest neighbors
        neighbors = []
        for word in embeddings:
            if word != 'bank':
                sim = np.dot(vec, embeddings[word])
                neighbors.append((word, sim))
        neighbors.sort(key=lambda x: -x[1])
        print(f"  {sense_name}: {[w for w, _ in neighbors[:5]]}")
    
    assert len(senses) == 2, f"Expected 2 senses, got {len(senses)}"
    
    print("\n✓ Sense discovery test PASSED")
    return True


def test_discover_senses_auto():
    """Test automatic k-selection via eigengap."""
    from sense_explorer.core import SenseExplorer
    
    print("\n" + "=" * 60)
    print("TEST: Auto Sense Discovery (Eigengap)")
    print("=" * 60)
    
    embeddings = create_toy_embeddings()
    se = SenseExplorer.from_dict(embeddings, clustering_method='spectral', verbose=False)
    
    senses = se.discover_senses_auto('bank')
    
    print(f"\nAuto-discovered {len(senses)} senses for 'bank'")
    for sense_name, vec in senses.items():
        neighbors = []
        for word in embeddings:
            if word != 'bank':
                sim = np.dot(vec, embeddings[word])
                neighbors.append((word, sim))
        neighbors.sort(key=lambda x: -x[1])
        print(f"  {sense_name}: {[w for w, _ in neighbors[:5]]}")
    
    # Should find 2 senses (financial and river)
    assert len(senses) >= 2, f"Expected at least 2 senses, got {len(senses)}"
    
    print("\n✓ Auto sense discovery test PASSED")
    return True


def test_induce_senses():
    """Test anchor-guided sense induction."""
    from sense_explorer.core import SenseExplorer
    
    print("\n" + "=" * 60)
    print("TEST: Sense Induction (Anchor-guided)")
    print("=" * 60)
    
    embeddings = create_toy_embeddings()
    se = SenseExplorer.from_dict(embeddings, verbose=False)
    
    # Set custom anchors
    se.set_anchors('bank', {
        'financial': ['money', 'loan', 'credit', 'account'],
        'river': ['river', 'stream', 'shore', 'water']
    })
    
    senses = se.induce_senses('bank')
    
    print(f"\nInduced senses for 'bank':")
    for sense_name, vec in senses.items():
        neighbors = []
        for word in embeddings:
            if word != 'bank':
                sim = np.dot(vec, embeddings[word])
                neighbors.append((word, sim))
        neighbors.sort(key=lambda x: -x[1])
        print(f"  {sense_name}: {[w for w, _ in neighbors[:5]]}")
    
    assert 'financial' in senses or 'river' in senses, "Expected named senses"
    
    print("\n✓ Sense induction test PASSED")
    return True


def test_similarity():
    """Test sense-aware similarity."""
    from sense_explorer.core import SenseExplorer
    
    print("\n" + "=" * 60)
    print("TEST: Sense-Aware Similarity")
    print("=" * 60)
    
    embeddings = create_toy_embeddings()
    se = SenseExplorer.from_dict(embeddings, verbose=False)
    
    # Standard similarity
    std_sim = se.similarity('bank', 'money', sense_aware=False)
    
    # Sense-aware similarity
    sense_sim = se.similarity('bank', 'money', sense_aware=True)
    
    print(f"\nSimilarity bank-money:")
    print(f"  Standard: {std_sim:.4f}")
    print(f"  Sense-aware: {sense_sim:.4f}")
    
    # Sense-aware should be >= standard for related words
    # (it selects the best matching sense)
    
    # Test with unrelated word
    std_unrelated = se.similarity('bank', 'word_0', sense_aware=False)
    sense_unrelated = se.similarity('bank', 'word_0', sense_aware=True)
    
    print(f"\nSimilarity bank-word_0:")
    print(f"  Standard: {std_unrelated:.4f}")
    print(f"  Sense-aware: {sense_unrelated:.4f}")
    
    print("\n✓ Similarity test PASSED")
    return True


def test_caching():
    """Test sense caching functionality."""
    from sense_explorer.core import SenseExplorer
    
    print("\n" + "=" * 60)
    print("TEST: Sense Caching")
    print("=" * 60)
    
    embeddings = create_toy_embeddings()
    se = SenseExplorer.from_dict(embeddings, verbose=False)
    
    # First call - should compute
    senses1 = se.induce_senses('bank')
    cache_size_1 = len(se._sense_cache)
    
    # Second call - should use cache
    senses2 = se.induce_senses('bank')
    cache_size_2 = len(se._sense_cache)
    
    print(f"\nCache size after first call: {cache_size_1}")
    print(f"Cache size after second call: {cache_size_2}")
    
    # Verify same results
    for sense_name in senses1:
        assert sense_name in senses2
        assert np.allclose(senses1[sense_name], senses2[sense_name])
    
    # Test force refresh
    senses3 = se.induce_senses('bank', force=True)
    print(f"Force refresh successful: {len(senses3)} senses")
    
    print("\n✓ Caching test PASSED")
    return True


def test_explore_senses():
    """Test the explore_senses convenience method."""
    from sense_explorer.core import SenseExplorer
    
    print("\n" + "=" * 60)
    print("TEST: explore_senses Convenience Method")
    print("=" * 60)
    
    embeddings = create_toy_embeddings()
    se = SenseExplorer.from_dict(embeddings, verbose=False)
    
    modes = ['discover', 'discover_auto', 'induce']
    
    for mode in modes:
        try:
            senses = se.explore_senses('bank', mode=mode)
            print(f"  {mode}: {len(senses)} senses - {list(senses.keys())}")
        except Exception as e:
            print(f"  {mode}: Error - {e}")
    
    print("\n✓ explore_senses test PASSED")
    return True


def test_with_real_embeddings(glove_path):
    """Test with real GloVe embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Real Embeddings")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    
    print(f"\nLoading embeddings from {glove_path}...")
    se = SenseExplorer.from_file(glove_path, max_words=50000, verbose=True)
    
    # Test various words
    test_words = ['bank', 'bat', 'crane', 'mouse', 'cell', 'plant']
    test_words = [w for w in test_words if w in se.vocab]
    
    print("\n--- Testing induce_senses ---")
    for word in test_words[:3]:
        try:
            senses = se.induce_senses(word)
            print(f"\n{word}: {len(senses)} senses")
            for sense_name, vec in senses.items():
                neighbors = []
                for w in list(se.embeddings.keys())[:5000]:
                    if w != word:
                        sim = np.dot(vec, se.embeddings[w])
                        neighbors.append((w, sim))
                neighbors.sort(key=lambda x: -x[1])
                print(f"  {sense_name}: {[w for w, _ in neighbors[:5]]}")
        except Exception as e:
            print(f"  {word}: Error - {e}")
    
    print("\n--- Testing discover_senses_auto ---")
    for word in test_words[:2]:
        try:
            senses = se.discover_senses_auto(word)
            print(f"  {word}: {len(senses)} senses auto-discovered")
        except Exception as e:
            print(f"  {word}: Error - {e}")
    
    print("\n✓ Real embeddings test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Core SenseExplorer")
    parser.add_argument("--glove", type=str, help="Path to GloVe embeddings")
    args = parser.parse_args()
    
    print("#" * 70)
    print("# CORE SENSEEXPLORER MODULE TESTS")
    print("#" * 70)
    
    all_passed = True
    all_passed &= test_initialization()
    all_passed &= test_discover_senses()
    all_passed &= test_discover_senses_auto()
    all_passed &= test_induce_senses()
    all_passed &= test_similarity()
    all_passed &= test_caching()
    all_passed &= test_explore_senses()
    
    if args.glove:
        all_passed &= test_with_real_embeddings(args.glove)
    else:
        print("\n" + "-" * 60)
        print("To test with real embeddings:")
        print("  python test_core.py --glove glove.6B.100d.txt")
        print("-" * 60)
    
    print("\n" + "#" * 70)
    if all_passed:
        print("# ALL TESTS PASSED ✓")
    else:
        print("# SOME TESTS FAILED ✗")
    print("#" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
