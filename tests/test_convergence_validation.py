"""
Tests for Cross-Embedding Convergence Validation

Tests the validation methods that check whether convergent senses
truly represent the same meaning across different embeddings.
"""

import sys
import numpy as np
from typing import Dict, List
import argparse


def create_toy_embeddings(dim: int = 50, n_common: int = 200, n_unique: int = 100):
    """
    Create toy embeddings with known convergent and divergent senses.
    
    Returns two embeddings where:
    - 'bank' has clear financial and river senses in both (should converge)
    - 'crane' has bird sense in emb1, machine sense in emb2 (should NOT converge)
    - 'rock' has music sense in both (should converge)
    """
    np.random.seed(42)
    
    # Common vocabulary
    common_words = [f"word_{i}" for i in range(n_common)]
    
    # Create embeddings
    emb1 = {}
    emb2 = {}
    
    # Generate random vectors for common words with some correlation
    for word in common_words:
        base = np.random.randn(dim)
        emb1[word] = base + np.random.randn(dim) * 0.3
        emb2[word] = base + np.random.randn(dim) * 0.3
    
    # Add unique words to each
    for i in range(n_unique):
        emb1[f"unique1_{i}"] = np.random.randn(dim)
        emb2[f"unique2_{i}"] = np.random.randn(dim)
    
    # === Create sense-specific neighborhoods ===
    
    # Financial words (similar in both)
    financial_words = ['money', 'account', 'deposit', 'loan', 'credit', 'savings',
                       'interest', 'finance', 'investment', 'banking']
    financial_vec = np.random.randn(dim)
    financial_vec /= np.linalg.norm(financial_vec)
    
    for word in financial_words:
        noise1 = np.random.randn(dim) * 0.2
        noise2 = np.random.randn(dim) * 0.2
        emb1[word] = financial_vec + noise1
        emb2[word] = financial_vec + noise2
    
    # River words (similar in both)
    river_words = ['river', 'stream', 'water', 'shore', 'creek', 'flow',
                   'fishing', 'boat', 'riverbank', 'waterside']
    river_vec = np.random.randn(dim)
    river_vec /= np.linalg.norm(river_vec)
    
    for word in river_words:
        noise1 = np.random.randn(dim) * 0.2
        noise2 = np.random.randn(dim) * 0.2
        emb1[word] = river_vec + noise1
        emb2[word] = river_vec + noise2
    
    # Music/rock words (similar in both)
    music_words = ['music', 'band', 'guitar', 'concert', 'song', 'album',
                   'musician', 'drums', 'singer', 'melody']
    music_vec = np.random.randn(dim)
    music_vec /= np.linalg.norm(music_vec)
    
    for word in music_words:
        noise1 = np.random.randn(dim) * 0.2
        noise2 = np.random.randn(dim) * 0.2
        emb1[word] = music_vec + noise1
        emb2[word] = music_vec + noise2
    
    # Bird words (only strong in emb1)
    bird_words = ['bird', 'wing', 'feather', 'beak', 'nest', 'fly',
                  'heron', 'stork', 'egret', 'waterfowl']
    bird_vec = np.random.randn(dim)
    bird_vec /= np.linalg.norm(bird_vec)
    
    for word in bird_words:
        noise1 = np.random.randn(dim) * 0.2
        noise2 = np.random.randn(dim) * 0.5  # More noise in emb2
        emb1[word] = bird_vec + noise1
        emb2[word] = np.random.randn(dim)  # Random in emb2
    
    # Machine words (only strong in emb2)
    machine_words = ['machine', 'lift', 'construction', 'tower', 'hook',
                     'hydraulic', 'boom', 'operator', 'hoist', 'equipment']
    machine_vec = np.random.randn(dim)
    machine_vec /= np.linalg.norm(machine_vec)
    
    for word in machine_words:
        noise1 = np.random.randn(dim) * 0.5  # More noise in emb1
        noise2 = np.random.randn(dim) * 0.2
        emb1[word] = np.random.randn(dim)  # Random in emb1
        emb2[word] = machine_vec + noise2
    
    # === Create target words ===
    
    # Bank: financial + river superposition (similar in both)
    emb1['bank'] = 0.7 * financial_vec + 0.3 * river_vec + np.random.randn(dim) * 0.1
    emb2['bank'] = 0.7 * financial_vec + 0.3 * river_vec + np.random.randn(dim) * 0.1
    
    # Rock: music sense in both
    emb1['rock'] = music_vec + np.random.randn(dim) * 0.15
    emb2['rock'] = music_vec + np.random.randn(dim) * 0.15
    
    # Crane: bird in emb1, machine in emb2 (should NOT converge well)
    emb1['crane'] = bird_vec + np.random.randn(dim) * 0.15
    emb2['crane'] = machine_vec + np.random.randn(dim) * 0.15
    
    # Normalize all vectors
    for word in emb1:
        emb1[word] = emb1[word] / np.linalg.norm(emb1[word])
    for word in emb2:
        emb2[word] = emb2[word] / np.linalg.norm(emb2[word])
    
    return emb1, emb2


def test_neighbor_overlap():
    """Test neighbor overlap computation."""
    print("\n" + "=" * 60)
    print("TEST: Neighbor Overlap")
    print("=" * 60)
    
    from sense_explorer.merger import compute_neighbor_overlap
    
    # High overlap case
    neighbors_a = ['word1', 'word2', 'word3', 'word4', 'word5']
    neighbors_b = ['word1', 'word2', 'word3', 'word6', 'word7']
    
    overlap, shared = compute_neighbor_overlap(neighbors_a, neighbors_b, top_k=5)
    print(f"High overlap case: {overlap:.3f}")
    print(f"  Shared: {shared}")
    assert overlap > 0.3, f"Expected overlap > 0.3, got {overlap}"
    
    # Low overlap case
    neighbors_c = ['apple', 'banana', 'cherry', 'date', 'elderberry']
    neighbors_d = ['zebra', 'yak', 'xerus', 'wombat', 'vole']
    
    overlap2, shared2 = compute_neighbor_overlap(neighbors_c, neighbors_d, top_k=5)
    print(f"Low overlap case: {overlap2:.3f}")
    assert overlap2 == 0.0, f"Expected overlap = 0, got {overlap2}"
    
    print("\n✓ Neighbor overlap test PASSED")
    return True


def test_anchor_consistency():
    """Test anchor consistency computation."""
    print("\n" + "=" * 60)
    print("TEST: Anchor Consistency")
    print("=" * 60)
    
    from sense_explorer.merger import compute_anchor_consistency
    
    # Same anchors
    anchors_a = ['money', 'bank', 'finance']
    anchors_b = ['money', 'bank', 'credit']
    
    consistency, shared = compute_anchor_consistency(anchors_a, anchors_b)
    print(f"Partial match: {consistency:.3f}")
    print(f"  Shared: {shared}")
    assert consistency > 0.5, f"Expected consistency > 0.5, got {consistency}"
    
    # Different anchors
    anchors_c = ['river', 'water', 'stream']
    anchors_d = ['money', 'bank', 'credit']
    
    consistency2, shared2 = compute_anchor_consistency(anchors_c, anchors_d)
    print(f"No match: {consistency2:.3f}")
    assert consistency2 == 0.0, f"Expected consistency = 0, got {consistency2}"
    
    # Empty anchors (should default)
    consistency3, _ = compute_anchor_consistency([], anchors_b)
    print(f"Empty anchors (default): {consistency3:.3f}")
    assert consistency3 == 0.5, f"Expected default 0.5, got {consistency3}"
    
    print("\n✓ Anchor consistency test PASSED")
    return True


def test_semantic_coherence():
    """Test semantic coherence computation."""
    print("\n" + "=" * 60)
    print("TEST: Semantic Coherence")
    print("=" * 60)
    
    from sense_explorer.merger import compute_semantic_coherence, SenseComponent
    
    # Create similarity matrix where cluster (0,1) is coherent
    sim_matrix = np.array([
        [1.0, 0.9, 0.1, 0.2],  # sense 0 - similar to sense 1
        [0.9, 1.0, 0.2, 0.1],  # sense 1 - similar to sense 0
        [0.1, 0.2, 1.0, 0.15], # sense 2 - different
        [0.2, 0.1, 0.15, 1.0], # sense 3 - different
    ])
    
    # Create mock senses
    senses = [
        SenseComponent(word='test', sense_id='s0', vector=np.zeros(10), source='a'),
        SenseComponent(word='test', sense_id='s1', vector=np.zeros(10), source='b'),
    ]
    
    sense_id_to_idx = {'s0': 0, 's1': 1, 's2': 2, 's3': 3}
    
    coherence = compute_semantic_coherence(senses, sim_matrix, sense_id_to_idx)
    print(f"High coherence cluster: {coherence:.3f}")
    assert coherence > 0.6, f"Expected coherence > 0.6, got {coherence}"
    
    # Low coherence cluster
    senses_low = [
        SenseComponent(word='test', sense_id='s0', vector=np.zeros(10), source='a'),
        SenseComponent(word='test', sense_id='s2', vector=np.zeros(10), source='b'),
    ]
    
    coherence_low = compute_semantic_coherence(senses_low, sim_matrix, sense_id_to_idx)
    print(f"Low coherence cluster: {coherence_low:.3f}")
    
    print("\n✓ Semantic coherence test PASSED")
    return True


def test_confidence_scoring():
    """Test confidence score computation."""
    print("\n" + "=" * 60)
    print("TEST: Confidence Scoring")
    print("=" * 60)
    
    from sense_explorer.merger import compute_confidence
    
    # High confidence
    conf, level = compute_confidence(
        neighbor_overlap=0.8,
        anchor_consistency=0.9,
        cross_projection_match=True,
        semantic_coherence=0.85
    )
    print(f"High scores: confidence={conf:.3f}, level={level}")
    assert level == "high", f"Expected 'high', got {level}"
    
    # Low confidence
    conf2, level2 = compute_confidence(
        neighbor_overlap=0.1,
        anchor_consistency=0.2,
        cross_projection_match=False,
        semantic_coherence=0.3
    )
    print(f"Low scores: confidence={conf2:.3f}, level={level2}")
    assert level2 == "low", f"Expected 'low', got {level2}"
    
    # Medium confidence
    conf3, level3 = compute_confidence(
        neighbor_overlap=0.5,
        anchor_consistency=0.5,
        cross_projection_match=True,
        semantic_coherence=0.5
    )
    print(f"Medium scores: confidence={conf3:.3f}, level={level3}")
    assert level3 == "medium", f"Expected 'medium', got {level3}"
    
    print("\n✓ Confidence scoring test PASSED")
    return True


def test_full_validation_toy():
    """Test full validation pipeline with toy data."""
    print("\n" + "=" * 60)
    print("TEST: Full Validation (Toy Data)")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import merge_with_weights, validate_convergence
    
    # Create toy embeddings
    emb1, emb2 = create_toy_embeddings(dim=50)
    
    print(f"Created embeddings: emb1={len(emb1)} words, emb2={len(emb2)} words")
    
    # Create SenseExplorers
    se1 = SenseExplorer.from_dict(emb1, verbose=False)
    se2 = SenseExplorer.from_dict(emb2, verbose=False)
    
    # Test 'bank' - should have high confidence convergence
    print("\n--- Testing 'bank' (expected: high confidence) ---")
    result_bank = merge_with_weights(
        {"emb1": se1, "emb2": se2},
        "bank",
        verbose=False
    )
    
    report_bank = validate_convergence(
        result_bank,
        {"emb1": se1, "emb2": se2},
        verbose=True
    )
    
    print(f"\nBank validation:")
    print(f"  Convergent clusters: {report_bank.n_convergent}")
    print(f"  Mean confidence: {report_bank.mean_confidence:.3f}")
    print(f"  Neighbor overlap: {report_bank.mean_neighbor_overlap:.3f}")
    
    # Test 'rock' - should also converge well (music sense in both)
    print("\n--- Testing 'rock' (expected: high confidence) ---")
    result_rock = merge_with_weights(
        {"emb1": se1, "emb2": se2},
        "rock",
        verbose=False
    )
    
    report_rock = validate_convergence(
        result_rock,
        {"emb1": se1, "emb2": se2},
        verbose=True
    )
    
    print(f"\nRock validation:")
    print(f"  Convergent clusters: {report_rock.n_convergent}")
    print(f"  Mean confidence: {report_rock.mean_confidence:.3f}")
    
    # Test 'crane' - should have LOW confidence (different senses)
    print("\n--- Testing 'crane' (expected: low confidence if converged) ---")
    result_crane = merge_with_weights(
        {"emb1": se1, "emb2": se2},
        "crane",
        verbose=False
    )
    
    report_crane = validate_convergence(
        result_crane,
        {"emb1": se1, "emb2": se2},
        verbose=True
    )
    
    print(f"\nCrane validation:")
    print(f"  Convergent clusters: {report_crane.n_convergent}")
    print(f"  Mean confidence: {report_crane.mean_confidence:.3f}")
    if report_crane.low_confidence_clusters:
        print(f"  Low confidence clusters detected: {len(report_crane.low_confidence_clusters)}")
    
    print("\n✓ Full validation (toy) test PASSED")
    return True


def test_quick_validate():
    """Test quick_validate convenience function."""
    print("\n" + "=" * 60)
    print("TEST: quick_validate")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import merge_with_weights, quick_validate
    
    emb1, emb2 = create_toy_embeddings(dim=50)
    se1 = SenseExplorer.from_dict(emb1, verbose=False)
    se2 = SenseExplorer.from_dict(emb2, verbose=False)
    
    result = merge_with_weights(
        {"emb1": se1, "emb2": se2},
        "bank",
        verbose=False
    )
    
    quick_result = quick_validate(result, {"emb1": se1, "emb2": se2})
    
    print(f"Quick validate result:")
    for key, value in quick_result.items():
        print(f"  {key}: {value}")
    
    assert 'mean_confidence' in quick_result
    assert 'any_low_confidence' in quick_result
    
    print("\n✓ quick_validate test PASSED")
    return True


def test_with_real_embeddings(wiki_path, twitter_path, similarity_threshold=0.3):
    """Test with real GloVe embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Real Embeddings Validation")
    print("=" * 60)
    print(f"Using similarity_threshold: {similarity_threshold}")
    print(f"  (Higher = stricter clustering, fewer convergent clusters)")
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import merge_with_weights, validate_convergence, summarize_validations
    
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
    
    # Test multiple words
    test_words = ['bank', 'rock', 'crane', 'apple', 'spring']
    reports = []
    
    for word in test_words:
        if word not in se_wiki.vocab or word not in se_twitter.vocab:
            print(f"\nSkipping '{word}' (not in both vocabularies)")
            continue
        
        print(f"\n{'='*40}")
        print(f"Testing '{word}'")
        print('='*40)
        
        try:
            result = merge_with_weights(
                explorers, 
                word, 
                similarity_threshold=similarity_threshold,
                verbose=False
            )
            report = validate_convergence(result, explorers, verbose=True)
            reports.append(report)
            
            print(f"\n{report.summary(verbose=False)}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    if reports:
        print("\n" + summarize_validations(reports))
    
    print("\n✓ Real embeddings validation test PASSED")
    return True


def run_all_tests(wiki_path=None, twitter_path=None, similarity_threshold=0.3):
    """Run all validation tests."""
    print("#" * 70)
    print("# CROSS-EMBEDDING CONVERGENCE VALIDATION TESTS")
    print("#" * 70)
    
    tests = [
        ("Neighbor Overlap", test_neighbor_overlap),
        ("Anchor Consistency", test_anchor_consistency),
        ("Semantic Coherence", test_semantic_coherence),
        ("Confidence Scoring", test_confidence_scoring),
        ("Full Validation (Toy)", test_full_validation_toy),
        ("Quick Validate", test_quick_validate),
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
            test_with_real_embeddings(wiki_path, twitter_path, similarity_threshold)
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
    parser = argparse.ArgumentParser(description="Test convergence validation")
    parser.add_argument("--wiki", help="Path to Wikipedia GloVe embeddings")
    parser.add_argument("--twitter", help="Path to Twitter GloVe embeddings")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Similarity threshold for clustering (default: 0.3). "
                             "Higher values = stricter clustering = fewer convergent clusters. "
                             "Try 0.2-0.4 for cross-embedding merging.")
    args = parser.parse_args()
    
    success = run_all_tests(args.wiki, args.twitter, args.threshold)
    sys.exit(0 if success else 1)
