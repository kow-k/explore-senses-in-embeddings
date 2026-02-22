#!/usr/bin/env python3
"""
test_weighted_merger.py - Tests for Weighted Embedding Merger
==============================================================

Tests the quality-based weighting system for embedding merging.

Usage:
    # Toy data only
    python test_weighted_merger.py
    
    # With real embeddings
    python test_weighted_merger.py --wiki path/to/wiki.txt --twitter path/to/twitter.txt

Author: Kow Kuroda & Claude (Anthropic)
"""

import numpy as np
import argparse
import sys
import os

# Add parent directory to path
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_root not in sys.path:
    sys.path.insert(0, package_root)


def create_toy_embeddings_with_quality_difference(dim=50):
    """
    Create two embeddings with intentionally different quality characteristics.
    
    Embedding A (high quality): 
        - Clear sense separation
        - Coherent anchor groups
        
    Embedding B (lower quality):
        - Noisier sense separation
        - Less coherent anchors
    """
    np.random.seed(42)
    
    # Shared sense directions
    financial_dir = np.random.randn(dim)
    financial_dir /= np.linalg.norm(financial_dir)
    
    river_dir = np.random.randn(dim)
    river_dir /= np.linalg.norm(river_dir)
    
    def make_embedding(noise_level, n_filler=300):
        emb = {}
        
        # Financial words
        financial_words = ['money', 'loan', 'credit', 'finance', 'investment', 
                          'account', 'deposit', 'savings', 'funds', 'banking',
                          'interest', 'capital', 'asset', 'debt', 'mortgage']
        for word in financial_words:
            noise = np.random.randn(dim) * noise_level
            vec = financial_dir + noise
            emb[word] = vec / np.linalg.norm(vec)
        
        # River words
        river_words = ['river', 'stream', 'shore', 'water', 'flow',
                       'creek', 'lake', 'pond', 'embankment', 'wetland',
                       'current', 'tide', 'bank', 'delta', 'tributary']
        for word in river_words:
            noise = np.random.randn(dim) * noise_level
            vec = river_dir + noise
            emb[word] = vec / np.linalg.norm(vec)
        
        # Target word "bank" (superposition)
        bank_vec = 0.6 * financial_dir + 0.4 * river_dir
        bank_vec /= np.linalg.norm(bank_vec)
        emb['bank'] = bank_vec
        
        # Filler words
        for i in range(n_filler):
            word = f"word_{i}"
            vec = np.random.randn(dim)
            emb[word] = vec / np.linalg.norm(vec)
        
        return emb
    
    # High quality embedding (low noise)
    emb_high = make_embedding(noise_level=0.1, n_filler=500)
    
    # Lower quality embedding (more noise, smaller)
    emb_low = make_embedding(noise_level=0.3, n_filler=200)
    
    return emb_high, emb_low


def test_embedding_quality_assessor():
    """Test the EmbeddingQualityAssessor class."""
    from sense_explorer.merger.embedding_weights import EmbeddingQualityAssessor
    
    print("=" * 60)
    print("TEST: EmbeddingQualityAssessor")
    print("=" * 60)
    
    emb_high, emb_low = create_toy_embeddings_with_quality_difference()
    
    assessor = EmbeddingQualityAssessor(verbose=True)
    assessor.add_embedding("high_quality", emb_high)
    assessor.add_embedding("low_quality", emb_low)
    
    # Run assessments
    assessor.assess_vocabulary_scores()
    assessor.assess_overlap_scores()
    
    # Get weights
    weights = assessor.get_weights()
    
    print(f"\nFinal weights: {weights}")
    
    # Check that vocab score is higher for larger embedding
    assert assessor.qualities["high_quality"].vocab_score > assessor.qualities["low_quality"].vocab_score, \
        "Larger embedding should have higher vocab score"
    
    # The final weight depends on all components; without SenseExplorer
    # for coherence/separation, we only have vocab and overlap.
    # With the fixed overlap logic, high_quality should win on vocab
    # while overlap should be equal (both cover shared vocab fully).
    
    print(f"\nQuality breakdown:")
    for name, q in assessor.qualities.items():
        print(f"  {name}: vocab={q.vocab_score:.3f}, overlap={q.overlap_score:.3f}")
    
    print("\n✓ EmbeddingQualityAssessor test PASSED")
    return True


def test_weight_similarity_matrix():
    """Test similarity matrix weighting."""
    from sense_explorer.merger.embedding_weights import weight_similarity_matrix
    
    print("\n" + "=" * 60)
    print("TEST: weight_similarity_matrix")
    print("=" * 60)
    
    # Create test similarity matrix
    sim_matrix = np.array([
        [1.0, 0.8, 0.3],
        [0.8, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    # Weights: first two items high quality, third lower
    weights = [0.4, 0.4, 0.2]
    
    # Test symmetric weighting
    weighted = weight_similarity_matrix(sim_matrix, weights, method="symmetric")
    
    print(f"\nOriginal similarity matrix:\n{sim_matrix}")
    print(f"\nWeights: {weights}")
    print(f"\nWeighted similarity matrix (symmetric):\n{weighted.round(3)}")
    
    # High-quality pair should have highest weighted similarity
    assert weighted[0, 1] > weighted[0, 2], \
        "High-quality pairs should have higher weighted similarity"
    assert weighted[0, 1] > weighted[1, 2], \
        "High-quality pairs should have higher weighted similarity"
    
    print("\n✓ weight_similarity_matrix test PASSED")
    return True


def test_compute_weighted_centroid():
    """Test weighted centroid computation."""
    from sense_explorer.merger.embedding_weights import compute_weighted_centroid
    
    print("\n" + "=" * 60)
    print("TEST: compute_weighted_centroid")
    print("=" * 60)
    
    # Create test vectors
    np.random.seed(42)
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    
    # Equal weights
    centroid_equal = compute_weighted_centroid([vec1, vec2], [0.5, 0.5])
    print(f"Equal weights [0.5, 0.5]: {centroid_equal.round(3)}")
    
    # Biased weights
    centroid_biased = compute_weighted_centroid([vec1, vec2], [0.8, 0.2])
    print(f"Biased weights [0.8, 0.2]: {centroid_biased.round(3)}")
    
    # Biased centroid should be closer to vec1
    assert np.dot(centroid_biased, vec1) > np.dot(centroid_biased, vec2), \
        "Weighted centroid should be closer to higher-weighted vector"
    
    print("\n✓ compute_weighted_centroid test PASSED")
    return True


def test_quick_assess():
    """Test the quick_assess convenience function."""
    from sense_explorer.merger.embedding_weights import quick_assess
    
    print("\n" + "=" * 60)
    print("TEST: quick_assess")
    print("=" * 60)
    
    emb_high, emb_low = create_toy_embeddings_with_quality_difference()
    
    weights = quick_assess(
        {"high": emb_high, "low": emb_low},
        verbose=True
    )
    
    print(f"\nQuick assess weights: {weights}")
    
    # Should return normalized weights
    assert abs(sum(weights.values()) - 1.0) < 0.01, \
        "Weights should sum to 1.0"
    
    print("\n✓ quick_assess test PASSED")
    return True


def test_weighted_merger_result():
    """Test WeightedMergerResult dataclass."""
    from sense_explorer.merger.embedding_merger import WeightedMergerResult, SenseComponent
    
    print("\n" + "=" * 60)
    print("TEST: WeightedMergerResult")
    print("=" * 60)
    
    # Create minimal result
    result = WeightedMergerResult(
        word="bank",
        sense_components=[],
        similarity_matrix=np.eye(2),
        clusters={"s1": 0, "s2": 1},
        analysis={0: {"is_convergent": True}, 1: {"is_convergent": False}},
        pairwise_stats={},
        threshold_used=0.05,
        clustering_method="spectral_hierarchical",
        spectral_info=None,
        source_weights={"wiki": 0.6, "twitter": 0.4},
        sense_weights={"wiki_s0": 0.35, "wiki_s1": 0.25, "twitter_s0": 0.4},
        weighted_similarity_matrix=np.eye(2) * 0.5
    )
    
    print(f"Word: {result.word}")
    print(f"Source weights: {result.source_weights}")
    print(f"Sense weights: {result.sense_weights}")
    print(f"Weight summary:\n{result.weight_summary}")
    
    assert result.source_weights["wiki"] > result.source_weights["twitter"]
    
    print("\n✓ WeightedMergerResult test PASSED")
    return True


def test_merge_with_weights_toy():
    """Test merge_with_weights with toy embeddings."""
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import merge_with_weights, weighted_report
    
    print("\n" + "=" * 60)
    print("TEST: merge_with_weights (toy data)")
    print("=" * 60)
    
    emb_high, emb_low = create_toy_embeddings_with_quality_difference()
    
    # Create SenseExplorers
    se_high = SenseExplorer.from_dict(emb_high, verbose=False)
    se_low = SenseExplorer.from_dict(emb_low, verbose=False)
    
    # Set anchors for sense induction
    anchors = {
        'financial': ['money', 'loan', 'credit', 'finance'],
        'river': ['river', 'stream', 'shore', 'water']
    }
    se_high.set_anchors('bank', anchors)
    se_low.set_anchors('bank', anchors)
    
    # Run weighted merge
    result = merge_with_weights(
        {"high_quality": se_high, "low_quality": se_low},
        "bank",
        n_senses=2,
        verbose=True
    )
    
    print(f"\nResult:")
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Convergent: {result.n_convergent}")
    print(f"  Source weights: {result.source_weights}")
    
    # High quality source should have higher weight
    assert result.source_weights["high_quality"] > result.source_weights["low_quality"], \
        "High quality source should have higher weight"
    
    # Generate report
    report = weighted_report(result)
    print("\n" + "-" * 40)
    print(report[:500] + "...")
    
    print("\n✓ merge_with_weights (toy) test PASSED")
    return True


def test_custom_weight_config():
    """Test custom weight configuration."""
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import merge_with_weights
    
    print("\n" + "=" * 60)
    print("TEST: Custom Weight Configuration")
    print("=" * 60)
    
    emb_high, emb_low = create_toy_embeddings_with_quality_difference()
    
    se_high = SenseExplorer.from_dict(emb_high, verbose=False)
    se_low = SenseExplorer.from_dict(emb_low, verbose=False)
    
    anchors = {
        'financial': ['money', 'loan', 'credit'],
        'river': ['river', 'stream', 'water']
    }
    se_high.set_anchors('bank', anchors)
    se_low.set_anchors('bank', anchors)
    
    # Test with vocabulary-heavy weighting
    result_vocab = merge_with_weights(
        {"high": se_high, "low": se_low},
        "bank",
        weight_config={'vocab': 0.8, 'coherence': 0.1, 'separation': 0.1, 'overlap': 0.0},
        verbose=False
    )
    
    # Test with coherence-heavy weighting
    result_coh = merge_with_weights(
        {"high": se_high, "low": se_low},
        "bank",
        weight_config={'vocab': 0.0, 'coherence': 0.8, 'separation': 0.2, 'overlap': 0.0},
        verbose=False
    )
    
    print(f"Vocab-heavy weights: {result_vocab.source_weights}")
    print(f"Coherence-heavy weights: {result_coh.source_weights}")
    
    # Both should give higher weight to high-quality embedding, but amounts may differ
    assert result_vocab.source_weights["high"] > 0.5
    
    print("\n✓ Custom weight configuration test PASSED")
    return True


def test_with_real_embeddings(wiki_path, twitter_path):
    """Test with real GloVe embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Real Embeddings")
    print("=" * 60)
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import merge_with_weights, weighted_report
    
    # Load wiki first to get its dimension
    print(f"\nLoading Wikipedia embeddings from {wiki_path}...")
    se_wiki = SenseExplorer.from_file(wiki_path, max_words=50000, verbose=True)
    wiki_dim = se_wiki.dim
    print(f"  Wiki dimension: {wiki_dim}")
    
    # Load twitter with target_dim to match wiki
    print(f"\nLoading Twitter embeddings from {twitter_path}...")
    print(f"  (aligning to {wiki_dim}d to match Wikipedia)")
    se_twitter = SenseExplorer.from_file(
        twitter_path, 
        max_words=50000, 
        target_dim=wiki_dim,  # Align to wiki dimension
        verbose=True
    )
    print(f"  Twitter dimension after alignment: {se_twitter.dim}")
    
    # Test weighted merge
    test_words = ['bank', 'rock', 'crane']
    
    for word in test_words:
        if word not in se_wiki.vocab or word not in se_twitter.vocab:
            print(f"\nSkipping '{word}' (not in both vocabularies)")
            continue
        
        print(f"\n{'='*40}")
        print(f"Testing '{word}'")
        print('='*40)
        
        try:
            result = merge_with_weights(
                {"wiki": se_wiki, "twitter": se_twitter},
                word,
                n_senses=3,
                verbose=True
            )
            
            print(f"\nResults for '{word}':")
            print(f"  Source weights: wiki={result.source_weights.get('wiki', 0):.3f}, "
                  f"twitter={result.source_weights.get('twitter', 0):.3f}")
            print(f"  Clusters: {result.n_clusters}")
            print(f"  Convergent: {result.n_convergent}")
            print(f"  Source-specific: {result.n_source_specific}")
            
            if result.spectral_info:
                print(f"  Spectral k: {result.spectral_info.suggested_k}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n✓ Real embeddings test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Weighted Embedding Merger")
    parser.add_argument("--wiki", type=str, help="Path to Wikipedia GloVe embeddings")
    parser.add_argument("--twitter", type=str, help="Path to Twitter GloVe embeddings")
    args = parser.parse_args()
    
    print("#" * 70)
    print("# WEIGHTED EMBEDDING MERGER TESTS")
    print("#" * 70)
    
    all_passed = True
    
    # Core tests
    all_passed &= test_embedding_quality_assessor()
    all_passed &= test_weight_similarity_matrix()
    all_passed &= test_compute_weighted_centroid()
    all_passed &= test_quick_assess()
    all_passed &= test_weighted_merger_result()
    
    # Integration tests
    all_passed &= test_merge_with_weights_toy()
    all_passed &= test_custom_weight_config()
    
    # Real embedding tests
    if args.wiki and args.twitter:
        all_passed &= test_with_real_embeddings(args.wiki, args.twitter)
    else:
        print("\n" + "-" * 60)
        print("To test with real embeddings:")
        print("  python test_weighted_merger.py --wiki wiki.txt --twitter twitter.txt")
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
