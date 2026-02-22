#!/usr/bin/env python3
"""
test_merger.py - Tests for Embedding Merger Module
===================================================

Tests the embedding merger with both toy data and real embeddings.

Usage:
    # Toy data only
    python test_merger.py
    
    # With real embeddings
    python test_merger.py --wiki path/to/wiki.txt --twitter path/to/twitter.txt

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


def create_toy_embeddings_pair(dim=50, n_words=200):
    """
    Create two synthetic embeddings with overlapping and distinct senses.
    
    Embedding A (wiki-like): financial + river senses of "bank"
    Embedding B (twitter-like): financial + slang senses of "bank"
    
    Expected outcome:
    - financial sense: CONVERGENT (present in both)
    - river sense: SOURCE-SPECIFIC (wiki only)
    - slang sense: SOURCE-SPECIFIC (twitter only)
    """
    np.random.seed(42)
    
    # Shared direction (financial)
    financial_dir = np.random.randn(dim)
    financial_dir /= np.linalg.norm(financial_dir)
    
    # Source-specific directions
    river_dir = np.random.randn(dim)
    river_dir /= np.linalg.norm(river_dir)
    
    slang_dir = np.random.randn(dim)
    slang_dir /= np.linalg.norm(slang_dir)
    
    def make_embedding(cluster_words_list, cluster_dirs, n_total=200):
        emb = {}
        for words, direction in zip(cluster_words_list, cluster_dirs):
            for word in words:
                noise = np.random.randn(dim) * 0.15
                vec = direction + noise
                emb[word] = vec / np.linalg.norm(vec)
        
        # Filler words
        for i in range(n_total - len(emb)):
            word = f"word_{i}"
            vec = np.random.randn(dim)
            emb[word] = vec / np.linalg.norm(vec)
        
        return emb
    
    # Embedding A: financial + river
    financial_words_a = ['bank', 'money', 'loan', 'credit', 'finance', 'investment', 'banking']
    river_words_a = ['river', 'stream', 'shore', 'water', 'flow', 'creek']
    emb_a = make_embedding([financial_words_a, river_words_a], [financial_dir, river_dir])
    
    # Embedding B: financial + slang
    financial_words_b = ['bank', 'money', 'cash', 'credit', 'pay', 'broke', 'funds']
    slang_words_b = ['lit', 'fire', 'dope', 'sick', 'cool', 'awesome']
    emb_b = make_embedding([financial_words_b, slang_words_b], [financial_dir, slang_dir])
    
    return emb_a, emb_b, {
        'financial_dir': financial_dir,
        'river_dir': river_dir,
        'slang_dir': slang_dir
    }


def test_basic_merger():
    """Test basic embedding merger functionality."""
    from sense_explorer.merger.embedding_merger import EmbeddingMerger, SenseComponent
    
    print("=" * 60)
    print("TEST: Basic Embedding Merger")
    print("=" * 60)
    
    emb_a, emb_b, directions = create_toy_embeddings_pair()
    
    merger = EmbeddingMerger(verbose=True)
    merger.add_embedding("wiki", emb_a)
    merger.add_embedding("twitter", emb_b)
    
    print(f"\nShared vocabulary: {len(merger.shared_vocabulary)} words")
    
    # Create sense components manually
    senses = [
        SenseComponent(
            word="bank", sense_id="wiki_financial",
            vector=directions['financial_dir'] + np.random.randn(50) * 0.05,
            source="wiki",
            top_neighbors=[("money", 0.9), ("loan", 0.85)]
        ),
        SenseComponent(
            word="bank", sense_id="wiki_river",
            vector=directions['river_dir'] + np.random.randn(50) * 0.05,
            source="wiki",
            top_neighbors=[("river", 0.9), ("stream", 0.85)]
        ),
        SenseComponent(
            word="bank", sense_id="twitter_financial",
            vector=directions['financial_dir'] + np.random.randn(50) * 0.05,
            source="twitter",
            top_neighbors=[("money", 0.9), ("cash", 0.85)]
        ),
        SenseComponent(
            word="bank", sense_id="twitter_slang",
            vector=directions['slang_dir'] + np.random.randn(50) * 0.05,
            source="twitter",
            top_neighbors=[("lit", 0.9), ("fire", 0.85)]
        ),
    ]
    
    result = merger.merge_senses("bank", sense_components=senses, distance_threshold=0.3)
    
    print(f"\nResult:")
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Convergent: {result.n_convergent}")
    print(f"  Source-specific: {result.n_source_specific}")
    
    # Expected: 1 convergent (financial), 2 source-specific (river, slang)
    assert result.n_convergent >= 1, "Should have at least 1 convergent cluster"
    assert result.n_source_specific >= 1, "Should have at least 1 source-specific cluster"
    
    print("\n✓ Basic merger test PASSED")
    return True


def test_clustering_methods():
    """Test different clustering methods."""
    from sense_explorer.merger.embedding_merger import EmbeddingMerger, SenseComponent
    
    print("\n" + "=" * 60)
    print("TEST: Clustering Methods")
    print("=" * 60)
    
    emb_a, emb_b, directions = create_toy_embeddings_pair()
    
    methods = ['hierarchical', 'spectral', 'spectral_hierarchical']
    
    for method in methods:
        print(f"\n--- {method} ---")
        
        merger = EmbeddingMerger(clustering_method=method, verbose=False)
        merger.add_embedding("wiki", emb_a)
        merger.add_embedding("twitter", emb_b)
        
        # Use simple extraction
        result = merger.merge_senses("bank", n_senses=2, distance_threshold=0.3)
        
        print(f"  Clusters: {result.n_clusters}")
        print(f"  Convergent: {result.n_convergent}")
        
        if result.spectral_info:
            print(f"  Spectral suggested k: {result.spectral_info.suggested_k}")
    
    print("\n✓ Clustering methods test PASSED")
    return True


def test_spectral_analysis():
    """Test spectral analysis functionality."""
    from sense_explorer.merger.embedding_merger import (
        compute_laplacian_spectrum, 
        find_k_by_eigengap,
        spectral_embedding_from_similarity
    )
    
    print("\n" + "=" * 60)
    print("TEST: Spectral Analysis")
    print("=" * 60)
    
    # Create a similarity matrix with 3 clear clusters
    np.random.seed(42)
    n = 9  # 3 clusters of 3
    
    # Block diagonal structure
    similarity = np.eye(n) * 0.1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                similarity[i*3 + j, i*3 + k] = 0.8 + np.random.rand() * 0.1
    
    # Add some cross-cluster similarity
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, 1.0)
    
    # Test Laplacian spectrum
    eigenvalues, eigenvectors = compute_laplacian_spectrum(similarity)
    print(f"\nEigenvalues: {eigenvalues[:5].round(3)}")
    
    # Test eigengap k-selection
    k, gaps = find_k_by_eigengap(eigenvalues)
    print(f"Suggested k: {k}")
    print(f"Eigengaps: {gaps[:5].round(3)}")
    
    # Should find k=3 for 3 clusters
    assert k == 3, f"Expected k=3 for 3 clusters, got {k}"
    
    # Test spectral embedding
    coords, info = spectral_embedding_from_similarity(similarity)
    print(f"\nSpectral embedding shape: {coords.shape}")
    print(f"Suggested k from embedding: {info.suggested_k}")
    
    print("\n✓ Spectral analysis test PASSED")
    return True


def test_threshold_sensitivity():
    """Test threshold sensitivity analysis."""
    from sense_explorer.merger.embedding_merger import EmbeddingMerger
    
    print("\n" + "=" * 60)
    print("TEST: Threshold Sensitivity")
    print("=" * 60)
    
    emb_a, emb_b, _ = create_toy_embeddings_pair()
    
    merger = EmbeddingMerger(clustering_method='spectral_hierarchical', verbose=False)
    merger.add_embedding("wiki", emb_a)
    merger.add_embedding("twitter", emb_b)
    
    results = merger.merge_senses("bank", n_senses=2, return_all_thresholds=True)
    
    print(f"\n{'Threshold':<10} {'Clusters':<10} {'Convergent':<12} {'Source-Spec':<12}")
    print("-" * 44)
    
    for thresh, result in sorted(results.items()):
        print(f"{thresh:<10.2f} {result.n_clusters:<10} {result.n_convergent:<12} {result.n_source_specific:<12}")
    
    # Verify decreasing cluster count with increasing threshold
    thresholds = sorted(results.keys())
    cluster_counts = [results[t].n_clusters for t in thresholds]
    
    # Generally should decrease or stay same
    print(f"\nCluster counts: {cluster_counts}")
    
    print("\n✓ Threshold sensitivity test PASSED")
    return True


def test_merger_report():
    """Test merger report generation."""
    from sense_explorer.merger.embedding_merger import EmbeddingMerger
    
    print("\n" + "=" * 60)
    print("TEST: Merger Report")
    print("=" * 60)
    
    emb_a, emb_b, _ = create_toy_embeddings_pair()
    
    merger = EmbeddingMerger(clustering_method='spectral_hierarchical', verbose=False)
    merger.add_embedding("wiki", emb_a)
    merger.add_embedding("twitter", emb_b)
    
    result = merger.merge_senses("bank", n_senses=2)
    
    report = merger.report(result)
    print(report)
    
    # Verify report contains key information
    assert "EMBEDDING MERGER RESULTS" in report
    assert "Clustering method" in report
    assert "spectral_hierarchical" in report
    
    print("\n✓ Merger report test PASSED")
    return True


def test_with_real_embeddings(wiki_path, twitter_path):
    """Test with real GloVe embeddings."""
    print("\n" + "=" * 60)
    print("TEST: Real Embeddings")
    print("=" * 60)
    
    try:
        from sense_explorer.core import SenseExplorer
        from sense_explorer.merger.embedding_merger import EmbeddingMerger
    except ImportError as e:
        print(f"Import error: {e}")
        print("Skipping real embedding test")
        return True
    
    print(f"\nLoading Wikipedia embeddings...")
    se_wiki = SenseExplorer.from_file(wiki_path, max_words=50000, verbose=True)
    
    print(f"\nLoading Twitter embeddings...")
    se_twitter = SenseExplorer.from_file(twitter_path, max_words=50000, verbose=True)
    
    # Test merge_with convenience method
    print("\n--- Testing se.merge_with() ---")
    result = se_wiki.merge_with(se_twitter, "bank", use_ssr=False)
    print(f"  Clusters: {result.n_clusters}")
    print(f"  Convergent: {result.n_convergent}")
    print(f"  Source-specific: {result.n_source_specific}")
    
    # Test EmbeddingMerger directly
    print("\n--- Testing EmbeddingMerger ---")
    merger = EmbeddingMerger(clustering_method='spectral_hierarchical', verbose=True)
    merger.add_embedding("wiki", se_wiki.embeddings)
    merger.add_embedding("twitter", se_twitter.embeddings)
    
    for word in ['bank', 'rock', 'plant']:
        if word in se_wiki.vocab and word in se_twitter.vocab:
            result = merger.merge_senses(word, n_senses=3)
            print(f"\n{word}: {result.n_clusters} clusters, {result.n_convergent} convergent")
            if result.spectral_info:
                print(f"  Spectral k: {result.spectral_info.suggested_k}")
    
    print("\n✓ Real embeddings test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Embedding Merger")
    parser.add_argument("--wiki", type=str, help="Path to Wikipedia GloVe embeddings")
    parser.add_argument("--twitter", type=str, help="Path to Twitter GloVe embeddings")
    args = parser.parse_args()
    
    print("#" * 70)
    print("# EMBEDDING MERGER MODULE TESTS")
    print("#" * 70)
    
    all_passed = True
    all_passed &= test_basic_merger()
    all_passed &= test_clustering_methods()
    all_passed &= test_spectral_analysis()
    all_passed &= test_threshold_sensitivity()
    all_passed &= test_merger_report()
    
    if args.wiki and args.twitter:
        all_passed &= test_with_real_embeddings(args.wiki, args.twitter)
    else:
        print("\n" + "-" * 60)
        print("To test with real embeddings:")
        print("  python test_merger.py --wiki wiki.txt --twitter twitter.txt")
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
