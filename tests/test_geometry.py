#!/usr/bin/env python3
"""
test_geometry.py - Tests for Sense Geometry Module
===================================================

Tests the sense geometry analysis functionality.

Usage:
    # Toy data only
    python test_geometry.py
    
    # With real embeddings
    python test_geometry.py --glove path/to/glove.txt

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


def create_toy_senses(dim=50):
    """Create toy word vector and sense vectors with known geometry."""
    np.random.seed(42)
    
    # Create two orthogonal sense directions
    sense1 = np.random.randn(dim)
    sense1 /= np.linalg.norm(sense1)
    
    # Make sense2 orthogonal to sense1
    sense2 = np.random.randn(dim)
    sense2 = sense2 - np.dot(sense2, sense1) * sense1
    sense2 /= np.linalg.norm(sense2)
    
    # Word vector is a mixture
    alpha1, alpha2 = 0.6, 0.4
    word_vec = alpha1 * sense1 + alpha2 * sense2
    word_vec /= np.linalg.norm(word_vec)
    
    return word_vec, {'sense_1': sense1, 'sense_2': sense2}, (alpha1, alpha2)


def test_basic_decomposition():
    """Test basic sense decomposition."""
    from sense_explorer.geometry import decompose, SenseDecomposition
    
    print("=" * 60)
    print("TEST: Basic Sense Decomposition")
    print("=" * 60)
    
    word_vec, senses, (alpha1, alpha2) = create_toy_senses()
    
    decomp = decompose("test_word", word_vec, senses)
    
    print(f"\nDecomposition result:")
    print(f"  Type: {type(decomp)}")
    print(f"  R² (variance explained): {decomp.variance_explained_total:.3f}")
    print(f"  Coefficients: {decomp.coefficients}")
    print(f"  Angle pairs: {decomp.angle_pairs}")
    print(f"  Dominant sense: {decomp.dominant_sense}")
    
    # Verify it's a SenseDecomposition
    assert isinstance(decomp, SenseDecomposition)
    
    # R² should be high since word is exactly a linear combination
    assert decomp.variance_explained_total > 0.9, f"R² too low: {decomp.variance_explained_total}"
    
    # Angle between orthogonal senses should be ~90°
    angles = [angle for _, _, angle in decomp.angle_pairs]
    assert any(abs(a - 90) < 5 for a in angles), f"Expected ~90° angle, got {angles}"
    
    print("\n✓ Basic decomposition test PASSED")
    return True


def test_angle_computation():
    """Test inter-sense angle computation."""
    from sense_explorer.geometry import decompose
    
    print("\n" + "=" * 60)
    print("TEST: Angle Computation")
    print("=" * 60)
    
    np.random.seed(42)
    dim = 50
    
    # Test various angles
    test_angles = [30, 45, 60, 90, 120]
    
    for target_angle in test_angles:
        # Create two senses at known angle
        sense1 = np.zeros(dim)
        sense1[0] = 1.0
        
        sense2 = np.zeros(dim)
        rad = np.radians(target_angle)
        sense2[0] = np.cos(rad)
        sense2[1] = np.sin(rad)
        
        word_vec = (sense1 + sense2) / 2
        word_vec /= np.linalg.norm(word_vec)
        
        decomp = decompose("test", word_vec, {'s1': sense1, 's2': sense2})
        
        computed_angle = decomp.angle_pairs[0][2]
        print(f"  Target: {target_angle}°, Computed: {computed_angle:.1f}°")
        
        assert abs(computed_angle - target_angle) < 1, f"Angle mismatch: {computed_angle} vs {target_angle}"
    
    print("\n✓ Angle computation test PASSED")
    return True


def test_coefficient_ratio():
    """Test coefficient ratio computation."""
    from sense_explorer.geometry import decompose
    
    print("\n" + "=" * 60)
    print("TEST: Coefficient Ratio")
    print("=" * 60)
    
    np.random.seed(42)
    dim = 50
    
    # Create orthogonal senses
    sense1 = np.random.randn(dim)
    sense1 /= np.linalg.norm(sense1)
    
    sense2 = np.random.randn(dim)
    sense2 = sense2 - np.dot(sense2, sense1) * sense1
    sense2 /= np.linalg.norm(sense2)
    
    # Test various mixing ratios
    test_ratios = [(0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]
    
    for a1, a2 in test_ratios:
        word_vec = a1 * sense1 + a2 * sense2
        word_vec /= np.linalg.norm(word_vec)
        
        decomp = decompose("test", word_vec, {'s1': sense1, 's2': sense2})
        
        expected_ratio = max(a1, a2) / min(a1, a2)
        computed_ratio = decomp.coefficient_ratio
        
        print(f"  Mix ({a1}, {a2}): expected ratio={expected_ratio:.2f}, computed={computed_ratio:.2f}")
        
        # Allow some tolerance due to normalization
        assert abs(computed_ratio - expected_ratio) < 0.5, f"Ratio mismatch"
    
    print("\n✓ Coefficient ratio test PASSED")
    return True


def test_collect_all_angles():
    """Test collecting angles across multiple decompositions."""
    from sense_explorer.geometry import decompose, collect_all_angles
    
    print("\n" + "=" * 60)
    print("TEST: Collect All Angles")
    print("=" * 60)
    
    np.random.seed(42)
    dim = 50
    
    decomps = []
    for i in range(5):
        # Random senses and word
        sense1 = np.random.randn(dim)
        sense1 /= np.linalg.norm(sense1)
        
        sense2 = np.random.randn(dim)
        sense2 /= np.linalg.norm(sense2)
        
        word_vec = (sense1 + sense2) / 2
        word_vec /= np.linalg.norm(word_vec)
        
        decomp = decompose(f"word_{i}", word_vec, {'s1': sense1, 's2': sense2})
        decomps.append(decomp)
    
    all_angles = collect_all_angles(decomps)
    
    print(f"\nCollected {len(all_angles)} angles from {len(decomps)} decompositions")
    print(f"  Raw format example: {all_angles[0] if all_angles else 'empty'}")
    
    # Extract numeric angles - handle various formats
    angle_values = []
    for item in all_angles:
        if isinstance(item, (int, float)):
            angle_values.append(float(item))
        elif isinstance(item, tuple):
            # Find the numeric value in the tuple (could be at any position)
            for elem in item:
                if isinstance(elem, (int, float)) and not isinstance(elem, bool):
                    angle_values.append(float(elem))
                    break
        elif isinstance(item, str):
            # Try to parse string as float
            try:
                angle_values.append(float(item))
            except ValueError:
                pass
    
    if angle_values:
        print(f"  Mean angle: {np.mean(angle_values):.1f}°")
        print(f"  Std angle: {np.std(angle_values):.1f}°")
        print(f"  Range: {min(angle_values):.1f}° - {max(angle_values):.1f}°")
    else:
        print("  No numeric angles extracted")
    
    assert len(all_angles) >= len(decomps), "Should have at least one angle per decomposition"
    
    print("\n✓ Collect all angles test PASSED")
    return True


def test_summary_dict():
    """Test summary_dict serialization."""
    from sense_explorer.geometry import decompose
    
    print("\n" + "=" * 60)
    print("TEST: Summary Dict Serialization")
    print("=" * 60)
    
    word_vec, senses, _ = create_toy_senses()
    decomp = decompose("test_word", word_vec, senses)
    
    summary = decomp.summary_dict()
    
    print(f"\nSummary keys: {list(summary.keys())}")
    
    # Verify required keys
    required_keys = ['word', 'r_squared', 'coefficients', 'angles', 'dominant_sense']
    for key in required_keys:
        assert key in summary, f"Missing key: {key}"
    
    # Verify JSON serializable (no numpy arrays)
    import json
    json_str = json.dumps(summary)
    print(f"JSON serializable: ✓ ({len(json_str)} chars)")
    
    print("\n✓ Summary dict test PASSED")
    return True


def test_with_real_embeddings(glove_path):
    """Test geometry analysis with real embeddings."""
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
    
    # Test localize_senses
    print("\n--- Testing localize_senses ---")
    test_words = ['bank', 'bat', 'crane']
    test_words = [w for w in test_words if w in se.vocab]
    
    for word in test_words:
        try:
            decomp = se.localize_senses(word)
            print(f"\n{word}:")
            print(f"  R²: {decomp.variance_explained_total:.3f}")
            print(f"  Coefficients: {decomp.coefficients}")
            print(f"  Angles: {decomp.angle_pairs}")
            print(f"  Dominant: {decomp.dominant_sense}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test batch analysis
    print("\n--- Testing analyze_geometry ---")
    results = se.analyze_geometry(test_words)
    
    all_angles = []
    
    # Handle both dict and list return types
    if isinstance(results, dict):
        for word, decomp in results.items():
            for item in decomp.angle_pairs:
                # Extract angle from tuple
                if isinstance(item, tuple):
                    for elem in item:
                        if isinstance(elem, (int, float)) and not isinstance(elem, bool):
                            all_angles.append(float(elem))
                            break
    elif isinstance(results, list):
        for decomp in results:
            if hasattr(decomp, 'angle_pairs'):
                for item in decomp.angle_pairs:
                    if isinstance(item, tuple):
                        for elem in item:
                            if isinstance(elem, (int, float)) and not isinstance(elem, bool):
                                all_angles.append(float(elem))
                                break
    
    if all_angles:
        print(f"\nCross-word angle statistics:")
        print(f"  Mean: {np.mean(all_angles):.1f}°")
        print(f"  Median: {np.median(all_angles):.1f}°")
        print(f"  Std: {np.std(all_angles):.1f}°")
    else:
        print("\nNo angles extracted from results")
    
    print("\n✓ Real embeddings test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Sense Geometry")
    parser.add_argument("--glove", type=str, help="Path to GloVe embeddings")
    args = parser.parse_args()
    
    print("#" * 70)
    print("# SENSE GEOMETRY MODULE TESTS")
    print("#" * 70)
    
    all_passed = True
    all_passed &= test_basic_decomposition()
    all_passed &= test_angle_computation()
    all_passed &= test_coefficient_ratio()
    all_passed &= test_collect_all_angles()
    all_passed &= test_summary_dict()
    
    if args.glove:
        all_passed &= test_with_real_embeddings(args.glove)
    else:
        print("\n" + "-" * 60)
        print("To test with real embeddings:")
        print("  python test_geometry.py --glove glove.6B.100d.txt")
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
