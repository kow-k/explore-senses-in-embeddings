#!/usr/bin/env python3
"""
run_all_tests.py - Run All SenseExplorer Tests
===============================================

Runs all test modules in sequence with optional real embedding support.

Usage:
    # Run all tests with toy data only
    python run_all_tests.py
    
    # Run with real embeddings
    python run_all_tests.py --glove path/to/glove.txt
    
    # Run with multiple embeddings (for merger tests)
    python run_all_tests.py --glove glove.txt --twitter twitter.txt

Author: Kow Kuroda & Claude (Anthropic)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_test(test_file: str, extra_args: list = None) -> bool:
    """Run a single test file and return success status."""
    cmd = [sys.executable, test_file]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*70}")
    print(f"RUNNING: {test_file}")
    print('='*70)
    
    result = subprocess.run(cmd, cwd=os.path.dirname(test_file))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run All SenseExplorer Tests")
    parser.add_argument("--glove", type=str, help="Path to GloVe embeddings")
    parser.add_argument("--wiki", type=str, help="Path to Wikipedia embeddings")
    parser.add_argument("--twitter", type=str, help="Path to Twitter embeddings")
    parser.add_argument("--news", type=str, help="Path to News embeddings")
    parser.add_argument("--skip", nargs='+', default=[], help="Test modules to skip")
    args = parser.parse_args()
    
    # Find tests directory
    script_dir = Path(__file__).parent
    if script_dir.name != 'tests':
        tests_dir = script_dir / 'tests'
    else:
        tests_dir = script_dir
    
    # Define test modules and their arguments
    tests = [
        ('test_core.py', ['--glove'] if args.glove else []),
        ('test_spectral.py', ['--glove'] if args.glove else []),
        ('test_geometry.py', ['--glove'] if args.glove else []),
        ('test_polarity.py', ['--glove'] if args.glove else []),
        ('test_distillation.py', ['--glove'] if args.glove else []),
        ('test_merger.py', []),  # Uses --wiki and --twitter
        ('test_staged_merger.py', []),  # Uses --wiki, --twitter, --news
    ]
    
    # Build extra args for each test
    glove_arg = [args.glove] if args.glove else []
    
    results = {}
    
    print("#" * 70)
    print("# SENSEEXPLORER COMPLETE TEST SUITE")
    print("#" * 70)
    
    for test_file, extra in tests:
        test_name = test_file.replace('.py', '').replace('test_', '')
        
        if test_name in args.skip:
            print(f"\nSKIPPING: {test_file}")
            results[test_file] = 'SKIPPED'
            continue
        
        test_path = tests_dir / test_file
        
        if not test_path.exists():
            print(f"\nNOT FOUND: {test_path}")
            results[test_file] = 'NOT FOUND'
            continue
        
        # Build arguments
        test_args = []
        if extra and args.glove:
            test_args = ['--glove', args.glove]
        
        # Special handling for merger tests
        if 'merger' in test_file:
            if args.wiki:
                test_args.extend(['--wiki', args.wiki])
            if args.twitter:
                test_args.extend(['--twitter', args.twitter])
            if args.news and 'staged' in test_file:
                test_args.extend(['--news', args.news])
        
        success = run_test(str(test_path), test_args if test_args else None)
        results[test_file] = 'PASSED' if success else 'FAILED'
    
    # Summary
    print("\n" + "#" * 70)
    print("# TEST SUMMARY")
    print("#" * 70)
    
    passed = sum(1 for r in results.values() if r == 'PASSED')
    failed = sum(1 for r in results.values() if r == 'FAILED')
    skipped = sum(1 for r in results.values() if r == 'SKIPPED')
    
    for test_file, result in results.items():
        status_icon = {'PASSED': '✓', 'FAILED': '✗', 'SKIPPED': '○', 'NOT FOUND': '?'}
        print(f"  {status_icon.get(result, '?')} {test_file}: {result}")
    
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("#" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
