"""
Large-scale Register Specificity Analysis (1000 words)

Tests whether the 1D correlation between similarity and drift
holds across a larger vocabulary sample.
"""

import sys
import argparse
import numpy as np
import time


def get_test_words(se_wiki, se_twitter, n_words=1000, seed=42):
    """
    Get n_words that exist in both embeddings.
    
    Prioritizes:
    1. Known polysemous words
    2. Common words (by frequency rank in embeddings)
    """
    np.random.seed(seed)
    
    # Start with known polysemous words
    polysemous = [
        'bank', 'rock', 'apple', 'spring', 'bat', 'crane', 'light', 'run',
        'play', 'break', 'fall', 'match', 'cold', 'fair', 'fine', 'mean',
        'sound', 'watch', 'board', 'bow', 'cell', 'chip', 'club', 'coach',
        'court', 'date', 'deck', 'draft', 'drill', 'drop', 'file', 'fire',
        'firm', 'flag', 'flat', 'fly', 'fold', 'form', 'frame', 'game',
        'gear', 'grade', 'grant', 'ground', 'guard', 'guide', 'hand', 'head',
        'hold', 'host', 'iron', 'jam', 'judge', 'key', 'kick', 'kind',
        'land', 'lead', 'leaf', 'left', 'letter', 'lie', 'lift', 'line',
        'link', 'list', 'live', 'lock', 'log', 'long', 'lot', 'mail',
        'major', 'mark', 'mass', 'master', 'match', 'matter', 'measure', 'meet',
        'might', 'mind', 'mine', 'miss', 'model', 'mount', 'move', 'nail',
        'name', 'net', 'note', 'object', 'order', 'organ', 'pack', 'palm',
        'park', 'part', 'party', 'pass', 'past', 'patch', 'pay', 'pen',
        'pick', 'piece', 'pile', 'pin', 'pipe', 'pitch', 'place', 'plain',
        'plan', 'plane', 'plant', 'plate', 'plot', 'point', 'pole', 'pool',
        'pop', 'port', 'post', 'pot', 'pound', 'power', 'present', 'press',
        'pride', 'prime', 'print', 'project', 'pump', 'punch', 'pupil', 'push',
        'race', 'rack', 'rail', 'rain', 'raise', 'range', 'rank', 'rate',
        'raw', 'ray', 'reach', 'record', 'refuse', 'rest', 'return', 'rich',
        'ride', 'right', 'ring', 'rise', 'river', 'road', 'rock', 'roll',
        'room', 'root', 'rose', 'round', 'row', 'rule', 'run', 'rush',
        'safe', 'sail', 'sake', 'sale', 'salt', 'same', 'sand', 'save',
        'saw', 'scale', 'scene', 'school', 'score', 'screen', 'seal', 'season',
        'seat', 'second', 'sense', 'serve', 'set', 'settle', 'shade', 'shake',
        'shape', 'share', 'sharp', 'shed', 'shell', 'shift', 'ship', 'shock',
        'shoot', 'shop', 'short', 'shot', 'show', 'shower', 'side', 'sign',
        'silver', 'single', 'sink', 'site', 'size', 'skill', 'skin', 'sleep',
        'slide', 'slip', 'slope', 'slow', 'smell', 'smoke', 'snap', 'snow',
        'soft', 'soil', 'solid', 'solution', 'sort', 'soul', 'sound', 'source',
        'space', 'spare', 'speak', 'special', 'speed', 'spell', 'spend', 'spin',
        'spirit', 'split', 'sport', 'spot', 'spread', 'spring', 'square', 'stable',
        'staff', 'stage', 'stake', 'stamp', 'stand', 'standard', 'star', 'start',
        'state', 'station', 'stay', 'steal', 'steel', 'stem', 'step', 'stick',
        'stock', 'stone', 'stop', 'store', 'storm', 'story', 'strain', 'strange',
        'stream', 'street', 'stress', 'stretch', 'strike', 'string', 'strip', 'stroke',
        'strong', 'structure', 'struggle', 'study', 'stuff', 'style', 'subject', 'suit',
        'supply', 'support', 'surface', 'surprise', 'surround', 'survey', 'survive', 'suspect',
        'swear', 'sweep', 'sweet', 'swim', 'swing', 'switch', 'table', 'tail',
        'talk', 'tank', 'tap', 'tape', 'target', 'task', 'taste', 'tax',
        'teach', 'team', 'tear', 'tell', 'temple', 'term', 'test', 'text',
        'theory', 'thick', 'thin', 'thing', 'think', 'thought', 'thread', 'threat',
        'throat', 'throw', 'tick', 'ticket', 'tide', 'tie', 'tight', 'till',
        'timber', 'time', 'tip', 'tire', 'title', 'today', 'toe', 'tone',
        'tongue', 'tool', 'tooth', 'top', 'topic', 'touch', 'tour', 'track',
        'trade', 'traffic', 'train', 'transfer', 'transport', 'trap', 'travel', 'treat',
        'tree', 'trial', 'trick', 'trip', 'trouble', 'truck', 'trust', 'truth',
        'tube', 'tune', 'turn', 'twist', 'type', 'uncle', 'union', 'unit',
        'upper', 'upset', 'use', 'usual', 'valley', 'value', 'van', 'variety',
        'vast', 'version', 'vessel', 'victim', 'view', 'village', 'visit', 'voice',
        'volume', 'vote', 'wage', 'wait', 'wake', 'walk', 'wall', 'war',
        'warm', 'warn', 'wash', 'waste', 'watch', 'water', 'wave', 'way',
        'wealth', 'weapon', 'wear', 'weather', 'wedding', 'week', 'weight', 'welcome',
        'west', 'wheel', 'while', 'white', 'whole', 'wide', 'wild', 'will',
        'wind', 'window', 'wine', 'wing', 'winter', 'wire', 'wise', 'wish',
        'witness', 'woman', 'wonder', 'wood', 'word', 'work', 'world', 'worry',
        'worth', 'wound', 'wrap', 'write', 'wrong', 'yard', 'year', 'yellow',
        'yesterday', 'young', 'youth', 'zone'
    ]
    
    # Get shared vocabulary
    shared_vocab = set(se_wiki.vocab) & set(se_twitter.vocab)
    print(f"Shared vocabulary: {len(shared_vocab):,} words")
    
    # Start with polysemous words that exist in both
    test_words = [w for w in polysemous if w in shared_vocab]
    print(f"Polysemous words found: {len(test_words)}")
    
    # Add more words from shared vocab
    remaining = list(shared_vocab - set(test_words))
    np.random.shuffle(remaining)
    
    # Filter out very short words and numbers
    remaining = [w for w in remaining if len(w) >= 3 and w.isalpha()]
    
    # Add until we have n_words
    n_needed = n_words - len(test_words)
    if n_needed > 0:
        test_words.extend(remaining[:n_needed])
    
    print(f"Total test words: {len(test_words)}")
    return test_words[:n_words]


def analyze_1k_words(wiki_path, twitter_path, n_words=1000, output_prefix="register_1k"):
    """Run large-scale register specificity analysis."""
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import (
        compare_registers,
        cluster_by_register_specificity,
        plot_register_specificity,
        plot_register_specificity_detailed
    )
    
    print("=" * 70)
    print(f"LARGE-SCALE REGISTER SPECIFICITY ANALYSIS ({n_words} words)")
    print("=" * 70)
    
    # Load embeddings
    print(f"\nLoading Wikipedia embeddings...")
    se_wiki = SenseExplorer.from_file(wiki_path, max_words=50000, verbose=True)
    wiki_dim = se_wiki.dim
    
    print(f"\nLoading Twitter embeddings...")
    se_twitter = SenseExplorer.from_file(
        twitter_path,
        max_words=50000,
        target_dim=wiki_dim,
        verbose=True
    )
    
    explorers = {"wiki": se_wiki, "twitter": se_twitter}
    
    # Get test words
    print("\n" + "-" * 40)
    test_words = get_test_words(se_wiki, se_twitter, n_words=n_words)
    
    # Analyze in batches (for progress tracking)
    print("\n" + "-" * 40)
    print("Analyzing words...")
    
    start_time = time.time()
    profiles = compare_registers(test_words, explorers, n_senses=2, verbose=False)
    elapsed = time.time() - start_time
    
    print(f"Analyzed {len(profiles)} words in {elapsed:.1f}s ({len(profiles)/elapsed:.1f} words/sec)")
    
    # Extract metrics
    words = list(profiles.keys())
    similarities = []
    drifts = []
    
    for word, profile in profiles.items():
        similarities.append(profile.overall_register_similarity)
        word_drifts = [sd.mean_drift for sd in profile.sense_drifts.values()]
        drifts.append(np.mean(word_drifts) if word_drifts else 0.0)
    
    similarities = np.array(similarities)
    drifts = np.array(drifts)
    
    # Compute correlation
    correlation = np.corrcoef(similarities, drifts)[0, 1]
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nWords analyzed: {len(profiles)}")
    print(f"\nSimilarity statistics:")
    print(f"  Mean: {similarities.mean():.3f}")
    print(f"  Std:  {similarities.std():.3f}")
    print(f"  Min:  {similarities.min():.3f}")
    print(f"  Max:  {similarities.max():.3f}")
    
    print(f"\nDrift statistics:")
    print(f"  Mean: {drifts.mean():.3f}")
    print(f"  Std:  {drifts.std():.3f}")
    print(f"  Min:  {drifts.min():.3f}")
    print(f"  Max:  {drifts.max():.3f}")
    
    print(f"\n*** Correlation (similarity vs drift): {correlation:.4f} ***")
    
    if abs(correlation) > 0.8:
        print("    → Strong negative correlation: effectively 1-dimensional")
    elif abs(correlation) > 0.5:
        print("    → Moderate correlation: some 2D structure")
    else:
        print("    → Weak correlation: genuine 2D variation")
    
    # Find outliers (off-diagonal words)
    # Expected: high similarity → low drift (and vice versa)
    # Outliers: high similarity + high drift OR low similarity + low drift
    
    # Compute residuals from regression line
    slope, intercept = np.polyfit(similarities, drifts, 1)
    predicted_drifts = slope * similarities + intercept
    residuals = drifts - predicted_drifts
    
    # Find words with large residuals
    residual_threshold = 2 * residuals.std()
    outlier_indices = np.where(np.abs(residuals) > residual_threshold)[0]
    
    print(f"\nOff-diagonal outliers ({len(outlier_indices)} words with |residual| > 2σ):")
    if len(outlier_indices) > 0:
        for idx in outlier_indices[:20]:  # Show top 20
            word = words[idx]
            print(f"  {word}: similarity={similarities[idx]:.3f}, drift={drifts[idx]:.3f}, residual={residuals[idx]:.3f}")
    else:
        print("  None found - distribution is strongly 1-dimensional")
    
    # Cluster analysis
    print("\n" + "-" * 40)
    print("Clustering by register specificity...")
    
    analysis = cluster_by_register_specificity(profiles, method="threshold")
    print(f"\n{analysis.report()}")
    
    # Generate plots
    print("\n" + "-" * 40)
    print("Generating plots...")
    
    try:
        # Simple scatter plot
        fig1 = plot_register_specificity(
            profiles,
            output_path=f"{output_prefix}_scatter.png",
            title=f"Register Specificity ({len(profiles)} words, Wiki vs Twitter)"
        )
        
        # Detailed plot
        fig2 = plot_register_specificity_detailed(
            profiles,
            output_path=f"{output_prefix}_detailed.png"
        )
        
        print(f"Plots saved: {output_prefix}_scatter.png, {output_prefix}_detailed.png")
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Save raw data
    print("\n" + "-" * 40)
    print("Saving raw data...")
    
    with open(f"{output_prefix}_data.tsv", "w") as f:
        f.write("word\tsimilarity\tdrift\tresidual\n")
        for i, word in enumerate(words):
            f.write(f"{word}\t{similarities[i]:.4f}\t{drifts[i]:.4f}\t{residuals[i]:.4f}\n")
    
    print(f"Data saved to {output_prefix}_data.tsv")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return profiles, correlation, outlier_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large-scale register specificity analysis")
    parser.add_argument("--wiki", required=True, help="Path to Wikipedia GloVe embeddings")
    parser.add_argument("--twitter", required=True, help="Path to Twitter GloVe embeddings")
    parser.add_argument("--n-words", type=int, default=1000, help="Number of words to analyze")
    parser.add_argument("--output", default="register_1k", help="Output file prefix")
    args = parser.parse_args()
    
    profiles, correlation, outliers = analyze_1k_words(
        args.wiki, 
        args.twitter,
        n_words=args.n_words,
        output_prefix=args.output
    )
