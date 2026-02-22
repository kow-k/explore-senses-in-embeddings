"""
Flexible Register Specificity Analysis

Compare any two embedding sources (wiki vs twitter, wiki vs news, etc.)
"""

import sys
import argparse
import numpy as np
import time


def get_test_words(se1, se2, n_words=1000, seed=42):
    """
    Get n_words that exist in both embeddings.
    """
    np.random.seed(seed)
    
    # Known polysemous words
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
    shared_vocab = set(se1.vocab) & set(se2.vocab)
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


def analyze_two_embeddings(
    emb1_path, 
    emb2_path, 
    emb1_name="source1",
    emb2_name="source2",
    n_words=1000, 
    output_prefix="register_comparison",
    max_words=50000
):
    """Run register specificity analysis between any two embeddings."""
    
    from sense_explorer.core import SenseExplorer
    from sense_explorer.merger import (
        compare_registers,
        cluster_by_register_specificity,
        plot_register_specificity,
        plot_register_specificity_detailed
    )
    
    print("=" * 70)
    print(f"REGISTER SPECIFICITY ANALYSIS")
    print(f"  {emb1_name} vs {emb2_name}")
    print(f"  {n_words} words")
    print("=" * 70)
    
    # Load first embedding
    print(f"\nLoading {emb1_name} embeddings from {emb1_path}...")
    se1 = SenseExplorer.from_file(emb1_path, max_words=max_words, verbose=True)
    dim1 = se1.dim
    
    # Load second embedding (align dimensions if needed)
    print(f"\nLoading {emb2_name} embeddings from {emb2_path}...")
    se2 = SenseExplorer.from_file(
        emb2_path,
        max_words=max_words,
        target_dim=dim1,
        verbose=True
    )
    
    print(f"\nDimensions: {emb1_name}={se1.dim}d, {emb2_name}={se2.dim}d")
    
    explorers = {emb1_name: se1, emb2_name: se2}
    
    # Get test words
    print("\n" + "-" * 40)
    test_words = get_test_words(se1, se2, n_words=n_words)
    
    # Analyze
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
    
    # Compute residuals from main line (drift = 1 - similarity)
    residuals = drifts + similarities - 1
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nWords analyzed: {len(profiles)}")
    print(f"\nSimilarity ({emb1_name} ↔ {emb2_name}):")
    print(f"  Mean: {similarities.mean():.3f}")
    print(f"  Std:  {similarities.std():.3f}")
    print(f"  Min:  {similarities.min():.3f}")
    print(f"  Max:  {similarities.max():.3f}")
    
    print(f"\nDrift:")
    print(f"  Mean: {drifts.mean():.3f}")
    print(f"  Std:  {drifts.std():.3f}")
    print(f"  Min:  {drifts.min():.3f}")
    print(f"  Max:  {drifts.max():.3f}")
    
    print(f"\n*** Correlation (similarity vs drift): {correlation:.4f} ***")
    
    # Analyze two populations
    print("\n" + "-" * 40)
    print("TWO-POPULATION ANALYSIS")
    print("-" * 40)
    
    # Main line: residual near 0
    # Secondary line: residual << 0 (low drift given similarity)
    threshold = -0.05
    
    main_line_count = np.sum(residuals >= threshold)
    secondary_line_count = np.sum(residuals < threshold)
    
    print(f"\nMain line (drift ≈ 1 - similarity): {main_line_count} words ({100*main_line_count/len(words):.1f}%)")
    print(f"Secondary line (drift << 1 - similarity): {secondary_line_count} words ({100*secondary_line_count/len(words):.1f}%)")
    
    # Show secondary line words
    secondary_words = [(w, similarities[i], drifts[i], residuals[i]) 
                       for i, w in enumerate(words) if residuals[i] < threshold]
    secondary_words.sort(key=lambda x: x[3])
    
    if secondary_words:
        print(f"\nSecondary line words (most deviant first):")
        for w, s, d, r in secondary_words[:25]:
            print(f"  {w}: sim={s:.3f}, drift={d:.3f}, residual={r:.4f}")
        if len(secondary_words) > 25:
            print(f"  ... and {len(secondary_words) - 25} more")
    
    # Generate plots
    print("\n" + "-" * 40)
    print("Generating plots...")
    
    try:
        fig1 = plot_register_specificity(
            profiles,
            output_path=f"{output_prefix}_scatter.png",
            title=f"Register Specificity ({len(profiles)} words, {emb1_name} vs {emb2_name})"
        )
        
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
        f.write("word\tsimilarity\tdrift\tresidual\tpopulation\n")
        for i, word in enumerate(words):
            pop = "secondary" if residuals[i] < threshold else "main"
            f.write(f"{word}\t{similarities[i]:.4f}\t{drifts[i]:.4f}\t{residuals[i]:.4f}\t{pop}\n")
    
    print(f"Data saved to {output_prefix}_data.tsv")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return profiles, correlation, secondary_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare register specificity between any two embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Wiki vs Twitter
  python test_register_compare.py \\
    --emb1 wiki.gz --name1 wiki \\
    --emb2 twitter.gz --name2 twitter

  # Wiki vs Google News
  python test_register_compare.py \\
    --emb1 glove-wiki-100.gz --name1 wiki \\
    --emb2 word2vec-google-news-300.gz --name2 news

  # Any two embeddings
  python test_register_compare.py \\
    --emb1 source1.gz --name1 formal \\
    --emb2 source2.gz --name2 informal
        """
    )
    parser.add_argument("--emb1", required=True, help="Path to first embedding file")
    parser.add_argument("--name1", default="source1", help="Name for first embedding (e.g., 'wiki')")
    parser.add_argument("--emb2", required=True, help="Path to second embedding file")
    parser.add_argument("--name2", default="source2", help="Name for second embedding (e.g., 'twitter')")
    parser.add_argument("--n-words", type=int, default=1000, help="Number of words to analyze")
    parser.add_argument("--output", default="register_comparison", help="Output file prefix")
    parser.add_argument("--max-words", type=int, default=50000, help="Max words to load from each embedding")
    
    args = parser.parse_args()
    
    profiles, correlation, secondary = analyze_two_embeddings(
        args.emb1,
        args.emb2,
        emb1_name=args.name1,
        emb2_name=args.name2,
        n_words=args.n_words,
        output_prefix=args.output,
        max_words=args.max_words
    )
