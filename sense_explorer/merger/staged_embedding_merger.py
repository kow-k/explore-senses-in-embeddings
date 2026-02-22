#!/usr/bin/env python3
"""
staged_embedding_merger.py - Staged Embedding Merger for SenseExplorer
=======================================================================

Memory-efficient multi-embedding merger using deliberate staging.

When embeddings are too large to process simultaneously, this module
provides strategies for optimal merge ordering to minimize information loss.

Key Insight:
    Merge order matters when staging is required. Merging similar embeddings
    first preserves more semantic structure than arbitrary ordering.

Strategies:
    1. Affinity-based: Merge most similar embedding pairs first
    2. Anchor-based: Start with highest-quality embedding as anchor
    3. Hierarchical: Build embedding dendrogram, merge bottom-up

Usage:
    ```python
    from sense_explorer.merger import StagedMerger, MergeStrategy
    
    # Define embeddings (paths, not loaded)
    embedding_specs = {
        "wiki": {"path": "glove-wiki-300d.txt", "format": "glove"},
        "twitter": {"path": "glove-twitter-200d.txt", "format": "glove"},
        "news": {"path": "word2vec-news-300d.bin", "format": "word2vec"},
        "common_crawl": {"path": "fasttext-cc-300d.vec", "format": "fasttext"},
    }
    
    # Create staged merger
    staged = StagedMerger(
        embedding_specs,
        max_concurrent=2,  # Memory constraint
        strategy=MergeStrategy.AFFINITY,
        verbose=True
    )
    
    # Compute optimal merge order (loads only samples)
    merge_plan = staged.plan_merge_order(sample_words=["bank", "rock", "plant"])
    print(merge_plan)
    
    # Execute staged merge for a word
    result = staged.merge_staged("bank", n_senses=3)
    ```

Author: Kow Kuroda & Claude (Anthropic)
Version: 0.1.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import warnings
import gc

try:
    from .embedding_merger import (
        EmbeddingMerger, SenseComponent, MergerResult, MergerBasis
    )
except ImportError:
    from embedding_merger import (
        EmbeddingMerger, SenseComponent, MergerResult, MergerBasis
    )


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class MergeStrategy(Enum):
    """Strategies for determining merge order."""
    AFFINITY = "affinity"      # Merge most similar embeddings first
    ANCHOR = "anchor"          # Start with best embedding, add others
    HIERARCHICAL = "hierarchical"  # Build dendrogram over embeddings
    SEQUENTIAL = "sequential"  # User-specified order


@dataclass
class EmbeddingSpec:
    """Specification for a lazily-loaded embedding."""
    name: str
    path: str
    format: str = "glove"  # glove, word2vec, fasttext
    dimension: int = None
    vocab_size: int = None
    
    # Computed properties (filled during planning)
    quality_score: float = None  # Sense separation quality
    sample_vocab: Set[str] = field(default_factory=set)


@dataclass
class MergeStep:
    """A single step in the merge plan."""
    step_number: int
    embeddings_to_load: List[str]
    embeddings_to_merge: List[str]
    rationale: str
    expected_convergence: float = None


@dataclass
class MergePlan:
    """Complete plan for staged merging."""
    steps: List[MergeStep]
    total_embeddings: int
    strategy: MergeStrategy
    affinity_matrix: Dict[Tuple[str, str], float] = None
    
    def __str__(self):
        lines = ["=" * 60]
        lines.append(f"MERGE PLAN ({self.strategy.value} strategy)")
        lines.append(f"Total embeddings: {self.total_embeddings}")
        lines.append("=" * 60)
        
        for step in self.steps:
            lines.append(f"\nStep {step.step_number}:")
            lines.append(f"  Load: {step.embeddings_to_load}")
            lines.append(f"  Merge: {step.embeddings_to_merge}")
            lines.append(f"  Rationale: {step.rationale}")
            if step.expected_convergence:
                lines.append(f"  Expected convergence: {step.expected_convergence:.2%}")
        
        return "\n".join(lines)


@dataclass
class StagedMergeResult:
    """Result of staged merging."""
    word: str
    final_result: MergerResult
    intermediate_results: List[MergerResult]
    merge_plan: MergePlan
    
    # Track information flow
    sense_lineage: Dict[str, List[str]]  # final_sense -> [source_senses]
    convergence_history: List[float]


# =============================================================================
# EMBEDDING LOADERS
# =============================================================================

def load_embedding_sample(
    path: str,
    format: str = "glove",
    sample_words: List[str] = None,
    max_words: int = 50000
) -> Dict[str, np.ndarray]:
    """
    Load a sample of an embedding for affinity computation.
    
    If sample_words provided, loads only those words.
    Otherwise loads first max_words.
    """
    import gzip
    
    vectors = {}
    
    open_fn = gzip.open if path.endswith('.gz') else open
    mode = 'rt' if path.endswith('.gz') else 'r'
    
    sample_set = set(sample_words) if sample_words else None
    
    try:
        with open_fn(path, mode, encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if sample_set is None and i >= max_words:
                    break
                
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                
                word = parts[0]
                
                # Skip if not in sample set
                if sample_set and word not in sample_set:
                    continue
                
                try:
                    vec = np.array([float(x) for x in parts[1:]])
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm
                    vectors[word] = vec
                except ValueError:
                    continue
                
                # Early exit if we have all sample words
                if sample_set and len(vectors) >= len(sample_set):
                    break
    except Exception as e:
        warnings.warn(f"Error loading {path}: {e}")
    
    return vectors


def load_embedding_full(
    path: str,
    format: str = "glove"
) -> Dict[str, np.ndarray]:
    """Load full embedding (memory-intensive)."""
    return load_embedding_sample(path, format, sample_words=None, max_words=float('inf'))


# =============================================================================
# AFFINITY COMPUTATION
# =============================================================================

def compute_neighborhood(
    word: str,
    vectors: Dict[str, np.ndarray],
    k: int = 50
) -> List[str]:
    """Get top-k neighbors of a word."""
    if word not in vectors:
        return []
    
    target = vectors[word]
    sims = []
    
    for w, vec in vectors.items():
        if w != word:
            sim = np.dot(target, vec)
            sims.append((w, sim))
    
    sims.sort(key=lambda x: -x[1])
    return [w for w, _ in sims[:k]]


def compute_embedding_affinity(
    emb_a: Dict[str, np.ndarray],
    emb_b: Dict[str, np.ndarray],
    sample_words: List[str],
    neighbor_k: int = 50
) -> Tuple[float, Dict]:
    """
    Compute affinity between two embeddings.
    
    Affinity = average neighborhood overlap across sample words.
    Higher affinity = more similar semantic structure = should merge early.
    
    Returns:
        (affinity_score, details_dict)
    """
    overlaps = []
    details = {"per_word": {}}
    
    shared_vocab = set(emb_a.keys()) & set(emb_b.keys())
    
    for word in sample_words:
        if word not in emb_a or word not in emb_b:
            continue
        
        neighbors_a = set(compute_neighborhood(word, emb_a, neighbor_k))
        neighbors_b = set(compute_neighborhood(word, emb_b, neighbor_k))
        
        # Filter to shared vocabulary
        neighbors_a = neighbors_a & shared_vocab
        neighbors_b = neighbors_b & shared_vocab
        
        if not neighbors_a or not neighbors_b:
            continue
        
        # Jaccard overlap
        intersection = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)
        overlap = intersection / union if union > 0 else 0
        
        overlaps.append(overlap)
        details["per_word"][word] = {
            "overlap": overlap,
            "shared_neighbors": list(neighbors_a & neighbors_b)[:10]
        }
    
    affinity = np.mean(overlaps) if overlaps else 0
    details["mean_overlap"] = affinity
    details["n_words_tested"] = len(overlaps)
    details["shared_vocab_size"] = len(shared_vocab)
    
    return affinity, details


def compute_sense_quality(
    word: str,
    vectors: Dict[str, np.ndarray],
    n_senses: int = 3
) -> float:
    """
    Estimate sense separation quality for an embedding.
    
    Uses cluster coherence as a proxy for quality.
    Higher = better sense separation = good anchor candidate.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    if word not in vectors:
        return 0.0
    
    # Get neighbors
    neighbors = compute_neighborhood(word, vectors, k=50)
    if len(neighbors) < n_senses * 2:
        return 0.0
    
    neighbor_vecs = np.array([vectors[w] for w in neighbors if w in vectors])
    
    if len(neighbor_vecs) < n_senses * 2:
        return 0.0
    
    # Cluster and measure quality
    try:
        kmeans = KMeans(n_clusters=n_senses, random_state=42, n_init=10)
        labels = kmeans.fit_predict(neighbor_vecs)
        
        if len(set(labels)) < 2:
            return 0.0
        
        score = silhouette_score(neighbor_vecs, labels)
        return max(0, score)  # Silhouette can be negative
    except:
        return 0.0


# =============================================================================
# MERGE PLANNING
# =============================================================================

def plan_affinity_based(
    specs: Dict[str, EmbeddingSpec],
    affinity_matrix: Dict[Tuple[str, str], float],
    max_concurrent: int = 2
) -> List[MergeStep]:
    """
    Plan merge order based on embedding affinity.
    
    Strategy: Merge most similar pairs first, then incorporate others.
    """
    names = list(specs.keys())
    n = len(names)
    
    if n <= max_concurrent:
        # Can merge all at once
        return [MergeStep(
            step_number=1,
            embeddings_to_load=names,
            embeddings_to_merge=names,
            rationale="All embeddings fit in memory"
        )]
    
    # Build priority queue of pairs by affinity
    pairs = []
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names[i+1:], i+1):
            key = (name_i, name_j) if (name_i, name_j) in affinity_matrix else (name_j, name_i)
            if key in affinity_matrix:
                pairs.append((affinity_matrix[key], name_i, name_j))
    
    pairs.sort(reverse=True)  # Highest affinity first
    
    # Greedy merge planning
    steps = []
    merged = set()
    merged_group = None  # The current merged result
    step_num = 1
    
    while len(merged) < n:
        if merged_group is None:
            # First step: merge highest affinity pair
            if pairs:
                _, a, b = pairs[0]
                steps.append(MergeStep(
                    step_number=step_num,
                    embeddings_to_load=[a, b],
                    embeddings_to_merge=[a, b],
                    rationale=f"Highest affinity pair ({affinity_matrix.get((a,b), affinity_matrix.get((b,a), 0)):.3f})",
                    expected_convergence=affinity_matrix.get((a,b), affinity_matrix.get((b,a), 0))
                ))
                merged.add(a)
                merged.add(b)
                merged_group = "merged_1"
            else:
                # No affinity info, just take first two
                a, b = names[0], names[1]
                steps.append(MergeStep(
                    step_number=step_num,
                    embeddings_to_load=[a, b],
                    embeddings_to_merge=[a, b],
                    rationale="First two embeddings (no affinity data)"
                ))
                merged.add(a)
                merged.add(b)
                merged_group = "merged_1"
        else:
            # Add next best embedding to merged group
            remaining = [n for n in names if n not in merged]
            if not remaining:
                break
            
            # Find embedding with highest average affinity to merged set
            best_name = None
            best_avg_affinity = -1
            
            for name in remaining:
                affinities = []
                for m in merged:
                    key = (name, m) if (name, m) in affinity_matrix else (m, name)
                    if key in affinity_matrix:
                        affinities.append(affinity_matrix[key])
                
                avg = np.mean(affinities) if affinities else 0
                if avg > best_avg_affinity:
                    best_avg_affinity = avg
                    best_name = name
            
            if best_name:
                steps.append(MergeStep(
                    step_number=step_num,
                    embeddings_to_load=[best_name],  # merged_group already in memory conceptually
                    embeddings_to_merge=[merged_group, best_name],
                    rationale=f"Best affinity to merged group ({best_avg_affinity:.3f})",
                    expected_convergence=best_avg_affinity
                ))
                merged.add(best_name)
                merged_group = f"merged_{step_num}"
        
        step_num += 1
    
    return steps


def plan_anchor_based(
    specs: Dict[str, EmbeddingSpec],
    quality_scores: Dict[str, float],
    max_concurrent: int = 2
) -> List[MergeStep]:
    """
    Plan merge order starting from highest-quality anchor.
    
    Strategy: Start with best sense separation, add others incrementally.
    """
    # Sort by quality
    ranked = sorted(quality_scores.items(), key=lambda x: -x[1])
    
    steps = []
    anchor = ranked[0][0]
    merged = {anchor}
    step_num = 1
    
    # First step: anchor alone (just sense separation)
    steps.append(MergeStep(
        step_number=step_num,
        embeddings_to_load=[anchor],
        embeddings_to_merge=[anchor],
        rationale=f"Anchor embedding (quality={quality_scores[anchor]:.3f})"
    ))
    step_num += 1
    
    # Add others in quality order
    for name, quality in ranked[1:]:
        steps.append(MergeStep(
            step_number=step_num,
            embeddings_to_load=[name],
            embeddings_to_merge=["previous_merged", name],
            rationale=f"Add by quality rank (quality={quality:.3f})"
        ))
        step_num += 1
    
    return steps


# =============================================================================
# MAIN CLASS
# =============================================================================

class StagedMerger:
    """
    Memory-efficient staged embedding merger.
    
    For large embeddings that can't be loaded simultaneously,
    this class plans and executes an optimal merge order.
    """
    
    def __init__(
        self,
        embedding_specs: Dict[str, Dict[str, Any]],
        max_concurrent: int = 2,
        strategy: MergeStrategy = MergeStrategy.AFFINITY,
        verbose: bool = True
    ):
        """
        Initialize staged merger.
        
        Args:
            embedding_specs: Dict of {name: {"path": ..., "format": ...}}
            max_concurrent: Maximum embeddings to load simultaneously
            strategy: Merge ordering strategy
            verbose: Print progress
        """
        self.specs = {
            name: EmbeddingSpec(name=name, **spec)
            for name, spec in embedding_specs.items()
        }
        self.max_concurrent = max_concurrent
        self.strategy = strategy
        self.verbose = verbose
        
        self._affinity_matrix: Dict[Tuple[str, str], float] = {}
        self._quality_scores: Dict[str, float] = {}
        self._merge_plan: Optional[MergePlan] = None
        
        # Cache for loaded embeddings
        self._loaded: Dict[str, Dict[str, np.ndarray]] = {}
    
    def plan_merge_order(
        self,
        sample_words: List[str] = None,
        neighbor_k: int = 50
    ) -> MergePlan:
        """
        Plan optimal merge order based on embedding affinities.
        
        This loads only samples of each embedding to compute affinities,
        then determines the best merge sequence.
        
        Args:
            sample_words: Words to use for affinity computation
            neighbor_k: Neighbors to consider per word
            
        Returns:
            MergePlan describing the merge sequence
        """
        if sample_words is None:
            sample_words = ["bank", "rock", "plant", "spring", "star",
                          "cell", "bat", "crane", "mouse", "run"]
        
        if self.verbose:
            print("=" * 60)
            print("PLANNING MERGE ORDER")
            print(f"Strategy: {self.strategy.value}")
            print(f"Max concurrent: {self.max_concurrent}")
            print(f"Sample words: {sample_words}")
            print("=" * 60)
        
        names = list(self.specs.keys())
        
        # Load samples
        samples = {}
        for name, spec in self.specs.items():
            if self.verbose:
                print(f"\nLoading sample from {name}...")
            
            samples[name] = load_embedding_sample(
                spec.path, spec.format,
                sample_words=sample_words + self._get_common_words(),
                max_words=10000
            )
            
            if self.verbose:
                print(f"  Loaded {len(samples[name])} words")
        
        # Compute pairwise affinities
        if self.verbose:
            print("\nComputing pairwise affinities...")
        
        for i, name_i in enumerate(names):
            for name_j in names[i+1:]:
                affinity, details = compute_embedding_affinity(
                    samples[name_i], samples[name_j],
                    sample_words, neighbor_k
                )
                self._affinity_matrix[(name_i, name_j)] = affinity
                
                if self.verbose:
                    print(f"  {name_i} <-> {name_j}: {affinity:.3f}")
        
        # Compute quality scores (for anchor strategy)
        if self.strategy == MergeStrategy.ANCHOR:
            if self.verbose:
                print("\nComputing quality scores...")
            
            for name in names:
                quality = np.mean([
                    compute_sense_quality(word, samples[name])
                    for word in sample_words
                    if word in samples[name]
                ])
                self._quality_scores[name] = quality
                
                if self.verbose:
                    print(f"  {name}: {quality:.3f}")
        
        # Generate plan
        if self.strategy == MergeStrategy.AFFINITY:
            steps = plan_affinity_based(self.specs, self._affinity_matrix, self.max_concurrent)
        elif self.strategy == MergeStrategy.ANCHOR:
            steps = plan_anchor_based(self.specs, self._quality_scores, self.max_concurrent)
        else:
            # Sequential: user-specified order
            steps = [MergeStep(
                step_number=i+1,
                embeddings_to_load=[name],
                embeddings_to_merge=["previous"] if i > 0 else [name],
                rationale="User-specified order"
            ) for i, name in enumerate(names)]
        
        self._merge_plan = MergePlan(
            steps=steps,
            total_embeddings=len(names),
            strategy=self.strategy,
            affinity_matrix=self._affinity_matrix
        )
        
        if self.verbose:
            print("\n" + str(self._merge_plan))
        
        # Clean up samples
        del samples
        gc.collect()
        
        return self._merge_plan
    
    def merge_staged(
        self,
        word: str,
        n_senses: int = 3,
        distance_threshold: float = 0.05,
        plan: MergePlan = None
    ) -> StagedMergeResult:
        """
        Execute staged merge for a word.
        
        Args:
            word: Target word
            n_senses: Senses per embedding
            distance_threshold: Clustering threshold
            plan: Merge plan (if None, uses previously computed plan)
            
        Returns:
            StagedMergeResult with final and intermediate results
        """
        if plan is None:
            plan = self._merge_plan
        
        if plan is None:
            raise ValueError("No merge plan. Call plan_merge_order() first.")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"STAGED MERGE: '{word}'")
            print("=" * 60)
        
        intermediate_results = []
        current_senses: List[SenseComponent] = []
        convergence_history = []
        sense_lineage = defaultdict(list)
        
        for step in plan.steps:
            if self.verbose:
                print(f"\n--- Step {step.step_number}: {step.rationale} ---")
            
            # Load required embeddings
            for emb_name in step.embeddings_to_load:
                if emb_name not in self._loaded and emb_name in self.specs:
                    if self.verbose:
                        print(f"  Loading {emb_name}...")
                    spec = self.specs[emb_name]
                    self._loaded[emb_name] = load_embedding_full(spec.path, spec.format)
                    if self.verbose:
                        print(f"    Loaded {len(self._loaded[emb_name])} words")
            
            # Extract senses from newly loaded embeddings
            new_senses = []
            for emb_name in step.embeddings_to_load:
                if emb_name in self._loaded:
                    senses = self._extract_senses_simple(
                        word, emb_name, self._loaded[emb_name], n_senses
                    )
                    new_senses.extend(senses)
                    
                    if self.verbose:
                        print(f"  {emb_name}: {len(senses)} senses")
                        for s in senses:
                            neighbors = [w for w, _ in s.top_neighbors[:5]]
                            print(f"    - {s.sense_id}: {', '.join(neighbors)}")
            
            # Combine with previous senses
            all_senses = current_senses + new_senses
            
            if len(all_senses) < 2:
                if self.verbose:
                    print(f"  Skipping merge (only {len(all_senses)} senses)")
                current_senses = all_senses
                continue
            
            # Create merger for this step
            merger = EmbeddingMerger(verbose=False)
            for emb_name, vectors in self._loaded.items():
                merger.add_embedding(emb_name, vectors)
            
            # Merge
            result = merger.merge_senses(
                word,
                sense_components=all_senses,
                distance_threshold=distance_threshold
            )
            
            intermediate_results.append(result)
            convergence_history.append(result.n_convergent / result.n_clusters if result.n_clusters > 0 else 0)
            
            if self.verbose:
                print(f"  Result: {result.n_clusters} clusters, {result.n_convergent} convergent")
            
            # Update current senses based on clustering
            # (For next step, we use cluster centroids as representatives)
            current_senses = self._compute_cluster_representatives(result, all_senses)
            
            # Track lineage
            for sense in new_senses:
                sense_lineage[sense.sense_id].append(sense.sense_id)
            
            # Unload embeddings not needed for next steps
            # (Keep only what's needed)
            self._maybe_unload(step, plan)
        
        # Final result is the last intermediate result
        final_result = intermediate_results[-1] if intermediate_results else None
        
        return StagedMergeResult(
            word=word,
            final_result=final_result,
            intermediate_results=intermediate_results,
            merge_plan=plan,
            sense_lineage=dict(sense_lineage),
            convergence_history=convergence_history
        )
    
    def _extract_senses_simple(
        self,
        word: str,
        source: str,
        vectors: Dict[str, np.ndarray],
        n_senses: int
    ) -> List[SenseComponent]:
        """Extract senses using simple k-means."""
        from sklearn.cluster import KMeans
        
        if word not in vectors:
            return []
        
        target_vec = vectors[word]
        
        # Get neighbors
        all_words = [w for w in vectors.keys() if w != word]
        similarities = []
        for w in all_words:
            sim = np.dot(target_vec, vectors[w])
            similarities.append((w, sim))
        
        similarities.sort(key=lambda x: -x[1])
        top_neighbors = similarities[:50]
        neighbor_words = [w for w, _ in top_neighbors]
        neighbor_vecs = np.array([vectors[w] for w in neighbor_words])
        
        # Cluster
        n_actual = min(n_senses, len(neighbor_words) // 2)
        if n_actual < 2:
            return []
        
        kmeans = KMeans(n_clusters=n_actual, random_state=42, n_init=10)
        labels = kmeans.fit_predict(neighbor_vecs)
        
        senses = []
        for sense_idx in range(n_actual):
            mask = labels == sense_idx
            sense_neighbors = [neighbor_words[i] for i in range(len(neighbor_words)) if mask[i]]
            
            if not sense_neighbors:
                continue
            
            sense_vec = np.mean([vectors[w] for w in sense_neighbors], axis=0)
            sense_vec /= np.linalg.norm(sense_vec) + 1e-10
            
            neighbors_with_sim = []
            for w in sense_neighbors:
                sim = np.dot(sense_vec, vectors[w])
                neighbors_with_sim.append((w, float(sim)))
            neighbors_with_sim.sort(key=lambda x: -x[1])
            
            senses.append(SenseComponent(
                word=word,
                sense_id=f"{source}_{word}_s{sense_idx}",
                vector=sense_vec,
                source=source,
                top_neighbors=neighbors_with_sim
            ))
        
        return senses
    
    def _compute_cluster_representatives(
        self,
        result: MergerResult,
        senses: List[SenseComponent]
    ) -> List[SenseComponent]:
        """Compute representative senses for each cluster."""
        cluster_senses = defaultdict(list)
        for sense in senses:
            cluster_id = result.clusters.get(sense.sense_id, -1)
            cluster_senses[cluster_id].append(sense)
        
        representatives = []
        for cluster_id, members in cluster_senses.items():
            if not members:
                continue
            
            # Use centroid as representative
            centroid = np.mean([s.vector for s in members], axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-10
            
            # Combine neighbors
            all_neighbors = []
            for s in members:
                all_neighbors.extend(s.top_neighbors)
            
            # Deduplicate and re-rank
            neighbor_dict = {}
            for w, sim in all_neighbors:
                if w not in neighbor_dict or sim > neighbor_dict[w]:
                    neighbor_dict[w] = sim
            sorted_neighbors = sorted(neighbor_dict.items(), key=lambda x: -x[1])
            
            representatives.append(SenseComponent(
                word=members[0].word,
                sense_id=f"merged_cluster_{cluster_id}",
                vector=centroid,
                source="merged",
                top_neighbors=sorted_neighbors[:50]
            ))
        
        return representatives
    
    def _maybe_unload(self, current_step: MergeStep, plan: MergePlan):
        """Unload embeddings not needed for future steps."""
        # Find what's needed in future steps
        needed = set()
        found_current = False
        
        for step in plan.steps:
            if step.step_number == current_step.step_number:
                found_current = True
                continue
            if found_current:
                needed.update(step.embeddings_to_load)
        
        # Unload what's not needed
        to_unload = [name for name in self._loaded if name not in needed]
        for name in to_unload:
            if self.verbose:
                print(f"  Unloading {name} to free memory")
            del self._loaded[name]
        
        if to_unload:
            gc.collect()
    
    def _get_common_words(self) -> List[str]:
        """Get common words for vocabulary overlap estimation."""
        return [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
            "time", "year", "people", "way", "day", "man", "thing", "woman",
            "life", "child", "world", "school", "state", "family", "student",
            "money", "work", "system", "problem", "fact", "business", "water"
        ]
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self._loaded.clear()
        gc.collect()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_staged_merge(
    embedding_paths: Dict[str, str],
    word: str,
    max_concurrent: int = 2,
    n_senses: int = 3,
    strategy: MergeStrategy = MergeStrategy.AFFINITY,
    verbose: bool = True
) -> StagedMergeResult:
    """
    Quick staged merge with automatic planning.
    
    Args:
        embedding_paths: Dict of {name: path}
        word: Word to merge
        max_concurrent: Memory constraint
        n_senses: Senses per embedding
        strategy: Merge ordering strategy
        verbose: Print progress
        
    Returns:
        StagedMergeResult
    """
    specs = {name: {"path": path, "format": "glove"} 
             for name, path in embedding_paths.items()}
    
    merger = StagedMerger(specs, max_concurrent, strategy, verbose)
    merger.plan_merge_order(sample_words=[word])
    return merger.merge_staged(word, n_senses)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Staged Embedding Merger")
    print("=" * 50)
    print("\nUsage:")
    print("  from staged_merger import StagedMerger, MergeStrategy")
    print("")
    print("  specs = {")
    print('      "wiki": {"path": "wiki.txt", "format": "glove"},')
    print('      "twitter": {"path": "twitter.txt", "format": "glove"},')
    print("  }")
    print("")
    print("  merger = StagedMerger(specs, max_concurrent=2)")
    print("  plan = merger.plan_merge_order(['bank', 'rock'])")
    print("  result = merger.merge_staged('bank')")
