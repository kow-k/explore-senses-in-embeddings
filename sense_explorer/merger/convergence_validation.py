"""
Cross-Embedding Convergence Validation

This module provides methods to validate whether senses that cluster together
("convergent") truly represent the same meaning across different embeddings.

Key validation methods:
1. Neighbor Overlap - Compare top-k neighbors between convergent senses
2. Anchor Consistency - Check if same anchors work across embeddings  
3. Cross-embedding Projection - Project and match across embedding spaces
4. Semantic Coherence - Measure intra-cluster vs random similarity
5. Convergence Confidence - Combined score for overall validation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import numpy as np
from collections import defaultdict
import warnings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SenseValidation:
    """Validation results for a single sense component."""
    sense_id: str
    source: str
    neighbors: List[str]  # Top neighbors for this sense
    anchor_words: List[str]  # Anchor words used (if available)


@dataclass 
class ClusterValidation:
    """Validation results for a convergent cluster."""
    cluster_id: int
    members: List[str]  # sense_ids in this cluster
    sources: List[str]  # unique sources in this cluster
    is_convergent: bool
    
    # Validation metrics
    neighbor_overlap: float = 0.0  # Jaccard similarity of neighbor sets
    anchor_consistency: float = 0.0  # Agreement on anchor words
    cross_projection_match: bool = False  # Cross-embedding projection validates
    semantic_coherence: float = 0.0  # Intra-cluster similarity
    
    # Combined score
    confidence: float = 0.0
    confidence_level: str = "unknown"  # "high", "medium", "low"
    
    # Detailed info
    shared_neighbors: List[str] = field(default_factory=list)
    shared_anchors: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)


@dataclass
class ConvergenceReport:
    """
    Complete validation report for a merger result.
    """
    word: str
    n_clusters: int
    n_convergent: int
    cluster_validations: List[ClusterValidation]
    
    # Aggregate metrics
    mean_confidence: float = 0.0
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    
    # Method-specific aggregate scores
    mean_neighbor_overlap: float = 0.0
    mean_anchor_consistency: float = 0.0
    mean_semantic_coherence: float = 0.0
    cross_projection_success_rate: float = 0.0
    
    def summary(self, verbose: bool = True) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"CONVERGENCE VALIDATION: '{self.word}'",
            "=" * 60,
            f"Clusters: {self.n_clusters} total, {self.n_convergent} convergent",
            f"Mean confidence: {self.mean_confidence:.3f}",
            f"  High confidence: {self.high_confidence_count}",
            f"  Low confidence: {self.low_confidence_count}",
            "",
            "Aggregate Metrics:",
            f"  Neighbor overlap: {self.mean_neighbor_overlap:.3f}",
            f"  Anchor consistency: {self.mean_anchor_consistency:.3f}",
            f"  Semantic coherence: {self.mean_semantic_coherence:.3f}",
            f"  Cross-projection success: {self.cross_projection_success_rate:.1%}",
        ]
        
        if verbose:
            lines.append("")
            lines.append("-" * 40)
            lines.append("CLUSTER DETAILS")
            lines.append("-" * 40)
            
            for cv in self.cluster_validations:
                if cv.is_convergent:
                    lines.append(f"\nCluster {cv.cluster_id}: CONVERGENT")
                    lines.append(f"  Sources: {', '.join(cv.sources)}")
                    lines.append(f"  Members: {', '.join(cv.members)}")
                    lines.append(f"  Confidence: {cv.confidence:.3f} ({cv.confidence_level})")
                    lines.append(f"    Neighbor overlap: {cv.neighbor_overlap:.3f}")
                    lines.append(f"    Anchor consistency: {cv.anchor_consistency:.3f}")
                    lines.append(f"    Semantic coherence: {cv.semantic_coherence:.3f}")
                    lines.append(f"    Cross-projection: {'✓' if cv.cross_projection_match else '✗'}")
                    if cv.shared_neighbors:
                        lines.append(f"  Shared neighbors: {', '.join(cv.shared_neighbors[:10])}")
                    if cv.shared_anchors:
                        lines.append(f"  Shared anchors: {', '.join(cv.shared_anchors)}")
                    if cv.validation_notes:
                        for note in cv.validation_notes:
                            lines.append(f"  Note: {note}")
                else:
                    lines.append(f"\nCluster {cv.cluster_id}: source-specific ({cv.sources[0]})")
                    lines.append(f"  Members: {', '.join(cv.members)}")
        
        return "\n".join(lines)
    
    @property
    def low_confidence_clusters(self) -> List[ClusterValidation]:
        """Get clusters with low confidence that may be false convergences."""
        return [cv for cv in self.cluster_validations 
                if cv.is_convergent and cv.confidence_level == "low"]
    
    @property
    def high_confidence_clusters(self) -> List[ClusterValidation]:
        """Get clusters with high confidence."""
        return [cv for cv in self.cluster_validations 
                if cv.is_convergent and cv.confidence_level == "high"]


# =============================================================================
# VALIDATION METHODS
# =============================================================================

def compute_neighbor_overlap(
    neighbors_a: List[str],
    neighbors_b: List[str],
    top_k: int = 50
) -> Tuple[float, List[str]]:
    """
    Compute Jaccard similarity between two neighbor lists.
    
    Args:
        neighbors_a: Neighbors of first sense
        neighbors_b: Neighbors of second sense
        top_k: Number of top neighbors to consider
        
    Returns:
        (jaccard_similarity, shared_neighbors)
    """
    set_a = set(neighbors_a[:top_k])
    set_b = set(neighbors_b[:top_k])
    
    if not set_a or not set_b:
        return 0.0, []
    
    intersection = set_a & set_b
    union = set_a | set_b
    
    jaccard = len(intersection) / len(union) if union else 0.0
    shared = list(intersection)
    
    return jaccard, shared


def compute_anchor_consistency(
    anchors_a: List[str],
    anchors_b: List[str]
) -> Tuple[float, List[str]]:
    """
    Compute consistency between anchor word sets.
    
    Args:
        anchors_a: Anchor words from first sense
        anchors_b: Anchor words from second sense
        
    Returns:
        (consistency_score, shared_anchors)
    """
    if not anchors_a or not anchors_b:
        return 0.5, []  # Default when anchors unavailable
    
    set_a = set(anchors_a)
    set_b = set(anchors_b)
    
    intersection = set_a & set_b
    
    # Score based on proportion of shared anchors
    # Use harmonic mean of coverage in both directions
    coverage_a = len(intersection) / len(set_a) if set_a else 0
    coverage_b = len(intersection) / len(set_b) if set_b else 0
    
    if coverage_a + coverage_b > 0:
        consistency = 2 * coverage_a * coverage_b / (coverage_a + coverage_b)
    else:
        consistency = 0.0
    
    return consistency, list(intersection)


def compute_cross_projection_match(
    sense_a: 'SenseComponent',
    sense_b: 'SenseComponent',
    embeddings_a: Dict[str, np.ndarray],
    embeddings_b: Dict[str, np.ndarray],
    shared_vocab: Set[str],
    top_k: int = 20,
    match_threshold: float = 0.5
) -> Tuple[bool, float]:
    """
    Project sense from embedding A to B's space and check for match.
    
    Uses shared vocabulary to create a projection and finds if the
    projected sense's nearest neighbors match the claimed convergent sense.
    
    Args:
        sense_a: Sense from embedding A
        sense_b: Sense from embedding B
        embeddings_a: Embedding A's word vectors
        embeddings_b: Embedding B's word vectors
        shared_vocab: Words present in both embeddings
        top_k: Number of neighbors to compare
        match_threshold: Minimum overlap to count as match
        
    Returns:
        (is_match, overlap_score)
    """
    # Get neighbors of sense_a in embedding A's space
    neighbors_a = []
    for word in shared_vocab:
        if word in embeddings_a:
            sim = np.dot(sense_a.vector, embeddings_a[word])
            neighbors_a.append((word, sim))
    neighbors_a.sort(key=lambda x: -x[1])
    top_neighbors_a = set(w for w, _ in neighbors_a[:top_k])
    
    # Get neighbors of sense_b in embedding B's space
    neighbors_b = []
    for word in shared_vocab:
        if word in embeddings_b:
            sim = np.dot(sense_b.vector, embeddings_b[word])
            neighbors_b.append((word, sim))
    neighbors_b.sort(key=lambda x: -x[1])
    top_neighbors_b = set(w for w, _ in neighbors_b[:top_k])
    
    # Check overlap
    if not top_neighbors_a or not top_neighbors_b:
        return False, 0.0
    
    overlap = len(top_neighbors_a & top_neighbors_b) / top_k
    is_match = overlap >= match_threshold
    
    return is_match, overlap


def compute_semantic_coherence(
    cluster_senses: List['SenseComponent'],
    similarity_matrix: np.ndarray,
    sense_id_to_idx: Dict[str, int],
    n_random_samples: int = 100
) -> float:
    """
    Compute semantic coherence of a cluster compared to random baseline.
    
    Args:
        cluster_senses: List of sense components in the cluster
        similarity_matrix: Full similarity matrix from merger
        sense_id_to_idx: Mapping from sense_id to matrix index
        n_random_samples: Number of random pairs to sample for baseline
        
    Returns:
        Coherence score (0-1, higher = more coherent than random)
    """
    if len(cluster_senses) < 2:
        return 1.0  # Single-member cluster is trivially coherent
    
    # Compute mean intra-cluster similarity
    intra_sims = []
    for i, s1 in enumerate(cluster_senses):
        for j, s2 in enumerate(cluster_senses):
            if i < j:
                idx1 = sense_id_to_idx.get(s1.sense_id)
                idx2 = sense_id_to_idx.get(s2.sense_id)
                if idx1 is not None and idx2 is not None:
                    intra_sims.append(similarity_matrix[idx1, idx2])
    
    if not intra_sims:
        return 0.5
    
    mean_intra = np.mean(intra_sims)
    
    # Compute random baseline (sample pairs from different clusters)
    n = similarity_matrix.shape[0]
    random_sims = []
    for _ in range(n_random_samples):
        i, j = np.random.randint(0, n, 2)
        if i != j:
            random_sims.append(similarity_matrix[i, j])
    
    if not random_sims:
        return 0.5
    
    mean_random = np.mean(random_sims)
    std_random = np.std(random_sims) + 1e-8
    
    # Z-score style coherence (clamped to 0-1)
    z_score = (mean_intra - mean_random) / std_random
    coherence = 1 / (1 + np.exp(-z_score))  # Sigmoid to 0-1
    
    return float(coherence)


def compute_confidence(
    neighbor_overlap: float,
    anchor_consistency: float,
    cross_projection_match: bool,
    semantic_coherence: float,
    weights: Dict[str, float] = None
) -> Tuple[float, str]:
    """
    Compute overall confidence score from individual metrics.
    
    Args:
        neighbor_overlap: Jaccard similarity of neighbors (0-1)
        anchor_consistency: Anchor agreement score (0-1)
        cross_projection_match: Whether projection validates
        semantic_coherence: Coherence score (0-1)
        weights: Optional custom weights for each metric
        
    Returns:
        (confidence_score, confidence_level)
    """
    if weights is None:
        weights = {
            'neighbor_overlap': 0.30,
            'anchor_consistency': 0.20,
            'cross_projection': 0.25,
            'semantic_coherence': 0.25
        }
    
    cross_proj_score = 1.0 if cross_projection_match else 0.0
    
    confidence = (
        weights['neighbor_overlap'] * neighbor_overlap +
        weights['anchor_consistency'] * anchor_consistency +
        weights['cross_projection'] * cross_proj_score +
        weights['semantic_coherence'] * semantic_coherence
    )
    
    # Determine level
    if confidence >= 0.7:
        level = "high"
    elif confidence >= 0.4:
        level = "medium"
    else:
        level = "low"
    
    return confidence, level


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_convergence(
    result: 'MergerResult',
    explorers: Dict[str, 'SenseExplorer'],
    neighbor_k: int = 50,
    projection_threshold: float = 0.3,
    verbose: bool = False
) -> ConvergenceReport:
    """
    Validate convergent clusters from a merger result.
    
    Args:
        result: MergerResult or WeightedMergerResult from merge operation
        explorers: Dict of SenseExplorer instances used in merge
        neighbor_k: Number of neighbors to consider for overlap
        projection_threshold: Threshold for cross-projection match
        verbose: Print progress
        
    Returns:
        ConvergenceReport with validation results for all clusters
    """
    if verbose:
        print(f"\nValidating convergence for '{result.word}'...")
    
    # Build sense lookup
    sense_by_id = {s.sense_id: s for s in result.sense_components}
    sense_id_to_idx = {s.sense_id: i for i, s in enumerate(result.sense_components)}
    
    # Get shared vocabulary across all embeddings
    shared_vocab = None
    for name, se in explorers.items():
        vocab = set(se.embeddings.keys()) if hasattr(se, 'embeddings') else set(se.vocab)
        if shared_vocab is None:
            shared_vocab = vocab
        else:
            shared_vocab &= vocab
    
    if verbose:
        print(f"  Shared vocabulary: {len(shared_vocab):,} words")
    
    # Build cluster membership
    cluster_members = defaultdict(list)
    for sense_id, cluster_id in result.clusters.items():
        cluster_members[cluster_id].append(sense_id)
    
    cluster_validations = []
    
    for cluster_id, members in cluster_members.items():
        senses = [sense_by_id[m] for m in members]
        sources = list(set(s.source for s in senses))
        is_convergent = len(sources) > 1
        
        cv = ClusterValidation(
            cluster_id=cluster_id,
            members=members,
            sources=sources,
            is_convergent=is_convergent
        )
        
        if is_convergent:
            if verbose:
                print(f"  Validating cluster {cluster_id} ({len(members)} members, {len(sources)} sources)")
            
            # === 1. Neighbor Overlap ===
            # Compare all pairs across sources
            neighbor_overlaps = []
            all_shared_neighbors = set()
            
            for i, s1 in enumerate(senses):
                for j, s2 in enumerate(senses):
                    if i < j and s1.source != s2.source:
                        # Get neighbors
                        n1 = [w for w, _ in (s1.top_neighbors or [])[:neighbor_k]]
                        n2 = [w for w, _ in (s2.top_neighbors or [])[:neighbor_k]]
                        
                        overlap, shared = compute_neighbor_overlap(n1, n2, neighbor_k)
                        neighbor_overlaps.append(overlap)
                        all_shared_neighbors.update(shared)
            
            cv.neighbor_overlap = np.mean(neighbor_overlaps) if neighbor_overlaps else 0.0
            cv.shared_neighbors = list(all_shared_neighbors)[:20]  # Top 20
            
            # === 2. Anchor Consistency ===
            # Check if senses use similar anchors
            anchor_consistencies = []
            all_shared_anchors = set()
            
            for i, s1 in enumerate(senses):
                for j, s2 in enumerate(senses):
                    if i < j and s1.source != s2.source:
                        # Extract anchor info from sense_id if available
                        # Format: "source_anchorname" or "source_sense_N"
                        anchor1 = _extract_anchor_name(s1.sense_id)
                        anchor2 = _extract_anchor_name(s2.sense_id)
                        
                        if anchor1 and anchor2:
                            # Simple string match for anchor names
                            if anchor1 == anchor2:
                                anchor_consistencies.append(1.0)
                                all_shared_anchors.add(anchor1)
                            else:
                                anchor_consistencies.append(0.0)
                        else:
                            anchor_consistencies.append(0.5)  # Unknown
            
            cv.anchor_consistency = np.mean(anchor_consistencies) if anchor_consistencies else 0.5
            cv.shared_anchors = list(all_shared_anchors)
            
            # === 3. Cross-Projection Match ===
            projection_matches = []
            
            for i, s1 in enumerate(senses):
                for j, s2 in enumerate(senses):
                    if i < j and s1.source != s2.source:
                        emb1 = explorers[s1.source].embeddings
                        emb2 = explorers[s2.source].embeddings
                        
                        is_match, overlap = compute_cross_projection_match(
                            s1, s2, emb1, emb2, shared_vocab,
                            top_k=neighbor_k // 2,
                            match_threshold=projection_threshold
                        )
                        projection_matches.append(is_match)
            
            cv.cross_projection_match = all(projection_matches) if projection_matches else False
            
            # === 4. Semantic Coherence ===
            cv.semantic_coherence = compute_semantic_coherence(
                senses,
                result.similarity_matrix,
                sense_id_to_idx
            )
            
            # === 5. Compute Confidence ===
            cv.confidence, cv.confidence_level = compute_confidence(
                cv.neighbor_overlap,
                cv.anchor_consistency,
                cv.cross_projection_match,
                cv.semantic_coherence
            )
            
            # Add validation notes
            if cv.neighbor_overlap < 0.2:
                cv.validation_notes.append("Low neighbor overlap - senses may capture different contexts")
            if cv.anchor_consistency < 0.3:
                cv.validation_notes.append("Different anchors used - may be related but distinct senses")
            if not cv.cross_projection_match:
                cv.validation_notes.append("Cross-projection failed - geometric relationship differs across embeddings")
            if cv.semantic_coherence < 0.5:
                cv.validation_notes.append("Low coherence - cluster may be an artifact of threshold choice")
        
        cluster_validations.append(cv)
    
    # Build report
    convergent_validations = [cv for cv in cluster_validations if cv.is_convergent]
    
    report = ConvergenceReport(
        word=result.word,
        n_clusters=len(cluster_validations),
        n_convergent=len(convergent_validations),
        cluster_validations=cluster_validations
    )
    
    # Compute aggregates
    if convergent_validations:
        report.mean_confidence = np.mean([cv.confidence for cv in convergent_validations])
        report.high_confidence_count = sum(1 for cv in convergent_validations if cv.confidence_level == "high")
        report.low_confidence_count = sum(1 for cv in convergent_validations if cv.confidence_level == "low")
        report.mean_neighbor_overlap = np.mean([cv.neighbor_overlap for cv in convergent_validations])
        report.mean_anchor_consistency = np.mean([cv.anchor_consistency for cv in convergent_validations])
        report.mean_semantic_coherence = np.mean([cv.semantic_coherence for cv in convergent_validations])
        report.cross_projection_success_rate = np.mean([cv.cross_projection_match for cv in convergent_validations])
    
    if verbose:
        print(f"  Validation complete: {report.high_confidence_count} high, "
              f"{report.low_confidence_count} low confidence")
    
    return report


def _extract_anchor_name(sense_id: str) -> Optional[str]:
    """
    Extract anchor name from sense_id.
    
    Expected formats:
    - "wiki_financial" -> "financial"
    - "twitter_sense_0" -> "sense_0" 
    - "embedding_anchor" -> "anchor"
    """
    parts = sense_id.split('_', 1)
    if len(parts) >= 2:
        return parts[1]
    return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validate(
    result: 'MergerResult',
    explorers: Dict[str, 'SenseExplorer']
) -> Dict[str, Any]:
    """
    Quick validation returning just key metrics.
    
    Returns:
        Dict with mean_confidence, high_count, low_count, flags
    """
    report = validate_convergence(result, explorers, verbose=False)
    
    return {
        'word': result.word,
        'n_convergent': report.n_convergent,
        'mean_confidence': report.mean_confidence,
        'high_confidence': report.high_confidence_count,
        'low_confidence': report.low_confidence_count,
        'mean_neighbor_overlap': report.mean_neighbor_overlap,
        'mean_semantic_coherence': report.mean_semantic_coherence,
        'any_low_confidence': report.low_confidence_count > 0,
        'all_high_confidence': report.low_confidence_count == 0 and report.high_confidence_count == report.n_convergent
    }


def validate_multiple(
    results: List['MergerResult'],
    explorers: Dict[str, 'SenseExplorer'],
    verbose: bool = True
) -> List[ConvergenceReport]:
    """
    Validate multiple merger results.
    
    Args:
        results: List of MergerResult objects
        explorers: Dict of SenseExplorer instances
        verbose: Print progress
        
    Returns:
        List of ConvergenceReport objects
    """
    reports = []
    
    for i, result in enumerate(results):
        if verbose:
            print(f"\n[{i+1}/{len(results)}] Validating '{result.word}'...")
        
        report = validate_convergence(result, explorers, verbose=False)
        reports.append(report)
        
        if verbose:
            print(f"  Confidence: {report.mean_confidence:.3f} "
                  f"({report.high_confidence_count} high, {report.low_confidence_count} low)")
    
    return reports


def summarize_validations(reports: List[ConvergenceReport]) -> str:
    """
    Generate aggregate summary across multiple validation reports.
    """
    if not reports:
        return "No reports to summarize"
    
    total_convergent = sum(r.n_convergent for r in reports)
    total_high = sum(r.high_confidence_count for r in reports)
    total_low = sum(r.low_confidence_count for r in reports)
    
    mean_confidence = np.mean([r.mean_confidence for r in reports if r.n_convergent > 0])
    mean_overlap = np.mean([r.mean_neighbor_overlap for r in reports if r.n_convergent > 0])
    mean_coherence = np.mean([r.mean_semantic_coherence for r in reports if r.n_convergent > 0])
    
    lines = [
        "=" * 60,
        "VALIDATION SUMMARY",
        "=" * 60,
        f"Words validated: {len(reports)}",
        f"Total convergent clusters: {total_convergent}",
        f"  High confidence: {total_high} ({total_high/max(1,total_convergent):.1%})",
        f"  Low confidence: {total_low} ({total_low/max(1,total_convergent):.1%})",
        "",
        "Mean metrics across words:",
        f"  Confidence: {mean_confidence:.3f}",
        f"  Neighbor overlap: {mean_overlap:.3f}",
        f"  Semantic coherence: {mean_coherence:.3f}",
    ]
    
    # List words with low confidence convergences
    low_conf_words = [r.word for r in reports if r.low_confidence_count > 0]
    if low_conf_words:
        lines.append("")
        lines.append(f"Words with low-confidence convergences ({len(low_conf_words)}):")
        for word in low_conf_words[:10]:
            lines.append(f"  - {word}")
        if len(low_conf_words) > 10:
            lines.append(f"  ... and {len(low_conf_words) - 10} more")
    
    return "\n".join(lines)
