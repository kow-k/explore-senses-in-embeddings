"""
Register-Specific Sense Profiles

This module characterizes how word senses differ across registers (e.g., formal
Wikipedia vs informal Twitter). It analyzes sense prevalence, register-specific
neighbors, sense drift, and register signatures.

Key insights this module can reveal:
- Which senses dominate in each register
- How the same sense has different associations across registers
- Geometric drift of senses between formal and informal usage
- Register-specific vocabulary patterns
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
class SensePrevalence:
    """Prevalence of a sense across registers."""
    sense_name: str
    prevalence_by_register: Dict[str, float]  # register -> prevalence (0-1)
    dominant_register: str  # which register this sense is strongest in
    specificity: float  # how register-specific (0 = universal, 1 = single-register)
    
    def __repr__(self):
        prev_str = ", ".join(f"{r}={p:.2f}" for r, p in self.prevalence_by_register.items())
        return f"SensePrevalence({self.sense_name}: {prev_str}, specificity={self.specificity:.2f})"


@dataclass
class RegisterNeighbors:
    """Neighbors of a sense in different registers."""
    sense_name: str
    neighbors_by_register: Dict[str, List[Tuple[str, float]]]  # register -> [(word, sim), ...]
    shared_neighbors: List[str]  # neighbors common to all registers
    unique_neighbors: Dict[str, List[str]]  # register -> unique neighbors
    overlap_score: float  # Jaccard similarity of neighbor sets
    
    def get_top_neighbors(self, register: str, n: int = 10) -> List[str]:
        """Get top n neighbors for a register."""
        if register not in self.neighbors_by_register:
            return []
        return [w for w, _ in self.neighbors_by_register[register][:n]]


@dataclass
class SenseDrift:
    """Drift of a sense between registers."""
    sense_name: str
    drift_scores: Dict[Tuple[str, str], float]  # (reg1, reg2) -> drift score
    mean_drift: float
    max_drift: float
    most_different_pair: Tuple[str, str]  # which register pair has highest drift
    
    def get_drift(self, register1: str, register2: str) -> float:
        """Get drift between two registers."""
        key = (register1, register2) if (register1, register2) in self.drift_scores else (register2, register1)
        return self.drift_scores.get(key, 0.0)


@dataclass
class RegisterSignature:
    """Characteristic features of a register for a word."""
    register: str
    word: str
    dominant_senses: List[str]  # senses that are stronger in this register
    characteristic_neighbors: List[str]  # neighbors unique to this register
    formality_indicators: Dict[str, float]  # linguistic markers
    

@dataclass
class RegisterProfile:
    """
    Complete profile of how a word's senses differ across registers.
    """
    word: str
    registers: List[str]
    
    # Core analyses
    sense_prevalences: List[SensePrevalence]
    register_neighbors: Dict[str, RegisterNeighbors]  # sense -> neighbors
    sense_drifts: Dict[str, SenseDrift]  # sense -> drift
    register_signatures: Dict[str, RegisterSignature]  # register -> signature
    
    # Aggregate metrics
    overall_register_similarity: float  # how similar the word is across registers
    n_shared_senses: int  # senses present in all registers
    n_register_specific_senses: int  # senses unique to one register
    
    # Raw data
    sense_vectors: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    # sense_name -> register -> vector
    
    @property
    def prevalence(self) -> Dict[str, Dict[str, float]]:
        """Get prevalence as nested dict: sense -> register -> prevalence."""
        result = {}
        for sp in self.sense_prevalences:
            result[sp.sense_name] = sp.prevalence_by_register.copy()
        return result
    
    def neighbors_by_register(self, sense_name: str) -> Dict[str, List[str]]:
        """Get top neighbors for a sense in each register."""
        if sense_name not in self.register_neighbors:
            return {}
        rn = self.register_neighbors[sense_name]
        return {reg: [w for w, _ in neighbors[:10]] 
                for reg, neighbors in rn.neighbors_by_register.items()}
    
    @property
    def sense_drift(self) -> Dict[str, float]:
        """Get mean drift for each sense."""
        return {name: sd.mean_drift for name, sd in self.sense_drifts.items()}
    
    def report(self, verbose: bool = True) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            f"REGISTER PROFILE: '{self.word}'",
            "=" * 70,
            f"Registers: {', '.join(self.registers)}",
            f"Overall register similarity: {self.overall_register_similarity:.3f}",
            f"Shared senses: {self.n_shared_senses}",
            f"Register-specific senses: {self.n_register_specific_senses}",
        ]
        
        # Sense prevalence
        lines.append("\n" + "-" * 40)
        lines.append("SENSE PREVALENCE")
        lines.append("-" * 40)
        
        for sp in sorted(self.sense_prevalences, key=lambda x: -max(x.prevalence_by_register.values())):
            prev_parts = [f"{r}={p:.2f}" for r, p in sp.prevalence_by_register.items()]
            lines.append(f"  {sp.sense_name}: {', '.join(prev_parts)}")
            lines.append(f"    Dominant: {sp.dominant_register}, Specificity: {sp.specificity:.2f}")
        
        # Sense drift
        lines.append("\n" + "-" * 40)
        lines.append("SENSE DRIFT (between registers)")
        lines.append("-" * 40)
        
        for sense_name, sd in sorted(self.sense_drifts.items(), key=lambda x: -x[1].mean_drift):
            lines.append(f"  {sense_name}: mean={sd.mean_drift:.3f}, max={sd.max_drift:.3f}")
            if verbose and sd.most_different_pair:
                lines.append(f"    Most different: {sd.most_different_pair[0]} ↔ {sd.most_different_pair[1]}")
        
        # Register-specific neighbors
        if verbose:
            lines.append("\n" + "-" * 40)
            lines.append("REGISTER-SPECIFIC NEIGHBORS")
            lines.append("-" * 40)
            
            for sense_name, rn in self.register_neighbors.items():
                lines.append(f"\n  {sense_name}:")
                lines.append(f"    Shared: {', '.join(rn.shared_neighbors[:5])}")
                for reg, unique in rn.unique_neighbors.items():
                    if unique:
                        lines.append(f"    {reg} only: {', '.join(unique[:5])}")
        
        # Register signatures
        if verbose:
            lines.append("\n" + "-" * 40)
            lines.append("REGISTER SIGNATURES")
            lines.append("-" * 40)
            
            for reg, sig in self.register_signatures.items():
                lines.append(f"\n  {reg}:")
                if sig.dominant_senses:
                    lines.append(f"    Dominant senses: {', '.join(sig.dominant_senses)}")
                if sig.characteristic_neighbors:
                    lines.append(f"    Characteristic words: {', '.join(sig.characteristic_neighbors[:8])}")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return f"RegisterProfile('{self.word}', registers={self.registers}, similarity={self.overall_register_similarity:.3f})"


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_sense_prevalence(
    word: str,
    explorers: Dict[str, 'SenseExplorer'],
    n_senses: int = 2
) -> List[SensePrevalence]:
    """
    Compute sense prevalence across registers.
    
    Prevalence is estimated from the projection coefficients in the
    sense decomposition: w = α₁s₁ + α₂s₂ + ...
    
    Args:
        word: Target word
        explorers: Dict mapping register names to SenseExplorer instances
        n_senses: Number of senses to extract
        
    Returns:
        List of SensePrevalence objects
    """
    register_senses = {}
    
    for reg_name, se in explorers.items():
        if word not in se.vocab:
            continue
        
        # Extract senses - returns Dict[str, np.ndarray]
        senses = se.induce_senses(word, n_senses=n_senses)
        
        # Get coefficients from localization - returns SenseDecomposition dataclass
        try:
            localization = se.localize_senses(word)
            # Access as attributes, not dict
            coefficients = getattr(localization, 'coefficients', {})
            r_squared = getattr(localization, 'total_r_squared', 0)
        except Exception:
            # Fallback if localization fails
            coefficients = {}
            r_squared = 0
        
        # Store sense info with coefficients
        register_senses[reg_name] = {
            'senses': senses,  # Dict[str, np.ndarray]
            'coefficients': coefficients if isinstance(coefficients, dict) else {},
            'r_squared': r_squared
        }
    
    # Align senses across registers by name
    all_sense_names = set()
    for reg_data in register_senses.values():
        # senses is Dict[str, np.ndarray], so keys() gives sense names
        all_sense_names.update(reg_data['senses'].keys())
    
    prevalences = []
    
    for sense_name in all_sense_names:
        prev_by_reg = {}
        
        for reg_name, reg_data in register_senses.items():
            # Find matching sense
            coef = 0.0
            coeffs = reg_data['coefficients']
            total_coef = sum(abs(c) for c in coeffs.values()) if coeffs else 0
            
            if sense_name in reg_data['senses'] and total_coef > 0:
                coef = abs(coeffs.get(sense_name, 0))
            
            # Prevalence = relative coefficient magnitude
            prev_by_reg[reg_name] = coef / total_coef if total_coef > 0 else 0.0
        
        # Compute specificity
        values = list(prev_by_reg.values())
        if len(values) > 1 and max(values) > 0:
            # Gini-like measure: 1 - (uniform distribution)
            mean_val = np.mean(values)
            specificity = np.std(values) / (mean_val + 1e-10)
            specificity = min(1.0, specificity)  # Cap at 1
        else:
            specificity = 0.0
        
        # Find dominant register
        dominant = max(prev_by_reg.keys(), key=lambda r: prev_by_reg[r]) if prev_by_reg else ""
        
        prevalences.append(SensePrevalence(
            sense_name=sense_name,
            prevalence_by_register=prev_by_reg,
            dominant_register=dominant,
            specificity=specificity
        ))
    
    return prevalences


def compute_register_neighbors(
    word: str,
    explorers: Dict[str, 'SenseExplorer'],
    n_senses: int = 2,
    top_k: int = 50
) -> Dict[str, RegisterNeighbors]:
    """
    Compute neighbors for each sense in each register.
    
    Args:
        word: Target word
        explorers: Dict mapping register names to SenseExplorer instances
        n_senses: Number of senses to extract
        top_k: Number of neighbors to retrieve
        
    Returns:
        Dict mapping sense names to RegisterNeighbors
    """
    # Extract senses and their neighbors from each register
    register_sense_neighbors = {}  # reg -> sense_name -> [(word, sim), ...]
    
    for reg_name, se in explorers.items():
        if word not in se.vocab:
            continue
        
        # senses is Dict[str, np.ndarray]
        senses = se.induce_senses(word, n_senses=n_senses)
        register_sense_neighbors[reg_name] = {}
        
        for sense_name, sense_vec in senses.items():
            if sense_vec is not None:
                # Find neighbors
                neighbors = []
                sense_dim = len(sense_vec)
                for w in se.vocab:
                    if w != word:
                        w_vec = se.embeddings[w]
                        # Handle dimension mismatch
                        min_dim = min(sense_dim, len(w_vec))
                        sim = float(np.dot(sense_vec[:min_dim], w_vec[:min_dim]))
                        neighbors.append((w, sim))
                neighbors.sort(key=lambda x: -x[1])
                register_sense_neighbors[reg_name][sense_name] = neighbors[:top_k]
    
    # Build RegisterNeighbors for each sense
    all_sense_names = set()
    for reg_data in register_sense_neighbors.values():
        all_sense_names.update(reg_data.keys())
    
    result = {}
    
    for sense_name in all_sense_names:
        neighbors_by_reg = {}
        neighbor_sets = {}
        
        for reg_name, reg_data in register_sense_neighbors.items():
            if sense_name in reg_data:
                neighbors_by_reg[reg_name] = reg_data[sense_name]
                neighbor_sets[reg_name] = set(w for w, _ in reg_data[sense_name][:top_k])
        
        # Compute shared and unique
        if neighbor_sets:
            all_sets = list(neighbor_sets.values())
            shared = set.intersection(*all_sets) if len(all_sets) > 1 else all_sets[0]
            
            unique = {}
            for reg, ns in neighbor_sets.items():
                other_sets = [s for r, s in neighbor_sets.items() if r != reg]
                if other_sets:
                    others_union = set.union(*other_sets)
                    unique[reg] = list(ns - others_union)[:20]
                else:
                    unique[reg] = []
            
            # Jaccard overlap
            if len(all_sets) >= 2:
                union = set.union(*all_sets)
                overlap = len(shared) / len(union) if union else 0.0
            else:
                overlap = 1.0
        else:
            shared = set()
            unique = {}
            overlap = 0.0
        
        result[sense_name] = RegisterNeighbors(
            sense_name=sense_name,
            neighbors_by_register=neighbors_by_reg,
            shared_neighbors=list(shared)[:20],
            unique_neighbors=unique,
            overlap_score=overlap
        )
    
    return result


def compute_sense_drift(
    word: str,
    explorers: Dict[str, 'SenseExplorer'],
    n_senses: int = 2,
    shared_vocab: Set[str] = None
) -> Dict[str, SenseDrift]:
    """
    Compute drift of senses between registers.
    
    Drift is measured as 1 - cosine_similarity between sense vectors
    projected onto shared vocabulary basis.
    
    Args:
        word: Target word
        explorers: Dict mapping register names to SenseExplorer instances
        n_senses: Number of senses to extract
        shared_vocab: Shared vocabulary for projection (computed if None)
        
    Returns:
        Dict mapping sense names to SenseDrift
    """
    # Get shared vocabulary
    if shared_vocab is None:
        shared_vocab = None
        for se in explorers.values():
            if shared_vocab is None:
                shared_vocab = set(se.vocab)
            else:
                shared_vocab &= set(se.vocab)
    
    shared_vocab_list = sorted(list(shared_vocab))[:500]  # Limit for efficiency
    
    # Extract senses from each register
    register_senses = {}  # reg -> sense_name -> vector
    
    for reg_name, se in explorers.items():
        if word not in se.vocab:
            continue
        
        # senses is Dict[str, np.ndarray]
        senses = se.induce_senses(word, n_senses=n_senses)
        register_senses[reg_name] = {}
        
        for sense_name, sense_vec in senses.items():
            if sense_vec is not None:
                # Project onto shared vocabulary basis
                projection = []
                sense_dim = len(sense_vec)
                for w in shared_vocab_list:
                    if w in se.embeddings:
                        w_vec = se.embeddings[w]
                        # Handle dimension mismatch
                        min_dim = min(sense_dim, len(w_vec))
                        sim = float(np.dot(sense_vec[:min_dim], w_vec[:min_dim]))
                        projection.append(sim)
                    else:
                        projection.append(0.0)
                
                proj_vec = np.array(projection)
                norm = np.linalg.norm(proj_vec)
                if norm > 1e-10:
                    proj_vec = proj_vec / norm
                
                register_senses[reg_name][sense_name] = proj_vec
    
    # Compute drift for each sense
    all_sense_names = set()
    for reg_data in register_senses.values():
        all_sense_names.update(reg_data.keys())
    
    result = {}
    registers = list(explorers.keys())
    
    for sense_name in all_sense_names:
        drift_scores = {}
        
        # Compare all register pairs
        for i, reg1 in enumerate(registers):
            for reg2 in registers[i+1:]:
                if reg1 in register_senses and reg2 in register_senses:
                    if sense_name in register_senses[reg1] and sense_name in register_senses[reg2]:
                        vec1 = register_senses[reg1][sense_name]
                        vec2 = register_senses[reg2][sense_name]
                        
                        # Drift = 1 - cosine similarity
                        sim = float(np.dot(vec1, vec2))
                        drift = 1.0 - sim
                        drift_scores[(reg1, reg2)] = max(0.0, drift)  # Clamp negative
        
        if drift_scores:
            values = list(drift_scores.values())
            mean_drift = np.mean(values)
            max_drift = np.max(values)
            most_diff = max(drift_scores.keys(), key=lambda k: drift_scores[k])
        else:
            mean_drift = 0.0
            max_drift = 0.0
            most_diff = ("", "")
        
        result[sense_name] = SenseDrift(
            sense_name=sense_name,
            drift_scores=drift_scores,
            mean_drift=mean_drift,
            max_drift=max_drift,
            most_different_pair=most_diff
        )
    
    return result


def compute_register_signatures(
    word: str,
    explorers: Dict[str, 'SenseExplorer'],
    prevalences: List[SensePrevalence],
    register_neighbors: Dict[str, RegisterNeighbors]
) -> Dict[str, RegisterSignature]:
    """
    Compute characteristic signatures for each register.
    
    Args:
        word: Target word
        explorers: Dict mapping register names to SenseExplorer instances
        prevalences: Pre-computed sense prevalences
        register_neighbors: Pre-computed register neighbors
        
    Returns:
        Dict mapping register names to RegisterSignature
    """
    result = {}
    
    for reg_name in explorers.keys():
        # Find dominant senses (higher prevalence in this register)
        dominant_senses = []
        for sp in prevalences:
            if sp.dominant_register == reg_name and sp.specificity > 0.2:
                dominant_senses.append(sp.sense_name)
        
        # Collect characteristic neighbors (unique to this register)
        characteristic = set()
        for rn in register_neighbors.values():
            if reg_name in rn.unique_neighbors:
                characteristic.update(rn.unique_neighbors[reg_name][:10])
        
        # Simple formality indicators (could be expanded)
        formality = {}
        # This could be extended with linguistic features
        
        result[reg_name] = RegisterSignature(
            register=reg_name,
            word=word,
            dominant_senses=dominant_senses,
            characteristic_neighbors=list(characteristic)[:20],
            formality_indicators=formality
        )
    
    return result


# =============================================================================
# MAIN PROFILE FUNCTION
# =============================================================================

def create_register_profile(
    word: str,
    explorers: Dict[str, 'SenseExplorer'],
    n_senses: int = 2,
    verbose: bool = False
) -> RegisterProfile:
    """
    Create a complete register profile for a word.
    
    Args:
        word: Target word
        explorers: Dict mapping register names to SenseExplorer instances
        n_senses: Number of senses to extract per register
        verbose: Print progress
        
    Returns:
        RegisterProfile with complete analysis
    """
    if verbose:
        print(f"\nCreating register profile for '{word}'...")
    
    registers = list(explorers.keys())
    
    # Check word exists in all registers
    missing = [r for r, se in explorers.items() if word not in se.vocab]
    if missing:
        warnings.warn(f"Word '{word}' not found in: {missing}")
    
    # Compute shared vocabulary
    shared_vocab = None
    for se in explorers.values():
        if shared_vocab is None:
            shared_vocab = set(se.vocab)
        else:
            shared_vocab &= set(se.vocab)
    
    if verbose:
        print(f"  Shared vocabulary: {len(shared_vocab):,} words")
    
    # 1. Sense prevalence
    if verbose:
        print("  Computing sense prevalence...")
    prevalences = compute_sense_prevalence(word, explorers, n_senses)
    
    # 2. Register-specific neighbors
    if verbose:
        print("  Computing register-specific neighbors...")
    reg_neighbors = compute_register_neighbors(word, explorers, n_senses)
    
    # 3. Sense drift
    if verbose:
        print("  Computing sense drift...")
    drifts = compute_sense_drift(word, explorers, n_senses, shared_vocab)
    
    # 4. Register signatures
    if verbose:
        print("  Computing register signatures...")
    signatures = compute_register_signatures(word, explorers, prevalences, reg_neighbors)
    
    # Aggregate metrics
    # Overall similarity: average of (1 - drift)
    all_drifts = [sd.mean_drift for sd in drifts.values() if sd.mean_drift > 0]
    overall_sim = 1.0 - np.mean(all_drifts) if all_drifts else 1.0
    
    # Count shared vs specific senses
    n_shared = sum(1 for sp in prevalences 
                   if all(sp.prevalence_by_register.get(r, 0) > 0.1 for r in registers))
    n_specific = sum(1 for sp in prevalences if sp.specificity > 0.5)
    
    # Store sense vectors for later analysis
    sense_vectors = {}
    for sense_name in drifts.keys():
        sense_vectors[sense_name] = {}
        for reg_name, se in explorers.items():
            if word in se.vocab:
                # senses is Dict[str, np.ndarray]
                senses = se.induce_senses(word, n_senses=n_senses)
                if sense_name in senses:
                    sense_vectors[sense_name][reg_name] = senses[sense_name]
    
    profile = RegisterProfile(
        word=word,
        registers=registers,
        sense_prevalences=prevalences,
        register_neighbors=reg_neighbors,
        sense_drifts=drifts,
        register_signatures=signatures,
        overall_register_similarity=overall_sim,
        n_shared_senses=n_shared,
        n_register_specific_senses=n_specific,
        sense_vectors=sense_vectors
    )
    
    if verbose:
        print(f"  Done. Overall similarity: {overall_sim:.3f}")
    
    return profile


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_registers(
    words: List[str],
    explorers: Dict[str, 'SenseExplorer'],
    n_senses: int = 2,
    verbose: bool = True
) -> Dict[str, RegisterProfile]:
    """
    Create register profiles for multiple words.
    
    Args:
        words: List of target words
        explorers: Dict mapping register names to SenseExplorer instances
        n_senses: Number of senses per word
        verbose: Print progress
        
    Returns:
        Dict mapping words to RegisterProfile
    """
    profiles = {}
    
    for i, word in enumerate(words):
        if verbose:
            print(f"\n[{i+1}/{len(words)}] Processing '{word}'...")
        
        # Check word exists
        available = [r for r, se in explorers.items() if word in se.vocab]
        if len(available) < 2:
            if verbose:
                print(f"  Skipping - only in {len(available)} register(s)")
            continue
        
        profile = create_register_profile(word, explorers, n_senses, verbose=False)
        profiles[word] = profile
        
        if verbose:
            print(f"  Similarity: {profile.overall_register_similarity:.3f}, "
                  f"Shared senses: {profile.n_shared_senses}, "
                  f"Register-specific: {profile.n_register_specific_senses}")
    
    return profiles


def summarize_register_comparison(profiles: Dict[str, RegisterProfile]) -> str:
    """
    Generate summary of register comparison across multiple words.
    """
    if not profiles:
        return "No profiles to summarize"
    
    lines = [
        "=" * 70,
        "REGISTER COMPARISON SUMMARY",
        "=" * 70,
        f"Words analyzed: {len(profiles)}",
    ]
    
    # Aggregate statistics
    similarities = [p.overall_register_similarity for p in profiles.values()]
    n_shared = [p.n_shared_senses for p in profiles.values()]
    n_specific = [p.n_register_specific_senses for p in profiles.values()]
    
    lines.extend([
        "",
        "Aggregate Metrics:",
        f"  Mean register similarity: {np.mean(similarities):.3f} (std={np.std(similarities):.3f})",
        f"  Mean shared senses: {np.mean(n_shared):.1f}",
        f"  Mean register-specific senses: {np.mean(n_specific):.1f}",
    ])
    
    # Most/least similar words
    sorted_by_sim = sorted(profiles.items(), key=lambda x: x[1].overall_register_similarity)
    
    lines.extend([
        "",
        "Most register-consistent words:",
    ])
    for word, profile in sorted_by_sim[-5:]:
        lines.append(f"  {word}: similarity={profile.overall_register_similarity:.3f}")
    
    lines.extend([
        "",
        "Most register-divergent words:",
    ])
    for word, profile in sorted_by_sim[:5]:
        lines.append(f"  {word}: similarity={profile.overall_register_similarity:.3f}")
    
    # Sense drift statistics
    all_drifts = []
    for profile in profiles.values():
        for sd in profile.sense_drifts.values():
            all_drifts.append(sd.mean_drift)
    
    if all_drifts:
        lines.extend([
            "",
            "Sense Drift Statistics:",
            f"  Mean drift: {np.mean(all_drifts):.3f}",
            f"  Max drift: {np.max(all_drifts):.3f}",
            f"  Min drift: {np.min(all_drifts):.3f}",
        ])
    
    return "\n".join(lines)


# =============================================================================
# WORD CLUSTERING BY REGISTER SPECIFICITY
# =============================================================================

@dataclass
class RegisterSpecificityCluster:
    """A cluster of words with similar register specificity patterns."""
    cluster_id: int
    label: str  # e.g., "register-consistent", "register-divergent"
    words: List[str]
    mean_similarity: float
    mean_drift: float
    characteristic_pattern: str  # description of the pattern


@dataclass 
class RegisterSpecificityAnalysis:
    """Analysis of words clustered by register specificity."""
    clusters: List[RegisterSpecificityCluster]
    word_to_cluster: Dict[str, int]
    profiles: Dict[str, RegisterProfile]
    
    # Aggregate statistics
    n_consistent: int  # words with high cross-register similarity
    n_moderate: int    # words with moderate similarity
    n_divergent: int   # words with low similarity / high drift
    
    def report(self) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 70,
            "REGISTER SPECIFICITY CLUSTERING",
            "=" * 70,
            f"Total words analyzed: {len(self.profiles)}",
            f"  Register-consistent: {self.n_consistent}",
            f"  Moderate: {self.n_moderate}",
            f"  Register-divergent: {self.n_divergent}",
        ]
        
        for cluster in sorted(self.clusters, key=lambda c: -c.mean_similarity):
            lines.append(f"\n{'-' * 50}")
            lines.append(f"CLUSTER {cluster.cluster_id}: {cluster.label.upper()}")
            lines.append(f"{'-' * 50}")
            lines.append(f"Words ({len(cluster.words)}): {', '.join(cluster.words[:15])}")
            if len(cluster.words) > 15:
                lines.append(f"  ... and {len(cluster.words) - 15} more")
            lines.append(f"Mean similarity: {cluster.mean_similarity:.3f}")
            lines.append(f"Mean drift: {cluster.mean_drift:.3f}")
            lines.append(f"Pattern: {cluster.characteristic_pattern}")
        
        return "\n".join(lines)


def cluster_by_register_specificity(
    profiles: Dict[str, RegisterProfile],
    n_clusters: int = 3,
    method: str = "threshold"
) -> RegisterSpecificityAnalysis:
    """
    Cluster words by their register specificity patterns.
    
    Args:
        profiles: Dict mapping words to RegisterProfile objects
        n_clusters: Number of clusters (for kmeans method)
        method: "threshold" (fixed boundaries) or "kmeans" (data-driven)
        
    Returns:
        RegisterSpecificityAnalysis with clustering results
    """
    if not profiles:
        return RegisterSpecificityAnalysis(
            clusters=[],
            word_to_cluster={},
            profiles={},
            n_consistent=0,
            n_moderate=0,
            n_divergent=0
        )
    
    # Extract features for each word
    word_features = {}
    for word, profile in profiles.items():
        similarity = profile.overall_register_similarity
        
        # Mean drift across senses
        drifts = [sd.mean_drift for sd in profile.sense_drifts.values()]
        mean_drift = np.mean(drifts) if drifts else 0.0
        
        # Max drift (most divergent sense)
        max_drift = max(drifts) if drifts else 0.0
        
        # Sense specificity (how register-specific are the senses)
        specificities = [sp.specificity for sp in profile.sense_prevalences]
        mean_specificity = np.mean(specificities) if specificities else 0.0
        
        word_features[word] = {
            'similarity': similarity,
            'mean_drift': mean_drift,
            'max_drift': max_drift,
            'specificity': mean_specificity
        }
    
    words = list(word_features.keys())
    
    if method == "threshold":
        # Simple threshold-based clustering
        clusters_dict = {
            0: [],  # register-consistent (high similarity, low drift)
            1: [],  # moderate
            2: []   # register-divergent (low similarity, high drift)
        }
        
        for word, features in word_features.items():
            sim = features['similarity']
            drift = features['mean_drift']
            
            # Thresholds based on typical values
            if sim >= 0.85 and drift <= 0.15:
                clusters_dict[0].append(word)
            elif sim <= 0.75 or drift >= 0.25:
                clusters_dict[2].append(word)
            else:
                clusters_dict[1].append(word)
        
        labels = ["register-consistent", "moderate", "register-divergent"]
        patterns = [
            "High cross-register similarity, senses stable across registers",
            "Moderate variation, some sense drift between registers",
            "Low similarity, significant sense drift or register-specific usage"
        ]
        
    else:  # kmeans
        # Data-driven clustering
        from sklearn.cluster import KMeans
        
        X = np.array([[
            word_features[w]['similarity'],
            word_features[w]['mean_drift'],
            word_features[w]['specificity']
        ] for w in words])
        
        # Normalize features
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_norm)
        
        clusters_dict = {i: [] for i in range(n_clusters)}
        for word, label in zip(words, cluster_labels):
            clusters_dict[label].append(word)
        
        # Label clusters by their characteristics
        labels = []
        patterns = []
        for i in range(n_clusters):
            cluster_words = clusters_dict[i]
            if cluster_words:
                mean_sim = np.mean([word_features[w]['similarity'] for w in cluster_words])
                mean_dr = np.mean([word_features[w]['mean_drift'] for w in cluster_words])
                
                if mean_sim >= 0.85:
                    labels.append("register-consistent")
                    patterns.append("High cross-register similarity")
                elif mean_sim <= 0.75:
                    labels.append("register-divergent")
                    patterns.append("Low cross-register similarity")
                else:
                    labels.append("moderate")
                    patterns.append("Moderate variation")
            else:
                labels.append("empty")
                patterns.append("")
    
    # Build cluster objects
    clusters = []
    word_to_cluster = {}
    
    for cluster_id, cluster_words in clusters_dict.items():
        if not cluster_words:
            continue
        
        mean_sim = np.mean([word_features[w]['similarity'] for w in cluster_words])
        mean_drift = np.mean([word_features[w]['mean_drift'] for w in cluster_words])
        
        cluster = RegisterSpecificityCluster(
            cluster_id=cluster_id,
            label=labels[cluster_id],
            words=sorted(cluster_words, key=lambda w: -word_features[w]['similarity']),
            mean_similarity=mean_sim,
            mean_drift=mean_drift,
            characteristic_pattern=patterns[cluster_id]
        )
        clusters.append(cluster)
        
        for w in cluster_words:
            word_to_cluster[w] = cluster_id
    
    # Count by category
    n_consistent = len([w for w, f in word_features.items() 
                        if f['similarity'] >= 0.85 and f['mean_drift'] <= 0.15])
    n_divergent = len([w for w, f in word_features.items() 
                       if f['similarity'] <= 0.75 or f['mean_drift'] >= 0.25])
    n_moderate = len(word_features) - n_consistent - n_divergent
    
    return RegisterSpecificityAnalysis(
        clusters=clusters,
        word_to_cluster=word_to_cluster,
        profiles=profiles,
        n_consistent=n_consistent,
        n_moderate=n_moderate,
        n_divergent=n_divergent
    )


def analyze_register_specificity(
    words: List[str],
    explorers: Dict[str, 'SenseExplorer'],
    n_senses: int = 2,
    n_clusters: int = 3,
    method: str = "threshold",
    verbose: bool = True
) -> RegisterSpecificityAnalysis:
    """
    Analyze and cluster words by register specificity.
    
    This is the main entry point for register specificity analysis.
    
    Args:
        words: List of words to analyze
        explorers: Dict mapping register names to SenseExplorer instances
        n_senses: Number of senses per word
        n_clusters: Number of clusters (for kmeans)
        method: "threshold" or "kmeans"
        verbose: Print progress
        
    Returns:
        RegisterSpecificityAnalysis with full results
        
    Example:
        ```python
        analysis = analyze_register_specificity(
            words=['bank', 'rock', 'crane', 'apple'],
            explorers={'wiki': se_wiki, 'twitter': se_twitter}
        )
        print(analysis.report())
        ```
    """
    if verbose:
        print(f"Analyzing register specificity for {len(words)} words...")
    
    # Create profiles
    profiles = compare_registers(words, explorers, n_senses=n_senses, verbose=verbose)
    
    if verbose:
        print(f"\nClustering {len(profiles)} words by register specificity...")
    
    # Cluster
    analysis = cluster_by_register_specificity(profiles, n_clusters, method)
    
    if verbose:
        print(f"  Found {len(analysis.clusters)} clusters")
        print(f"  Consistent: {analysis.n_consistent}, Moderate: {analysis.n_moderate}, Divergent: {analysis.n_divergent}")
    
    return analysis


# =============================================================================
# VISUALIZATION (optional, requires matplotlib)
# =============================================================================

def plot_register_specificity(
    profiles: Dict[str, RegisterProfile],
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
    annotate: bool = True,
    title: str = None
):
    """
    Plot words on a 2D plane by register specificity.
    
    X-axis: Register similarity (0 = divergent, 1 = consistent)
    Y-axis: Mean sense drift (0 = stable, 1 = high drift)
    
    Args:
        profiles: Dict mapping words to RegisterProfile objects
        output_path: Path to save figure (optional)
        figsize: Figure size
        annotate: Whether to label points with word names
        title: Plot title (optional)
        
    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for visualization")
        return None
    
    if not profiles:
        warnings.warn("No profiles to plot")
        return None
    
    # Extract coordinates
    words = []
    x_vals = []  # similarity
    y_vals = []  # drift
    
    for word, profile in profiles.items():
        similarity = profile.overall_register_similarity
        
        # Mean drift across senses
        drifts = [sd.mean_drift for sd in profile.sense_drifts.values()]
        mean_drift = np.mean(drifts) if drifts else 0.0
        
        words.append(word)
        x_vals.append(similarity)
        y_vals.append(mean_drift)
    
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by quadrant
    colors = []
    for x, y in zip(x_vals, y_vals):
        if x >= 0.8 and y <= 0.2:
            colors.append('#2ecc71')  # Green: consistent & stable
        elif x <= 0.7 or y >= 0.3:
            colors.append('#e74c3c')  # Red: divergent or drifting
        else:
            colors.append('#f39c12')  # Orange: moderate
    
    # Scatter plot
    scatter = ax.scatter(x_vals, y_vals, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Annotate points
    if annotate:
        for i, word in enumerate(words):
            ax.annotate(
                word,
                (x_vals[i], y_vals[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
    
    # Add quadrant labels
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel('Register Similarity →\n(0 = divergent, 1 = consistent)', fontsize=11)
    ax.set_ylabel('Mean Sense Drift →\n(0 = stable, 1 = high drift)', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title('Register Specificity: Words by Cross-Register Behavior', fontsize=13)
    
    # Set axis limits with padding
    x_margin = (x_vals.max() - x_vals.min()) * 0.1 or 0.1
    y_margin = (y_vals.max() - y_vals.min()) * 0.1 or 0.05
    ax.set_xlim(max(0, x_vals.min() - x_margin), min(1, x_vals.max() + x_margin))
    ax.set_ylim(max(0, y_vals.min() - y_margin), min(1, y_vals.max() + y_margin))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Consistent & Stable'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Moderate'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Divergent / Drifting')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add quadrant annotations
    ax.text(0.95, 0.05, 'Consistent\n& Stable', transform=ax.transAxes, 
            ha='right', va='bottom', fontsize=9, color='#2ecc71', alpha=0.7)
    ax.text(0.05, 0.95, 'Divergent\n& Drifting', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='#e74c3c', alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    return fig


def plot_register_specificity_detailed(
    profiles: Dict[str, RegisterProfile],
    output_path: str = None,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Create a detailed multi-panel visualization of register specificity.
    
    Panel 1: Similarity vs Drift scatter
    Panel 2: Similarity distribution
    Panel 3: Drift distribution by sense
    Panel 4: Word ranking by divergence
    
    Args:
        profiles: Dict mapping words to RegisterProfile objects
        output_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for visualization")
        return None
    
    if not profiles:
        return None
    
    # Extract data
    words = list(profiles.keys())
    similarities = [p.overall_register_similarity for p in profiles.values()]
    
    drifts_by_word = {}
    all_drifts = []
    for word, profile in profiles.items():
        word_drifts = [sd.mean_drift for sd in profile.sense_drifts.values()]
        drifts_by_word[word] = word_drifts
        all_drifts.extend(word_drifts)
    
    mean_drifts = [np.mean(drifts_by_word[w]) if drifts_by_word[w] else 0 for w in words]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Panel 1: Scatter plot
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if s >= 0.8 and d <= 0.2 else 
              '#e74c3c' if s <= 0.7 or d >= 0.3 else 
              '#f39c12' 
              for s, d in zip(similarities, mean_drifts)]
    
    ax1.scatter(similarities, mean_drifts, c=colors, s=80, alpha=0.7, edgecolors='black')
    for i, word in enumerate(words):
        ax1.annotate(word, (similarities[i], mean_drifts[i]), 
                     xytext=(3, 3), textcoords='offset points', fontsize=8)
    ax1.set_xlabel('Register Similarity')
    ax1.set_ylabel('Mean Sense Drift')
    ax1.set_title('Words by Register Specificity')
    ax1.axvline(x=0.8, color='gray', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.2, color='gray', linestyle='--', alpha=0.3)
    
    # Panel 2: Similarity histogram
    ax2 = axes[0, 1]
    ax2.hist(similarities, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(similarities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(similarities):.3f}')
    ax2.set_xlabel('Register Similarity')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Register Similarity')
    ax2.legend()
    
    # Panel 3: Drift by sense (box plot style)
    ax3 = axes[1, 0]
    sorted_words = sorted(words, key=lambda w: -np.mean(drifts_by_word[w]) if drifts_by_word[w] else 0)
    drift_data = [drifts_by_word[w] for w in sorted_words[:15]]  # Top 15
    positions = range(len(drift_data))
    
    bp = ax3.boxplot(drift_data, positions=positions, vert=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_yticklabels(sorted_words[:15], fontsize=8)
    ax3.set_xlabel('Sense Drift')
    ax3.set_title('Drift Distribution by Word (Top 15 Most Divergent)')
    
    # Panel 4: Ranking bar chart
    ax4 = axes[1, 1]
    sorted_by_divergence = sorted(zip(words, similarities, mean_drifts), 
                                   key=lambda x: x[1] - x[2])  # similarity - drift
    
    bar_words = [x[0] for x in sorted_by_divergence[:15]]
    bar_scores = [x[1] - x[2] for x in sorted_by_divergence[:15]]  # divergence score
    bar_colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in bar_scores]
    
    ax4.barh(range(len(bar_words)), bar_scores, color=bar_colors, edgecolor='black')
    ax4.set_yticks(range(len(bar_words)))
    ax4.set_yticklabels(bar_words, fontsize=8)
    ax4.set_xlabel('Consistency Score (similarity - drift)')
    ax4.set_title('Word Ranking by Register Consistency')
    ax4.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved detailed plot to {output_path}")
    
    return fig

def plot_register_profile(
    profile: RegisterProfile,
    output_path: str = None
):
    """
    Visualize a register profile.
    
    Args:
        profile: RegisterProfile to visualize
        output_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not available for visualization")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Sense prevalence
    ax1 = axes[0]
    sense_names = [sp.sense_name for sp in profile.sense_prevalences]
    x = np.arange(len(sense_names))
    width = 0.8 / len(profile.registers)
    
    for i, reg in enumerate(profile.registers):
        values = [sp.prevalence_by_register.get(reg, 0) for sp in profile.sense_prevalences]
        ax1.bar(x + i * width, values, width, label=reg)
    
    ax1.set_xlabel('Sense')
    ax1.set_ylabel('Prevalence')
    ax1.set_title(f"Sense Prevalence: '{profile.word}'")
    ax1.set_xticks(x + width * (len(profile.registers) - 1) / 2)
    ax1.set_xticklabels(sense_names, rotation=45, ha='right')
    ax1.legend()
    
    # 2. Sense drift
    ax2 = axes[1]
    drift_names = list(profile.sense_drifts.keys())
    drift_values = [profile.sense_drifts[n].mean_drift for n in drift_names]
    
    ax2.barh(drift_names, drift_values)
    ax2.set_xlabel('Drift (1 - similarity)')
    ax2.set_title('Sense Drift Between Registers')
    ax2.set_xlim(0, 1)
    
    # 3. Neighbor overlap
    ax3 = axes[2]
    overlap_scores = [rn.overlap_score for rn in profile.register_neighbors.values()]
    sense_labels = list(profile.register_neighbors.keys())
    
    ax3.barh(sense_labels, overlap_scores)
    ax3.set_xlabel('Neighbor Overlap (Jaccard)')
    ax3.set_title('Cross-Register Neighbor Similarity')
    ax3.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig
