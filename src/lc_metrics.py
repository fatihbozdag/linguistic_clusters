#!/usr/bin/env python3
"""
LC Framework Metrics with Dual-Lane Acceptance

Calculates Language Construction metrics for pattern analysis:
- ATP (Average Transition Probability): Replaces invalid MI averaging
- ΔP (Delta P backward): Backward transitional probability
- H_slot (Slot Entropy): Entropy of verb distribution within pattern (lexical productivity)
- Dispersion: Gries's DP for document distribution
- G² test: Log-likelihood statistical significance
- FDR correction: Benjamini-Hochberg False Discovery Rate control (informational)

Dual-Lane Acceptance System:
- Lane 1 (NPMI): Constructional association with G²-based significance
- Lane 2 (H_slot): Lexical productivity with G²-based significance

Either lane can pass for schema acceptance (OR logic).

NOTE: H_slot was previously mislabeled as "IG" (Information Gain). It is actually
the Shannon entropy of verb lemmas within a pattern, measuring slot diversity.
"""

from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import math
import numpy as np


def apply_fdr_correction(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.

    The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR),
    the expected proportion of false positives among rejected hypotheses.

    Args:
        p_values: List of raw p-values
        alpha: Desired FDR level (default 0.05)

    Returns:
        List of FDR-adjusted p-values (same order as input)

    Example:
        >>> raw_p = [0.001, 0.01, 0.03, 0.04, 0.05]
        >>> adjusted_p = apply_fdr_correction(raw_p, alpha=0.05)
        >>> # Adjusted p-values will be higher than raw p-values
    """
    n = len(p_values)
    if n == 0:
        return []

    # Create array with original indices
    indexed_p = [(p, i) for i, p in enumerate(p_values)]

    # Sort by p-value
    indexed_p.sort(key=lambda x: x[0])

    # Apply Benjamini-Hochberg correction
    # adjusted_p[i] = min(p[i] * n / rank[i], 1.0)
    # Then enforce monotonicity from the largest rank down
    adjusted = [0.0] * n

    # Calculate adjusted p-values
    for rank, (p, orig_idx) in enumerate(indexed_p, start=1):
        adjusted_p = p * n / rank
        adjusted[orig_idx] = min(adjusted_p, 1.0)

    # Enforce monotonicity: working backwards, each p-value should be
    # at most as large as the next larger one
    # First, sort by original p-value rank to get monotonic sequence
    sorted_by_rank = sorted(range(n), key=lambda i: p_values[i])

    # Working from largest to smallest p-value
    min_so_far = 1.0
    for i in reversed(sorted_by_rank):
        adjusted[i] = min(adjusted[i], min_so_far)
        min_so_far = adjusted[i]

    return adjusted


def calculate_atp(pattern: str, constructions: List[Dict]) -> float:
    """
    Calculate Average Transition Probability (ATP) at schema level.

    Schema-level ATP = Average of instance-level ATP values.

    Instance ATP uses bidirectional ΔP formula:
    ATP_instance = Average of [(ΔP_forward + ΔP_backward) / 2] for all word pairs

    Args:
        pattern: Pattern string like "past-be,Ø" (TAM,COMP format)
        constructions: List of construction dicts with 'instance_atp' field

    Returns:
        Schema-level ATP score (0.0 to 1.0)

    Example:
        >>> constructions = [
        ...     {'pattern': 'past-be,Ø', 'instance_atp': 0.15},
        ...     {'pattern': 'past-be,Ø', 'instance_atp': 0.18},
        ...     {'pattern': 'pres-be,Ø', 'instance_atp': 0.12},
        ... ]
        >>> atp = calculate_atp('past-be,Ø', constructions)
        >>> # Returns average of 0.15 and 0.18 = 0.165
    """
    # Filter constructions matching this pattern
    pattern_consts = [c for c in constructions if c.get('pattern') == pattern]

    if not pattern_consts:
        return 0.0

    # Average instance-level ATP values
    atp_values = [c.get('instance_atp', 0.0) for c in pattern_consts]

    return sum(atp_values) / len(atp_values)


def calculate_delta_p_backward(pattern: str, constructions: List[Dict]) -> float:
    """
    Calculate Delta P backward (ΔP) at schema level.

    Schema-level ΔP = Average of instance-level ΔP_backward values.

    Instance ΔP_backward measures backward association strength:
    ΔP_backward = P(w1|w2) - P(w1)

    For passives: measures how predictable the auxiliary is given the participle.

    Args:
        pattern: Pattern string like "past-be,Ø" (TAM,COMP format)
        constructions: List of construction dicts with 'instance_dpb' field

    Returns:
        Schema-level ΔP score (-1.0 to 1.0), typically 0.0 to 1.0

    Example:
        >>> constructions = [
        ...     {'pattern': 'past-be,Ø', 'instance_dpb': 0.18},
        ...     {'pattern': 'past-be,Ø', 'instance_dpb': 0.22},
        ...     {'pattern': 'pres-be,Ø', 'instance_dpb': 0.15},
        ... ]
        >>> delta_p = calculate_delta_p_backward('past-be,Ø', constructions)
        >>> # Returns average of 0.18 and 0.22 = 0.20
    """
    # Filter constructions matching this pattern
    pattern_consts = [c for c in constructions if c.get('pattern') == pattern]

    if not pattern_consts:
        return 0.0

    # Average instance-level ΔP_backward values
    dpb_values = [c.get('instance_dpb', 0.0) for c in pattern_consts]

    return sum(dpb_values) / len(dpb_values)


def calculate_slot_entropy(pattern: str, constructions: List[Dict]) -> float:
    """
    Calculate Slot Entropy (H_slot) for lexical productivity at schema level.

    H_slot = Shannon entropy of verb (head_lemma) distribution within this pattern.
    H_slot = -Σ p(verb) * log2(p(verb))

    Higher H_slot = more diverse verbs = more productive pattern.
    H_slot ≥ 1.5 indicates productive construction (many different fillers).

    NOTE: This was previously mislabeled as "Information Gain (IG)". True IG
    would be H(verb) - H(verb|pattern), measuring how much the pattern helps
    predict the verb. This metric measures verb diversity within the slot.

    Args:
        pattern: Pattern string like "past-be,Ø" (TAM,COMP format)
        constructions: List of construction dicts with 'head_lemma' field

    Returns:
        H_slot score (bits), typically 0.0 to ~5.0

    Example:
        >>> constructions = [
        ...     {'pattern': 'past-be,Ø', 'head_lemma': 'write'},
        ...     {'pattern': 'past-be,Ø', 'head_lemma': 'analyze'},
        ...     {'pattern': 'past-be,Ø', 'head_lemma': 'write'},
        ... ]
        >>> h_slot = calculate_slot_entropy('past-be,Ø', constructions)
        >>> # High H_slot if many different verbs, low if dominated by one verb
    """
    # Filter constructions matching this pattern
    pattern_consts = [c for c in constructions if c.get('pattern') == pattern]

    if len(pattern_consts) == 0:
        return 0.0

    # Count verb (head_lemma) frequencies
    verb_counts = Counter()
    for const in pattern_consts:
        verb = const.get('head_lemma', 'unknown')
        verb_counts[verb] += 1

    total = sum(verb_counts.values())

    if total == 0:
        return 0.0

    # Calculate entropy H(verb | pattern)
    entropy = 0.0
    for count in verb_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def calculate_dispersion(pattern: str, constructions: List[Dict]) -> float:
    """
    Calculate Gries's DP (Dispersion) metric.

    Measures how evenly pattern is distributed across documents.
    Lower DP = more evenly distributed = more robust pattern.

    DP ranges from 0 (perfect distribution) to 1 (concentrated in one doc).

    Args:
        pattern: Pattern string
        constructions: List of construction dicts with 'doc_id' field

    Returns:
        DP score (0.0 to 1.0)

    Example:
        >>> disp = calculate_dispersion(pattern, constructions)
        >>> if disp < 0.5:
        ...     print("Well distributed!")
    """
    # Group by document
    doc_counts = Counter()
    total_pattern_count = 0
    all_docs = set()

    for const in constructions:
        doc_id = const.get('doc_id', 'unknown')
        const_pattern = const.get('pattern', '')

        all_docs.add(doc_id)

        if const_pattern == pattern:
            doc_counts[doc_id] += 1
            total_pattern_count += 1

    if total_pattern_count == 0 or len(all_docs) == 0:
        return 1.0  # Maximally dispersed (not present)

    # Total corpus size (all constructions per document)
    corpus_doc_counts = Counter()
    for const in constructions:
        doc_id = const.get('doc_id', 'unknown')
        corpus_doc_counts[doc_id] += 1

    total_corpus_size = sum(corpus_doc_counts.values())

    # Calculate DP
    dp_sum = 0.0
    for doc in all_docs:
        # Expected proportion if evenly distributed
        expected_proportion = corpus_doc_counts[doc] / total_corpus_size

        # Observed proportion
        observed_proportion = doc_counts[doc] / total_pattern_count

        # Absolute difference
        dp_sum += abs(observed_proportion - expected_proportion)

    # DP is half the sum (ranges 0 to 1)
    dp = dp_sum / 2.0

    return dp


def g_squared_test(pattern: str, constructions: List[Dict]) -> Tuple[float, float]:
    """
    Calculate G² (log-likelihood ratio) test using Dunning (1993) method.

    Uses observed marginal frequencies (not uniform distribution) as the
    null hypothesis. This is the standard approach in corpus linguistics.

    The test uses a 2x2 contingency table:
    |              | This Pattern | Other Patterns | Total |
    |--------------|--------------|----------------|-------|
    | Observed     |      O       |     N - O      |   N   |
    | Expected     |      E       |     N - E      |   N   |

    Where E = N / k (expected under uniform distribution across k patterns)

    G² = 2 * Σ O_i * ln(O_i / E_i)

    Reference:
        Dunning, T. (1993). Accurate methods for the statistics of surprise
        and coincidence. Computational Linguistics, 19(1), 61-74.

    Args:
        pattern: Pattern string
        constructions: List of construction dicts

    Returns:
        Tuple of (g_squared, p_value)
        - g_squared: Log-likelihood ratio statistic
        - p_value: Chi-square p-value with df=1

    Example:
        >>> g2, p_val = g_squared_test(pattern, constructions)
        >>> if p_val < 0.05:
        ...     print("Statistically significant!")
    """
    # Count observed frequencies
    total = len(constructions)
    if total == 0:
        return 0.0, 1.0

    # Count pattern occurrences
    pattern_count = sum(1 for c in constructions if c.get('pattern') == pattern)

    if pattern_count == 0:
        return 0.0, 1.0

    # Count all unique patterns for expected frequency calculation
    pattern_counts = Counter(c.get('pattern') for c in constructions)
    n_patterns = len(pattern_counts)

    if n_patterns <= 1:
        return 0.0, 1.0

    # Expected frequency under uniform distribution
    # E_i = N / k where k = number of unique patterns
    expected = total / n_patterns
    other_count = total - pattern_count
    other_expected = total - expected

    # Dunning's G² formula (log-likelihood ratio)
    # G² = 2 * Σ O_i * ln(O_i / E_i)
    # For 2-cell comparison: this pattern vs. others

    g_squared = 0.0

    # Term for this pattern
    if pattern_count > 0 and expected > 0:
        g_squared += pattern_count * math.log(pattern_count / expected)

    # Term for other patterns (aggregate)
    if other_count > 0 and other_expected > 0:
        g_squared += other_count * math.log(other_count / other_expected)

    g_squared *= 2  # Multiply by 2 for G² statistic

    # Calculate p-value using chi-square distribution with df=1
    # Using survival function: P(X > g_squared)
    try:
        from scipy import stats
        p_value = stats.chi2.sf(g_squared, df=1)
    except ImportError:
        # Fallback approximation if scipy not available
        # Wilson-Hilferty approximation for chi-square
        if g_squared <= 0:
            p_value = 1.0
        else:
            # Simple approximation: p ≈ exp(-G²/2) for large G²
            # More accurate: use chi-square CDF approximation
            p_value = math.exp(-g_squared / 2)

    return g_squared, p_value


def g_squared_contingency(o11: int, o12: int, o21: int, o22: int) -> Tuple[float, float]:
    """
    Calculate G² for a 2x2 contingency table (Dunning 1993).

    This is the full contingency table version for collocational analysis.

    Table layout:
    |           | word2 | ~word2 |
    |-----------|-------|--------|
    | word1     |  o11  |  o12   |
    | ~word1    |  o21  |  o22   |

    G² = 2 * Σ O_ij * ln(O_ij / E_ij)

    Args:
        o11: Frequency of word1 + word2
        o12: Frequency of word1 + ~word2
        o21: Frequency of ~word1 + word2
        o22: Frequency of ~word1 + ~word2

    Returns:
        Tuple of (g_squared, p_value)
    """
    # Total
    n = o11 + o12 + o21 + o22
    if n == 0:
        return 0.0, 1.0

    # Row and column totals
    r1 = o11 + o12  # row 1 total
    r2 = o21 + o22  # row 2 total
    c1 = o11 + o21  # col 1 total
    c2 = o12 + o22  # col 2 total

    # Expected frequencies
    e11 = (r1 * c1) / n if n > 0 else 0
    e12 = (r1 * c2) / n if n > 0 else 0
    e21 = (r2 * c1) / n if n > 0 else 0
    e22 = (r2 * c2) / n if n > 0 else 0

    # G² calculation with safe log
    def safe_term(observed, expected):
        if observed > 0 and expected > 0:
            return observed * math.log(observed / expected)
        return 0.0

    g_squared = 2 * (
        safe_term(o11, e11) +
        safe_term(o12, e12) +
        safe_term(o21, e21) +
        safe_term(o22, e22)
    )

    # P-value from chi-square with df=1
    try:
        from scipy import stats
        p_value = stats.chi2.sf(g_squared, df=1)
    except ImportError:
        p_value = math.exp(-g_squared / 2) if g_squared > 0 else 1.0

    return g_squared, p_value


def calculate_npmi_schema_level(constructions: List[Dict]) -> Dict[str, float]:
    """
    Calculate schema-level NPMI between TAM and COMP components.

    CRITICAL: This measures CONSTRUCTIONAL ASSOCIATION at TAM×COMP level,
    NOT at individual pattern level. This is Layer 2 formulaicity (paradigmatic).

    For TAM×COMP combination "pres-be × by_NP":
    - Sums ALL patterns with pres-be AND by_NP (across all SUBJTYPEs)
    - Measures: P(pres-be, by_NP) vs P(pres-be) × P(by_NP)
    - Same NPMI score assigned to all patterns sharing this TAM×COMP

    High NPMI = TAM and COMP prefer to co-occur (constructional preference)
    Low NPMI = TAM and COMP occur independently (no preference)

    This is different from ATP which measures syntagmatic cohesion
    in specific word sequences.

    **CORRECT APPROACH:** Calculate at TAM×COMP level, not pattern level.

    Args:
        constructions: List of PassiveConstruction dicts with keys:
            - pattern: Full pattern string "[SUBJTYPE=X],TAM,COMP"
            - tam: TAM category
            - comp: Complement type

    Returns:
        Dict mapping pattern → NPMI score (-1.0 to 1.0)
        All patterns with same TAM×COMP get same NPMI score.

    Example:
        >>> npmi = calculate_npmi_schema_level(constructions)
        >>> # All patterns with "prog-be,Ø" get the same NPMI score
        >>> print(npmi["[SUBJTYPE=DEF_NP],prog-be,Ø"])  # +0.039
        >>> print(npmi["[SUBJTYPE=PL_NP],prog-be,Ø"])   # +0.039 (same!)
    """
    if len(constructions) == 0:
        return {}

    total = len(constructions)  # Total number of construction tokens

    # Count tokens at TAM×COMP level (aggregate across all SUBJTYPEs)
    tam_comp_counts = Counter()  # (TAM, COMP) → count
    tam_counts = Counter()       # TAM → count
    comp_counts = Counter()      # COMP → count

    # Also track which patterns exist (for output)
    patterns_seen = set()

    for const in constructions:
        pattern = const.get('pattern', '')
        if not pattern:
            continue

        patterns_seen.add(pattern)

        # Parse pattern to get TAM and COMP
        # Pattern format: "TAM,COMP" (2 parts, no SUBJTYPE)
        parts = pattern.split(',')
        if len(parts) < 2:
            continue

        tam = parts[0].strip()
        comp = parts[1].strip()

        # Accumulate counts at TAM×COMP level (not pattern level!)
        tam_comp_counts[(tam, comp)] += 1
        tam_counts[tam] += 1
        comp_counts[comp] += 1

    # Calculate NPMI at TAM×COMP level
    tam_comp_npmi = {}  # (TAM, COMP) → NPMI score

    for (tam, comp), count in tam_comp_counts.items():
        # P(TAM, COMP) - joint probability at TAM×COMP level
        p_joint = count / total

        # P(TAM) and P(COMP) - marginal probabilities
        p_tam = tam_counts[tam] / total
        p_comp = comp_counts[comp] / total

        # Calculate PMI and NPMI
        if p_tam > 0 and p_comp > 0 and p_joint > 0:
            # PMI = log(P(TAM,COMP) / (P(TAM) × P(COMP)))
            pmi = math.log2(p_joint / (p_tam * p_comp))

            # NPMI = PMI / -log(P(TAM,COMP))
            # Normalizes PMI to [-1, 1] range
            npmi = pmi / -math.log2(p_joint)

            tam_comp_npmi[(tam, comp)] = npmi
        else:
            tam_comp_npmi[(tam, comp)] = 0.0

    # Assign TAM×COMP NPMI to all patterns with that combination
    npmi_scores = {}

    for pattern in patterns_seen:
        # Pattern format: "TAM,COMP" (2 parts, no SUBJTYPE)
        parts = pattern.split(',')
        if len(parts) < 2:
            npmi_scores[pattern] = 0.0
            continue

        tam = parts[0].strip()
        comp = parts[1].strip()

        # Look up NPMI for this TAM×COMP combination
        npmi_scores[pattern] = tam_comp_npmi.get((tam, comp), 0.0)

    return npmi_scores


def calculate_lc_metrics(constructions: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate all LC Framework metrics for all patterns.

    Args:
        constructions: List of construction dicts with keys:
            - pattern: Full pattern string
            - tam: TAM category
            - comp: Complement type
            - subjtype: Subject type
            - doc_id: Document identifier

    Returns:
        Dict mapping pattern → metrics dict with keys:
            - n_tokens: Token frequency
            - n_docs: Document frequency
            - atp: Average Transition Probability (Layer 1 - syntagmatic)
            - delta_p: Delta P backward
            - h_slot: Slot Entropy (verb diversity)
            - npmi: Normalized PMI (Layer 2 - paradigmatic, TAM×COMP)
            - dispersion: Gries's DP
            - g_squared: G² statistic
            - p_value: Statistical significance

    Example:
        >>> metrics = calculate_lc_metrics(constructions)
        >>> for pattern, m in sorted(metrics.items(), key=lambda x: x[1]['n_tokens'], reverse=True)[:5]:
        ...     print(f"{pattern}: {m['n_tokens']} tokens, ATP={m['atp']:.3f}, NPMI={m['npmi']:.3f}")
    """
    # Group by pattern
    pattern_constructions = defaultdict(list)
    pattern_docs = defaultdict(set)

    for const in constructions:
        pattern = const.get('pattern')
        if pattern:
            pattern_constructions[pattern].append(const)
            doc_id = const.get('doc_id', 'unknown')
            pattern_docs[pattern].add(doc_id)

    # Calculate schema-level NPMI once for all patterns (NEW)
    npmi_scores = calculate_npmi_schema_level(constructions)

    # Calculate metrics for each pattern
    metrics = {}

    for pattern in pattern_constructions.keys():
        pattern_consts = pattern_constructions[pattern]

        # Basic frequencies
        n_tokens = len(pattern_consts)
        n_docs = len(pattern_docs[pattern])

        # LC Framework metrics
        atp = calculate_atp(pattern, constructions)
        delta_p = calculate_delta_p_backward(pattern, constructions)
        h_slot = calculate_slot_entropy(pattern, constructions)
        dispersion = calculate_dispersion(pattern, constructions)
        g_squared, p_value = g_squared_test(pattern, constructions)

        metrics[pattern] = {
            'n_tokens': n_tokens,
            'n_docs': n_docs,
            'atp': atp,
            'delta_p': delta_p,
            'h_slot': h_slot,
            'npmi': npmi_scores.get(pattern, 0.0),  # schema-level
            'dispersion': dispersion,
            'g_squared': g_squared,
            'p_value': p_value
        }

    # Apply FDR correction to all p-values
    if metrics:
        patterns = list(metrics.keys())
        raw_p_values = [metrics[p]['p_value'] for p in patterns]
        fdr_p_values = apply_fdr_correction(raw_p_values)

        for i, pattern in enumerate(patterns):
            metrics[pattern]['p_value_fdr'] = fdr_p_values[i]

    return metrics


def apply_dual_lane_acceptance(
    metrics: Dict[str, Dict],
    mode: str = 'production'
) -> Tuple[Dict[str, Dict], Dict]:
    """
    Apply dual-lane acceptance with G²-based significance (OR logic).

    VALIDATION APPROACH: Permutation Testing (not FDR)
    ---------------------
    FDR correction assumes independent tests, but TAM×COMP patterns share
    structural dependencies. Permutation testing validates schemas against
    a null model that preserves corpus structure - more appropriate for
    construction mining. See: permutation_test.py

    FDR-corrected p-values (p_value_fdr) are calculated and included in
    output for informational purposes only.

    Layer 1 (NPMI Lane): Constructional Association
    - NPMI ≥ 0.05 (production) or 0.02 (discovery)
    - G² ≥ 3.84 (p < 0.05)
    - Measures: Do TAM and COMP prefer each other? (fixed collocations)

    Layer 2 (H_slot Lane): Lexical Productivity
    - H_slot ≥ 1.5 (slot entropy threshold)
    - G² ≥ 1.0
    - NPMI ≥ 0 (floor to filter negative associations)
    - Measures: How diverse are the verbs? (productive slots)

    EITHER lane can pass (OR logic) - captures both types:
    1. Associative stability (NPMI-strong) = fixed collocations
    2. Productive stability (H_slot-strong) = generative templates

    Args:
        metrics: Dict mapping pattern → metrics dict
        mode: 'production' or 'discovery' (default: 'production')

    Returns:
        Tuple of (accepted_schemas, acceptance_stats)

    Example:
        >>> accepted, stats = apply_dual_lane_acceptance(metrics, mode='production')
        >>> print(f"Accepted: {stats['total_accepted']}/{stats['total_patterns']}")
        >>> print(f"  NPMI-only: {stats['npmi_only_passed']}")
        >>> print(f"  H_slot-only: {stats['h_slot_only_passed']}")
        >>> print(f"  Both: {stats['both_lanes_passed']}")
    """
    # Mode-dependent thresholds
    if mode == 'production':
        MIN_NPMI = 0.05
    else:  # discovery
        MIN_NPMI = 0.02

    # Fixed thresholds (G²-based significance)
    MIN_G2_NPMI_LANE = 3.84  # χ²(1) at p=0.05
    MIN_H_SLOT = 1.5  # Slot entropy threshold for productivity
    MIN_G2_H_SLOT_LANE = 1.0  # More lenient for productivity

    accepted = {}

    stats = {
        'total_patterns': len(metrics),
        'lane1_npmi_passed': 0,
        'lane1_npmi_failed': 0,
        'lane1_g2_failed': 0,
        'lane2_h_slot_passed': 0,
        'lane2_h_slot_failed': 0,
        'lane2_g2_failed': 0,
        'npmi_only_passed': 0,      # Passed NPMI but not H_slot
        'h_slot_only_passed': 0,    # Passed H_slot but not NPMI
        'both_lanes_passed': 0,     # Passed both NPMI and H_slot
        'total_accepted': 0,        # Sum of above three
        'h_slot_filtered_by_npmi_floor': 0,  # H_slot passed but NPMI < 0
        'mode': mode,
        'thresholds': {
            'npmi': MIN_NPMI,
            'g2_npmi': MIN_G2_NPMI_LANE,
            'h_slot': MIN_H_SLOT,
            'g2_h_slot': MIN_G2_H_SLOT_LANE
        }
    }

    for pattern, pattern_metrics in metrics.items():
        npmi = pattern_metrics.get('npmi', 0)
        h_slot = pattern_metrics.get('h_slot', 0)
        g2 = pattern_metrics.get('g_squared', 0)

        # Lane 1: NPMI (Constructional Association)
        # G²-based significance testing
        npmi_pass = npmi >= MIN_NPMI and g2 >= MIN_G2_NPMI_LANE

        if npmi_pass:
            stats['lane1_npmi_passed'] += 1
        elif npmi < MIN_NPMI:
            stats['lane1_npmi_failed'] += 1
        else:
            stats['lane1_g2_failed'] += 1

        # Lane 2: H_slot (Lexical Productivity / Slot Entropy)
        # G²-based significance with NPMI floor
        h_slot_pass = h_slot >= MIN_H_SLOT and g2 >= MIN_G2_H_SLOT_LANE

        # NPMI floor for H_slot lane: require non-negative association
        # This filters out patterns where TAM and COMP are negatively associated
        if h_slot_pass and npmi < 0:
            h_slot_pass = False
            stats['h_slot_filtered_by_npmi_floor'] += 1

        if h_slot_pass:
            stats['lane2_h_slot_passed'] += 1
        elif h_slot < MIN_H_SLOT:
            stats['lane2_h_slot_failed'] += 1
        else:
            stats['lane2_g2_failed'] += 1

        # OR LOGIC: Accept if EITHER lane passes
        if npmi_pass or h_slot_pass:
            accepted[pattern] = pattern_metrics
            stats['total_accepted'] += 1

            # Track which lane(s) passed
            if npmi_pass and h_slot_pass:
                stats['both_lanes_passed'] += 1
            elif npmi_pass:
                stats['npmi_only_passed'] += 1
            elif h_slot_pass:
                stats['h_slot_only_passed'] += 1

    return accepted, stats


def print_dual_lane_statistics(stats: Dict):
    """Print dual-lane acceptance statistics."""
    print("\n" + "="*60)
    print("DUAL-LANE FORMULAICITY VALIDATION (OR LOGIC + G²)")
    print("="*60)
    print(f"Mode: {stats['mode'].upper()}")
    print(f"Validation: Permutation testing (see permutation_test.py)")
    print()
    print("Layer 1 (Constructional Association - NPMI):")
    print(f"  NPMI threshold: ≥ {stats['thresholds']['npmi']}")
    print(f"  G² threshold: ≥ {stats['thresholds']['g2_npmi']}")
    print(f"  Passed: {stats['lane1_npmi_passed']}")
    print(f"  Failed (low NPMI): {stats['lane1_npmi_failed']}")
    print(f"  Failed (low G²): {stats.get('lane1_g2_failed', 0)}")
    print()
    print("Layer 2 (Lexical Productivity - H_slot):")
    print(f"  H_slot threshold: ≥ {stats['thresholds']['h_slot']}")
    print(f"  G² threshold: ≥ {stats['thresholds']['g2_h_slot']}")
    print(f"  NPMI floor: ≥ 0")
    print(f"  Passed: {stats['lane2_h_slot_passed']}")
    print(f"  Filtered (NPMI < 0): {stats['h_slot_filtered_by_npmi_floor']}")
    print(f"  Failed (low H_slot): {stats['lane2_h_slot_failed']}")
    print(f"  Failed (low G²): {stats.get('lane2_g2_failed', 0)}")
    print()
    print("✅ ACCEPTED SCHEMAS (EITHER lane passes):")
    print(f"   NPMI-only (fixed collocations): {stats['npmi_only_passed']}")
    print(f"   H_slot-only (productive slots): {stats['h_slot_only_passed']}")
    print(f"   Both (highly formulaic): {stats['both_lanes_passed']}")
    print(f"   Total accepted: {stats['total_accepted']} ({stats['total_accepted']}/{stats['total_patterns']} = "
          f"{stats['total_accepted']/stats['total_patterns']*100:.1f}%)")
    print("="*60)


# ============================================================================
# Instance-Level Prefilter Metrics
# ============================================================================

def build_corpus_statistics(constructions: List[Dict]) -> Dict:
    """
    Build corpus-wide word and bigram frequencies from passive constructions.

    CRITICAL: Only count words within passive constructions,
    not the entire corpus text!

    This is because we're measuring formulaicity WITHIN the passive
    construction domain, not general language.

    Args:
        constructions: All extracted constructions (before prefiltering)

    Returns:
        Dict with:
        - unigram_counts: Counter of individual words
        - bigram_counts: Counter of word pairs
        - follower_counts: Dict[pattern] -> Counter of following words
        - total_words: Total word count
    """
    from collections import defaultdict

    stats = {
        'unigram_counts': Counter(),
        'bigram_counts': Counter(),
        'follower_counts': defaultdict(Counter),
        'total_words': 0
    }

    for const in constructions:
        # Get auxiliary chain words
        aux_chain = const.get('aux_chain', [])

        if not aux_chain:
            continue

        # Count unigrams (individual words)
        for word in aux_chain:
            word_lower = word.lower()
            stats['unigram_counts'][word_lower] += 1
            stats['total_words'] += 1

        # Count bigrams (word pairs)
        for i in range(len(aux_chain) - 1):
            w1 = aux_chain[i].lower()
            w2 = aux_chain[i + 1].lower()
            bigram = f"{w1} {w2}"
            stats['bigram_counts'][bigram] += 1

        # Count followers (word after construction)
        pattern = const.get('pattern', '')
        following = const.get('following_word')
        if pattern and following:  # following can be None
            following = following.lower()
            stats['follower_counts'][pattern][following] += 1

    return stats


def _calculate_delta_p_forward(w1: str, w2: str, corpus_stats: Dict) -> float:
    """
    Calculate ΔP_forward for a word pair.

    ΔP_forward = P(w2|w1) - P(w2)

    Measures how much w1 predicts w2 above baseline.

    Args:
        w1: First word in pair
        w2: Second word in pair
        corpus_stats: Dict with unigram_counts, bigram_counts, total_words

    Returns:
        float: ΔP_forward score
    """
    # P(w2|w1)
    bigram = f"{w1} {w2}"
    bigram_count = corpus_stats['bigram_counts'].get(bigram, 0)
    w1_count = corpus_stats['unigram_counts'].get(w1, 0)
    p_w2_given_w1 = bigram_count / w1_count if w1_count > 0 else 0.0

    # P(w2)
    total_words = corpus_stats['total_words']
    w2_count = corpus_stats['unigram_counts'].get(w2, 0)
    p_w2 = w2_count / total_words if total_words > 0 else 0.0

    return p_w2_given_w1 - p_w2


def _calculate_delta_p_backward_pair(w1: str, w2: str, corpus_stats: Dict) -> float:
    """
    Calculate ΔP_backward for a word pair.

    ΔP_backward = P(w1|w2) - P(w1)

    Measures how much w2 predicts w1 above baseline.

    Args:
        w1: First word in pair
        w2: Second word in pair
        corpus_stats: Dict with unigram_counts, bigram_counts, total_words

    Returns:
        float: ΔP_backward score
    """
    # P(w1|w2)
    bigram = f"{w1} {w2}"
    bigram_count = corpus_stats['bigram_counts'].get(bigram, 0)
    w2_count = corpus_stats['unigram_counts'].get(w2, 0)
    p_w1_given_w2 = bigram_count / w2_count if w2_count > 0 else 0.0

    # P(w1)
    total_words = corpus_stats['total_words']
    w1_count = corpus_stats['unigram_counts'].get(w1, 0)
    p_w1 = w1_count / total_words if total_words > 0 else 0.0

    return p_w1_given_w2 - p_w1


def calculate_instance_atp(construction: Dict, corpus_stats: Dict) -> float:
    """
    Calculate ATP for single construction instance.

    CORRECT FORMULA (bidirectional):
    ATP = Average of [(ΔP_forward + ΔP_backward) / 2] for all adjacent pairs

    This captures bidirectional transitional probability, measuring mutual
    predictability between words in the sequence.

    For "has been analyzed":
    - Pair 1: (has, been)
      - ΔP_fwd: P(been|has) - P(been)
      - ΔP_bwd: P(has|been) - P(has)
      - ATP_pair1 = (ΔP_fwd + ΔP_bwd) / 2

    - Pair 2: (been, analyzed)
      - ΔP_fwd: P(analyzed|been) - P(analyzed)
      - ΔP_bwd: P(been|analyzed) - P(been)
      - ATP_pair2 = (ΔP_fwd + ΔP_bwd) / 2

    - ATP_instance = (ATP_pair1 + ATP_pair2) / 2

    Args:
        construction: Single construction dict with 'aux_chain' field
        corpus_stats: Dict with unigram_counts, bigram_counts, total_words

    Returns:
        float: ATP score, threshold is 0.10
    """
    aux_chain = construction.get('aux_chain', [])

    if len(aux_chain) < 2:
        return 0.0

    pair_atp_scores = []

    # Calculate ATP for each adjacent pair
    for i in range(len(aux_chain) - 1):
        w1 = aux_chain[i].lower()
        w2 = aux_chain[i + 1].lower()

        # Calculate both directions
        delta_p_fwd = _calculate_delta_p_forward(w1, w2, corpus_stats)
        delta_p_bwd = _calculate_delta_p_backward_pair(w1, w2, corpus_stats)

        # Average them for this pair (bidirectional ATP)
        pair_atp = (delta_p_fwd + delta_p_bwd) / 2
        pair_atp_scores.append(pair_atp)

    # Average across all pairs in the construction
    return sum(pair_atp_scores) / len(pair_atp_scores) if pair_atp_scores else 0.0


def calculate_instance_delta_p_backward(construction: Dict, corpus_stats: Dict) -> float:
    """
    Calculate ΔP_backward for single construction instance.

    ΔP_backward = P(w1|w2) - P(w1)

    For passives: measures how predictable the auxiliary is given the participle.
    Key metric: if participle strongly predicts auxiliary → high ΔP → formulaic

    Example: "been analyzed"
    - P(been|analyzed) = how often "been" precedes "analyzed"
    - P(been) = base rate of "been" in corpus
    - If difference is high → strong backward association

    Args:
        construction: Single construction dict with 'aux_chain' field
        corpus_stats: Dict with unigram_counts, bigram_counts, total_words

    Returns:
        float: ΔP_backward score, threshold is 0.10
    """
    aux_chain = construction.get('aux_chain', [])

    if len(aux_chain) < 2:
        return 0.0

    # For passives: focus on final transition (last_aux → participle)
    # e.g., "been" → "analyzed"
    w1 = aux_chain[-2].lower() if len(aux_chain) >= 2 else aux_chain[0].lower()
    w2 = aux_chain[-1].lower()

    # P(w1|w2) - backward conditional probability
    bigram = f"{w1} {w2}"
    bigram_count = corpus_stats['bigram_counts'].get(bigram, 0)
    w2_count = corpus_stats['unigram_counts'].get(w2, 0)
    p_w1_given_w2 = bigram_count / w2_count if w2_count > 0 else 0.0

    # P(w1) - base rate
    total_words = corpus_stats['total_words']
    w1_count = corpus_stats['unigram_counts'].get(w1, 0)
    p_w1 = w1_count / total_words if total_words > 0 else 0.0

    return p_w1_given_w2 - p_w1


def calculate_instance_boundary_entropy(construction: Dict, corpus_stats: Dict) -> float:
    """
    Calculate boundary entropy (H_r) for single construction instance.

    H_r = -Σ P(word_after) × log₂(P(word_after))

    Measures predictability of right boundary (what comes after construction).
    Lower entropy = more formulaic/fixed boundary = more predictable.

    Example: If "is analyzed" is always followed by "by/in/for"
    → low entropy (predictable) → formulaic

    Args:
        construction: Single construction dict with 'following_word'
        corpus_stats: Dict with follower_counts per pattern

    Returns:
        float: Boundary entropy in bits, threshold is 2.8
    """
    # Get the pattern to look up followers
    pattern = construction.get('pattern', '')
    if not pattern:
        return float('inf')

    # Get distribution of words that follow this pattern
    follower_counts = corpus_stats['follower_counts'].get(pattern, {})
    total_followers = sum(follower_counts.values())

    if total_followers == 0:
        return float('inf')  # No boundary info = uncertain

    # Calculate entropy
    entropy = 0.0
    for word, count in follower_counts.items():
        p = count / total_followers
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


# ============================================================================
# Testing
# ============================================================================

def test_lc_metrics():
    """Test LC metrics functions."""
    print("="*70)
    print("Testing LC Metrics Module")
    print("="*70)

    # Create test data
    test_constructions = [
        {'pattern': '[SUBJTYPE=DEF_NP],past-be,Ø', 'tam': 'past-be', 'comp': 'Ø',
         'subjtype': 'DEF_NP', 'doc_id': 'doc1'},
        {'pattern': '[SUBJTYPE=DEF_NP],past-be,Ø', 'tam': 'past-be', 'comp': 'Ø',
         'subjtype': 'DEF_NP', 'doc_id': 'doc1'},
        {'pattern': '[SUBJTYPE=DEF_NP],past-be,Ø', 'tam': 'past-be', 'comp': 'Ø',
         'subjtype': 'DEF_NP', 'doc_id': 'doc2'},
        {'pattern': '[SUBJTYPE=DEF_NP],past-be,by_NP', 'tam': 'past-be', 'comp': 'by_NP',
         'subjtype': 'DEF_NP', 'doc_id': 'doc1'},
        {'pattern': '[SUBJTYPE=DEF_NP],pres-be,Ø', 'tam': 'pres-be', 'comp': 'Ø',
         'subjtype': 'DEF_NP', 'doc_id': 'doc2'},
        {'pattern': '[SUBJTYPE=PRON],past-be,Ø', 'tam': 'past-be', 'comp': 'Ø',
         'subjtype': 'PRON', 'doc_id': 'doc3'},
    ]

    print("\n1. Testing ATP calculation:")
    pattern1 = '[SUBJTYPE=DEF_NP],past-be,Ø'
    atp = calculate_atp(pattern1, test_constructions)
    print(f"   Pattern: {pattern1}")
    print(f"   ATP: {atp:.4f}")
    print(f"   ✓ ATP in valid range (0-1): {0 <= atp <= 1}")

    print("\n2. Testing Delta P calculation:")
    delta_p = calculate_delta_p_backward(pattern1, test_constructions)
    print(f"   ΔP: {delta_p:.4f}")
    print(f"   ✓ ΔP >= 0.15 (formulaic): {delta_p >= 0.15}")

    print("\n3. Testing Slot Entropy:")
    h_slot = calculate_slot_entropy(pattern1, test_constructions)
    print(f"   H_slot: {h_slot:.4f}")
    print(f"   ✓ H_slot >= 1.5 (productive): {h_slot >= 1.5}")

    print("\n4. Testing Dispersion:")
    disp = calculate_dispersion(pattern1, test_constructions)
    print(f"   Dispersion: {disp:.4f}")
    print(f"   ✓ Well distributed (< 0.5): {disp < 0.5}")

    print("\n5. Testing G² test:")
    g2, p_val = g_squared_test(pattern1, test_constructions)
    print(f"   G²: {g2:.4f}")
    print(f"   p-value: {p_val:.4f}")

    print("\n6. Testing full metrics calculation:")
    metrics = calculate_lc_metrics(test_constructions)
    print(f"   Total patterns: {len(metrics)}")
    for pattern, m in sorted(metrics.items(), key=lambda x: x[1]['n_tokens'], reverse=True):
        print(f"   {pattern}:")
        print(f"     n_tokens={m['n_tokens']}, n_docs={m['n_docs']}, ATP={m['atp']:.3f}")

    print("\n7. Testing double-lock acceptance:")
    accepted, stats = apply_dual_lane_acceptance(metrics, mode='discovery')
    print(f"   Mode: {stats['mode']}")
    print(f"   Total: {stats['total_patterns']}, Accepted: {stats['both_lanes_passed']}")
    print(f"   Layer 1 pass (NPMI): {stats['lane1_npmi_passed']}, Layer 2 pass (H_slot): {stats['lane2_h_slot_passed']}")

    print("\n8. Testing dual-lane statistics printing:")
    print_dual_lane_statistics(stats)

    print("\n✓ LC metrics module ready!")


if __name__ == "__main__":
    test_lc_metrics()
